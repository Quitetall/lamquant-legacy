#!/usr/bin/env python3
"""A/B benchmark for 7 proposed LML compression improvements.

Each improvement modifies ONE stage of the production pipeline.
All variants roundtrip bit-exact. Reports CR and delta vs baseline.

Usage:
    python scripts/benchmark_improvements.py /path/to/q31_events/ --max-files 20 --max-windows 10
"""
import argparse
import os
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Import production codec primitives
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lamquant_codec.ops.lifting import forward_nlevel_int, inverse_nlevel_int
from lamquant_codec.ops.lpc import (
    analyze_channel as lpc_analyze_channel,
    analyze_int as lpc_analyze_int,
    synthesize_int as lpc_synthesize_int,
    Q_LPC,
)
from lamquant_codec.ops.bias import cancel_jit as bias_cancel, restore_jit as bias_restore
from lamquant_codec.ops.golomb import (
    encode_dense,
    decode_dense,
    compute_adaptive_k,
    zigzag_encode,
    zigzag_decode,
    BitWriter,
    BitReader,
)
from lamquant_codec.ops.constants import BIAS_CTX_LEN

# Production defaults
N_LEVELS = 3
WINDOW_T = 2500


# ===================================================================
# LPC schedule (mirrors production)
# ===================================================================

def _sub_lpc_schedule(n_levels: int) -> dict:
    sched = {f'l{n_levels}_approx': 1}
    for lvl in range(n_levels, 0, -1):
        if lvl == n_levels:
            sched[f'l{lvl}_detail'] = 1
        elif lvl == n_levels - 1:
            sched[f'l{lvl}_detail'] = 2
        else:
            sched[f'l{lvl}_detail'] = 3
    return sched


def _subband_keys(n_levels: int) -> list:
    return [f'l{n_levels}_approx'] + \
           [f'l{lvl}_detail' for lvl in range(n_levels, 0, -1)]


# ===================================================================
# Improvement #1: adaptive_k_block
# Splits each subband into blocks of 64 and encodes each block with
# its own k and GR bitstream.
# Format per block: [k:u8][n_block:u16 LE][bitstream]
# ===================================================================

BLOCK_SIZE = 64


def encode_adaptive_k_block(data: np.ndarray) -> bytes:
    """GR-encode with per-block adaptive k (block size = 64)."""
    data = np.asarray(data, dtype=np.int64).ravel()
    n = len(data)
    if n == 0:
        return encode_dense(data)

    parts = []
    for start in range(0, n, BLOCK_SIZE):
        block = data[start:start + BLOCK_SIZE]
        parts.append(encode_dense(block))

    # Prefix: total number of blocks (uint16) so decoder knows when to stop
    n_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    header = struct.pack('<HI', n_blocks, n)  # n_blocks, total_samples
    return header + b''.join(parts)


def decode_adaptive_k_block(data_bytes, offset=0):
    """Decode per-block adaptive k GR stream."""
    # Normalise to bytes for header parsing and uint8 array for decode_dense
    if isinstance(data_bytes, np.ndarray):
        raw = bytes(data_bytes)
        data_arr = data_bytes
    else:
        raw = data_bytes if isinstance(data_bytes, (bytes, bytearray)) else bytes(data_bytes)
        data_arr = np.frombuffer(raw, dtype=np.uint8).copy()

    pos = offset
    n_blocks, total_samples = struct.unpack('<HI', raw[pos:pos + 6])
    pos += 6
    parts = []
    remaining = total_samples

    for _ in range(n_blocks):
        block_n = min(BLOCK_SIZE, remaining)
        decoded, consumed = decode_dense(data_arr, pos)
        assert len(decoded) == block_n, f"Block decode mismatch: {len(decoded)} vs {block_n}"
        parts.append(decoded)
        pos += consumed
        remaining -= block_n

    result = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
    bytes_consumed = pos - offset
    return result, bytes_consumed


# ===================================================================
# Improvement #2: rle_zeros
# Before GR encoding, separate into (zero_run_length, non_zero_values).
# Format: repeated segments of [n_zeros:u16][n_nonzero:u16][GR of non-zeros]
# Terminated when all samples consumed.
# ===================================================================

def encode_rle_zeros(data: np.ndarray) -> bytes:
    """RLE zero separation + GR encode of non-zero segments."""
    data = np.asarray(data, dtype=np.int64).ravel()
    n = len(data)
    if n == 0:
        return struct.pack('<I', 0)  # total_samples = 0

    parts = [struct.pack('<I', n)]  # total_samples header
    i = 0
    while i < n:
        # Count zeros
        z_start = i
        while i < n and data[i] == 0:
            i += 1
        n_zeros = i - z_start

        # Count/collect non-zeros
        nz_start = i
        while i < n and data[i] != 0:
            i += 1
        n_nonzero = i - nz_start

        # Segment header
        parts.append(struct.pack('<HH', n_zeros, n_nonzero))
        if n_nonzero > 0:
            parts.append(encode_dense(data[nz_start:nz_start + n_nonzero]))

    return b''.join(parts)


def decode_rle_zeros(data_bytes, offset=0):
    """Decode RLE zero-separated GR stream."""
    if isinstance(data_bytes, np.ndarray):
        raw = bytes(data_bytes)
    else:
        raw = data_bytes
    pos = offset
    total_samples, = struct.unpack('<I', raw[pos:pos + 4])
    pos += 4
    if total_samples == 0:
        return np.array([], dtype=np.int64), pos - offset

    if isinstance(data_bytes, np.ndarray):
        data_arr = data_bytes
    else:
        data_arr = np.frombuffer(raw, dtype=np.uint8).copy()

    result = np.zeros(total_samples, dtype=np.int64)
    out_i = 0
    while out_i < total_samples:
        n_zeros, n_nonzero = struct.unpack('<HH', raw[pos:pos + 4])
        pos += 4
        out_i += n_zeros  # skip zeros (already zero-initialized)
        if n_nonzero > 0:
            decoded, consumed = decode_dense(data_arr, pos)
            assert len(decoded) == n_nonzero
            result[out_i:out_i + n_nonzero] = decoded
            pos += consumed
        out_i += n_nonzero

    return result, pos - offset


# ===================================================================
# Improvement #6: delta_lpc_coeffs
# For channels 1..C-1, transmit coeffs[ch] - coeffs[ch-1] and GR-encode
# the deltas. Channel 0 transmits absolute coefficients.
# ===================================================================

def encode_delta_coeffs(coeffs_q27: np.ndarray) -> bytes:
    """GR-encode Q27 coefficients as int32 raw bytes."""
    # The coefficients are already small int32s, just use raw bytes
    return coeffs_q27.astype(np.int32).tobytes()


def encode_delta_coeffs_gr(delta_q27: np.ndarray) -> bytes:
    """GR-encode Q27 coefficient deltas."""
    return encode_dense(delta_q27.astype(np.int64))


def decode_delta_coeffs_gr(data_bytes, offset=0):
    """Decode GR-encoded Q27 coefficient deltas."""
    decoded, consumed = decode_dense(
        np.frombuffer(data_bytes, dtype=np.uint8).copy()
        if isinstance(data_bytes, (bytes, bytearray)) else data_bytes,
        offset
    )
    return decoded.astype(np.int32), consumed


# ===================================================================
# Experimental compress/decompress
# ===================================================================

# Flag bits for enabled improvements
FLAG_ADAPTIVE_K_BLOCK = 1 << 0   # #1
FLAG_RLE_ZEROS        = 1 << 1   # #2
FLAG_LPC_ORDER_6      = 1 << 2   # #3
FLAG_CROSS_CHANNEL    = 1 << 3   # #4
FLAG_SUBBAND_REORDER  = 1 << 4   # #5
FLAG_DELTA_LPC_COEFFS = 1 << 5   # #6
FLAG_BIAS_CTX_32      = 1 << 6   # #7


def compress_experimental(signal: np.ndarray, flags: int = 0) -> bytes:
    """Compress [C, T] int signal with toggleable improvements.

    Mirrors the production pipeline but with 7 toggleable modifications.
    Returns a custom binary blob (NOT production LML format).
    """
    n_ch, T = signal.shape
    n_levels = N_LEVELS
    signal_int = np.round(signal).astype(np.int64)

    # Adaptive n_levels for short signals
    while T < (4 * (1 << n_levels)) and n_levels > 0:
        n_levels -= 1

    sub_lpc = _sub_lpc_schedule(n_levels)
    subband_keys = _subband_keys(n_levels)

    # Determine bias context length
    bias_ctx = 32 if (flags & FLAG_BIAS_CTX_32) else BIAS_CTX_LEN

    # Override: improvement #3 — LPC order 6 for l1_detail
    if flags & FLAG_LPC_ORDER_6:
        sub_lpc['l1_detail'] = 6

    # Step 1: lifting per channel
    all_subs = []
    for ch in range(n_ch):
        subs = forward_nlevel_int(signal_int[ch], n_levels)
        all_subs.append(subs)

    # Step 2: improvement #4 — cross-channel differencing (after lifting, before LPC)
    if flags & FLAG_CROSS_CHANNEL:
        for key in subband_keys:
            for ch in range(n_ch - 1, 0, -1):
                all_subs[ch][key] = all_subs[ch][key] - all_subs[ch - 1][key]

    # Step 3: LPC + bias cancel + encode
    lpc_parts = []
    payload_parts = []

    # Track previous channel's coefficients for delta encoding
    prev_coeffs_by_key: Dict[str, np.ndarray] = {}

    if flags & FLAG_SUBBAND_REORDER:
        # Improvement #5: group by subband level, encode all channels together
        for key in subband_keys:
            order = sub_lpc.get(key, 4)
            group_residuals = []
            for ch in range(n_ch):
                sub_data = all_subs[ch][key]
                effective_order = order
                if len(sub_data) < effective_order * 4:
                    effective_order = max(1, len(sub_data) // 4)
                acl = min(256, max(effective_order + 1, len(sub_data) // 2))
                if len(sub_data) <= effective_order or len(sub_data) < 3 or acl <= effective_order:
                    effective_order = 0
                    coeffs_q27 = np.array([], dtype=np.int32)
                    residual = sub_data.astype(np.int64)
                else:
                    coeffs_f, _ = lpc_analyze_channel(
                        sub_data.astype(np.float64), order=effective_order,
                        autocorr_len=acl)
                    coeffs_q27, residual = lpc_analyze_int(sub_data, coeffs_f, effective_order)

                corrected = bias_cancel(residual.astype(np.int64), bias_ctx)

                # LPC metadata
                if (flags & FLAG_DELTA_LPC_COEFFS) and ch > 0 and key in prev_coeffs_by_key:
                    prev = prev_coeffs_by_key[key]
                    if len(prev) == len(coeffs_q27) and len(coeffs_q27) > 0:
                        delta = coeffs_q27.astype(np.int64) - prev.astype(np.int64)
                        lpc_parts.append(bytes([effective_order | 0x80]))  # flag: delta mode
                        lpc_parts.append(encode_delta_coeffs_gr(delta))
                    else:
                        lpc_parts.append(bytes([effective_order]))
                        lpc_parts.append(coeffs_q27.astype(np.int32).tobytes())
                else:
                    lpc_parts.append(bytes([effective_order]))
                    lpc_parts.append(coeffs_q27.astype(np.int32).tobytes())

                if len(coeffs_q27) > 0:
                    prev_coeffs_by_key[key] = coeffs_q27.copy()

                group_residuals.append(corrected)

            # Concatenate all channels' residuals for this subband level
            combined = np.concatenate(group_residuals)

            if flags & FLAG_ADAPTIVE_K_BLOCK:
                payload_parts.append(encode_adaptive_k_block(combined))
            elif flags & FLAG_RLE_ZEROS:
                payload_parts.append(encode_rle_zeros(combined))
            else:
                payload_parts.append(encode_dense(combined))
    else:
        # Standard channel-major ordering
        for ch in range(n_ch):
            for key in subband_keys:
                sub_data = all_subs[ch][key]
                order = sub_lpc.get(key, 4)
                if len(sub_data) < order * 4:
                    order = max(1, len(sub_data) // 4)
                acl = min(256, max(order + 1, len(sub_data) // 2))
                if len(sub_data) <= order or len(sub_data) < 3 or acl <= order:
                    order = 0
                    coeffs_q27 = np.array([], dtype=np.int32)
                    residual = sub_data.astype(np.int64)
                else:
                    coeffs_f, _ = lpc_analyze_channel(
                        sub_data.astype(np.float64), order=order,
                        autocorr_len=acl)
                    coeffs_q27, residual = lpc_analyze_int(sub_data, coeffs_f, order)

                corrected = bias_cancel(residual.astype(np.int64), bias_ctx)

                # LPC metadata
                if (flags & FLAG_DELTA_LPC_COEFFS) and ch > 0 and key in prev_coeffs_by_key:
                    prev = prev_coeffs_by_key[key]
                    if len(prev) == len(coeffs_q27) and len(coeffs_q27) > 0:
                        delta = coeffs_q27.astype(np.int64) - prev.astype(np.int64)
                        lpc_parts.append(bytes([order | 0x80]))  # flag: delta mode
                        lpc_parts.append(encode_delta_coeffs_gr(delta))
                    else:
                        lpc_parts.append(bytes([order]))
                        lpc_parts.append(coeffs_q27.astype(np.int32).tobytes())
                else:
                    lpc_parts.append(bytes([order]))
                    lpc_parts.append(coeffs_q27.astype(np.int32).tobytes())

                if len(coeffs_q27) > 0:
                    prev_coeffs_by_key[key] = coeffs_q27.copy()

                # Entropy encode
                if flags & FLAG_ADAPTIVE_K_BLOCK:
                    payload_parts.append(encode_adaptive_k_block(corrected))
                elif flags & FLAG_RLE_ZEROS:
                    payload_parts.append(encode_rle_zeros(corrected))
                else:
                    payload_parts.append(encode_dense(corrected))

    lpc_meta = b''.join(lpc_parts)
    subband_payload = b''.join(payload_parts)

    # Pack a minimal header: magic + dimensions + flags + lengths
    header = struct.pack('<4sHHBBHII',
                         b'EXP1',
                         n_ch, T,
                         n_levels, 0,  # reserved
                         flags,
                         len(lpc_meta),
                         len(subband_payload))
    return header + lpc_meta + subband_payload


def decompress_experimental(data: bytes, flags: int = 0) -> np.ndarray:
    """Decompress experimental blob back to [C, T] int64."""
    HDR_SIZE = 20  # 4+2+2+1+1+2+4+4
    pos = 0
    magic, n_ch, T, n_levels, _, exp_flags, lpc_len, sub_len = \
        struct.unpack('<4sHHBBHII', data[pos:pos + HDR_SIZE])
    assert magic == b'EXP1', f"Bad magic: {magic!r}"
    assert exp_flags == flags, f"Flag mismatch: header={exp_flags}, arg={flags}"
    pos += HDR_SIZE

    lpc_data_raw = data[pos:pos + lpc_len]
    pos += lpc_len

    # Convert payload to uint8 array for decode_dense
    payload_start = pos
    data_arr = np.frombuffer(data, dtype=np.uint8).copy()

    sub_lpc = _sub_lpc_schedule(n_levels)
    subband_keys = _subband_keys(n_levels)

    bias_ctx = 32 if (flags & FLAG_BIAS_CTX_32) else BIAS_CTX_LEN

    if flags & FLAG_LPC_ORDER_6:
        sub_lpc['l1_detail'] = 6

    # Parse LPC metadata
    lpc_entries = []  # list of (order, coeffs_q27) or (order, delta_encoded_bytes)
    lpc_pos = 0
    prev_coeffs_by_key: Dict[str, np.ndarray] = {}

    def _read_lpc_entry(key_hint):
        nonlocal lpc_pos
        order_byte = lpc_data_raw[lpc_pos]
        lpc_pos_start = lpc_pos
        lpc_pos += 1
        is_delta = bool(order_byte & 0x80)
        order = order_byte & 0x7F

        if order == 0:
            coeffs_q27 = np.array([], dtype=np.int32)
        elif is_delta and (flags & FLAG_DELTA_LPC_COEFFS):
            # GR-encoded delta coefficients
            delta, consumed = decode_delta_coeffs_gr(
                lpc_data_raw, lpc_pos)
            lpc_pos += consumed
            prev = prev_coeffs_by_key.get(key_hint, np.zeros(order, dtype=np.int32))
            coeffs_q27 = (prev.astype(np.int64) + delta[:order].astype(np.int64)).astype(np.int32)
        else:
            coeffs_q27 = np.frombuffer(
                lpc_data_raw[lpc_pos:lpc_pos + order * 4],
                dtype=np.int32).copy()
            lpc_pos += order * 4

        if len(coeffs_q27) > 0:
            prev_coeffs_by_key[key_hint] = coeffs_q27.copy()
        return order, coeffs_q27

    # Decode payload
    if flags & FLAG_SUBBAND_REORDER:
        # Subband-major: for each key, all channels packed together
        all_subs = [{} for _ in range(n_ch)]

        for key in subband_keys:
            # Read LPC metadata for each channel
            channel_lpc = []
            for ch in range(n_ch):
                order, coeffs_q27 = _read_lpc_entry(key)
                channel_lpc.append((order, coeffs_q27))

            # Compute expected subband length per channel
            # (all channels produce the same length for a given key)
            # We need to figure out the subband length.
            # For a 2500-sample signal with 3-level lifting:
            #   l3_approx: 313, l3_detail: 312, l2_detail: 625, l1_detail: 1250
            # We compute by doing a forward pass on a dummy signal.
            dummy = np.zeros(T, dtype=np.int64)
            dummy_subs = forward_nlevel_int(dummy, n_levels)
            sub_len_per_ch = len(dummy_subs[key])
            total_group = sub_len_per_ch * n_ch

            # Decode the combined payload for this subband level
            if flags & FLAG_ADAPTIVE_K_BLOCK:
                combined, consumed = decode_adaptive_k_block(data_arr, pos)
            elif flags & FLAG_RLE_ZEROS:
                combined, consumed = decode_rle_zeros(data, pos)
            else:
                combined, consumed = decode_dense(data_arr, pos)
            pos += consumed

            assert len(combined) == total_group, \
                f"Group decode mismatch for {key}: {len(combined)} vs {total_group}"

            # Split back into per-channel and apply inverse bias + LPC
            for ch in range(n_ch):
                corrected = combined[ch * sub_len_per_ch:(ch + 1) * sub_len_per_ch]
                order, coeffs_q27 = channel_lpc[ch]
                if order == 0:
                    all_subs[ch][key] = corrected.astype(np.int64)
                else:
                    residual = bias_restore(corrected.astype(np.int64), bias_ctx)
                    all_subs[ch][key] = lpc_synthesize_int(residual, coeffs_q27, order)

    else:
        # Channel-major ordering
        all_subs = [{} for _ in range(n_ch)]

        for ch in range(n_ch):
            for key in subband_keys:
                order, coeffs_q27 = _read_lpc_entry(key)

                if flags & FLAG_ADAPTIVE_K_BLOCK:
                    decoded, consumed = decode_adaptive_k_block(data_arr, pos)
                elif flags & FLAG_RLE_ZEROS:
                    decoded, consumed = decode_rle_zeros(data, pos)
                else:
                    decoded, consumed = decode_dense(data_arr, pos)
                pos += consumed

                if order == 0:
                    all_subs[ch][key] = decoded.astype(np.int64)
                else:
                    residual = bias_restore(decoded.astype(np.int64), bias_ctx)
                    all_subs[ch][key] = lpc_synthesize_int(residual, coeffs_q27, order)

    # Improvement #4: inverse cross-channel differencing
    if flags & FLAG_CROSS_CHANNEL:
        for key in subband_keys:
            for ch in range(1, n_ch):
                all_subs[ch][key] = all_subs[ch][key] + all_subs[ch - 1][key]

    # Inverse lifting
    signal_out = np.zeros((n_ch, T), dtype=np.int64)
    for ch in range(n_ch):
        signal_out[ch] = inverse_nlevel_int(all_subs[ch], n_levels)

    return signal_out.astype(np.float64)


# ===================================================================
# Baseline (mirrors production _compress_bytes_ref exactly, but
# returns just the payload size for fair comparison)
# ===================================================================

def compress_baseline(signal: np.ndarray) -> bytes:
    """Compress using the production pipeline. Returns custom blob."""
    return compress_experimental(signal, flags=0)


def decompress_baseline(data: bytes) -> np.ndarray:
    """Decompress baseline blob."""
    return decompress_experimental(data, flags=0)


# ===================================================================
# Variant definitions
# ===================================================================

VARIANTS = [
    ('baseline',          0),
    ('adaptive_k_block',  FLAG_ADAPTIVE_K_BLOCK),
    ('rle_zeros',         FLAG_RLE_ZEROS),
    ('lpc_order_6',       FLAG_LPC_ORDER_6),
    ('cross_channel',     FLAG_CROSS_CHANNEL),
    ('subband_reorder',   FLAG_SUBBAND_REORDER),
    ('delta_lpc_coeffs',  FLAG_DELTA_LPC_COEFFS),
    ('bias_ctx_32',       FLAG_BIAS_CTX_32),
    ('ALL COMBINED',      FLAG_ADAPTIVE_K_BLOCK | FLAG_LPC_ORDER_6 |
                          FLAG_CROSS_CHANNEL | FLAG_SUBBAND_REORDER |
                          FLAG_DELTA_LPC_COEFFS | FLAG_BIAS_CTX_32),
]
# Note: ALL COMBINED excludes rle_zeros because it conflicts with
# adaptive_k_block (both modify the entropy encoding stage).


# ===================================================================
# Data loading
# ===================================================================

def load_windows(npz_path: str, max_windows: int = 10) -> List[np.ndarray]:
    """Load [C, 2500] int32 windows from an NPZ file.

    Returns list of numpy arrays, each [C, 2500].
    """
    try:
        f = np.load(npz_path)
    except Exception as e:
        print(f"  WARN: skipping {npz_path}: {e}", file=sys.stderr)
        return []

    if 'data' not in f:
        print(f"  WARN: skipping {npz_path}: no 'data' key", file=sys.stderr)
        return []

    data = f['data']  # [C, total_samples] int32
    if data.ndim != 2:
        print(f"  WARN: skipping {npz_path}: data.ndim={data.ndim}", file=sys.stderr)
        return []

    n_ch, total_T = data.shape
    n_windows = total_T // WINDOW_T
    if n_windows == 0:
        print(f"  WARN: skipping {npz_path}: too short ({total_T} samples)", file=sys.stderr)
        return []

    n_windows = min(n_windows, max_windows)
    windows = []
    for w in range(n_windows):
        start = w * WINDOW_T
        win = data[:, start:start + WINDOW_T].astype(np.int64)
        windows.append(win)

    return windows


# ===================================================================
# Main benchmark
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description='A/B benchmark for 7 proposed LML compression improvements')
    parser.add_argument('data_dir', type=str,
                        help='Directory containing q31 .npz files')
    parser.add_argument('--max-files', type=int, default=20,
                        help='Max number of NPZ files to process')
    parser.add_argument('--max-windows', type=int, default=10,
                        help='Max windows per file')
    parser.add_argument('--variant', type=str, default=None,
                        help='Run only this variant (name from the table)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"ERROR: {data_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Find NPZ files
    npz_files = sorted(data_dir.glob('*.npz'))
    if not npz_files:
        print(f"ERROR: no .npz files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    npz_files = npz_files[:args.max_files]

    # Load all windows
    print(f"Loading data from {len(npz_files)} files ...", flush=True)
    all_windows = []
    for npz_path in npz_files:
        wins = load_windows(str(npz_path), args.max_windows)
        all_windows.extend(wins)

    if not all_windows:
        print("ERROR: no valid windows loaded", file=sys.stderr)
        sys.exit(1)

    n_ch, T = all_windows[0].shape
    n_windows = len(all_windows)
    print(f"Loaded {n_windows} windows ({n_ch}ch x {T})")

    # Filter variants
    variants = VARIANTS
    if args.variant:
        variants = [(name, f) for name, f in VARIANTS if name == args.variant]
        if not variants:
            print(f"ERROR: unknown variant '{args.variant}'", file=sys.stderr)
            print(f"  Available: {', '.join(n for n, _ in VARIANTS)}")
            sys.exit(1)

    # Raw size per window (for CR computation)
    raw_bytes_per_window = n_ch * T * 4  # int32 = 4 bytes

    # Run each variant
    results = []
    for name, flags in variants:
        print(f"\n  Running: {name} (flags=0x{flags:04X}) ...", end=' ', flush=True)
        t0 = time.monotonic()
        total_compressed = 0
        total_raw = 0
        roundtrip_ok = 0
        roundtrip_fail = 0

        for win_i, win in enumerate(all_windows):
            try:
                compressed = compress_experimental(win, flags=flags)
                total_compressed += len(compressed)
                total_raw += raw_bytes_per_window

                # Roundtrip verify
                reconstructed = decompress_experimental(compressed, flags=flags)
                recon_int = np.round(reconstructed).astype(np.int64)

                if np.array_equal(win, recon_int):
                    roundtrip_ok += 1
                else:
                    roundtrip_fail += 1
                    diff = np.abs(win - recon_int)
                    print(f"\n    ROUNDTRIP FAIL window {win_i}: "
                          f"max_diff={diff.max()}, n_diff={np.count_nonzero(diff)}",
                          file=sys.stderr)
            except Exception as e:
                roundtrip_fail += 1
                print(f"\n    ERROR window {win_i}: {e}", file=sys.stderr)

        elapsed = time.monotonic() - t0
        cr = total_raw / total_compressed if total_compressed > 0 else 0.0
        results.append((name, cr, roundtrip_ok, roundtrip_fail, elapsed))
        print(f"CR={cr:.3f}:1  ({elapsed:.1f}s)")

    # Print results table
    baseline_cr = results[0][1] if results else 1.0
    print()
    print(f"LML Improvement A/B Benchmark "
          f"({len(npz_files)} files, {n_windows} windows, {n_ch}ch x {T})")
    sep = '\u2500' * 66
    print(sep)
    print(f"  {'Variant':<24s} {'CR':>9s}   {'vs baseline':>12s}   {'roundtrip':>10s}")
    print(sep)
    for name, cr, ok, fail, elapsed in results:
        total = ok + fail
        rt_str = f"{ok}/{total}"
        if name == 'baseline':
            delta_str = "(reference)"
        else:
            if baseline_cr > 0:
                pct = (cr - baseline_cr) / baseline_cr * 100
                sign = '+' if pct >= 0 else ''
                delta_str = f"{sign}{pct:.1f}%"
            else:
                delta_str = "N/A"
        print(f"  {name:<24s} {cr:>7.3f}:1   {delta_str:>12s}   {rt_str:>10s}")
    print(sep)

    # Exit with error if any roundtrip failures
    total_fails = sum(f for _, _, _, f, _ in results)
    if total_fails > 0:
        print(f"\nWARNING: {total_fails} total roundtrip failures!", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
