#!/usr/bin/env python3
"""benchmark_lossless.py — Empirical lossless codec benchmark.

Bit-perfect roundtrip verification + compression ratio measurement
on real clinical EEG data across all datasets. Tests both Golomb-Rice
(current) and rANS (proposed) entropy coding.

Every window is:
  1. Encoded with both codecs
  2. Decoded back
  3. Compared bit-for-bit against original
  4. Any mismatch = hard failure with full diagnostic

Usage:
    python lamquant_codec/benchmark_lossless.py
    python lamquant_codec/benchmark_lossless.py --max-windows 100
    python lamquant_codec/benchmark_lossless.py --dataset chbmit
"""
from __future__ import annotations

import argparse
import glob
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent

from lamquant_codec.ops.golomb import encode_dense, decode_dense
from lamquant_codec.ops.rans import compute_freq, encode_with_freq, decode as rans_decode
from lamquant_codec.ops.lpc import analyze_channel as lpc_analyze_channel, analyze_int as lpc_analyze_int
from lamquant_codec.ops.lifting import forward_1d_int as lifting_1d_forward_int, inverse_1d_int as lifting_1d_inverse_int


# ============================================================
# Full encode/decode pipelines for each codec
# ============================================================

def _decompose_channel(signal_1d: np.ndarray, n_levels: int = 3):
    """Lifting decomposition of one channel. Returns list of subbands."""
    approx = signal_1d.copy()
    details = []
    for _ in range(n_levels):
        approx, detail = lifting_1d_forward_int(approx)
        details.append(detail)
    return [approx] + list(reversed(details))


def _recompose_channel(subbands: list, n_levels: int = 3) -> np.ndarray:
    """Inverse lifting from subbands. Returns reconstructed channel.

    subbands order: [l3_approx, l3_detail, l2_detail, l1_detail]
    Inverse: l3_approx + l3_detail → l2, then l2 + l2_detail → l1, etc.
    Details are already in deepest-first order (matching decompose output).
    """
    approx = subbands[0].copy()
    # Details are in order [l3_detail, l2_detail, l1_detail] — apply sequentially
    for detail in subbands[1:]:
        approx = lifting_1d_inverse_int(approx, detail)
    return approx


def _lpc_forward(sub: np.ndarray, order: int = 8):
    """LPC analysis → (coeffs_q27, residual, order)."""
    order = min(order, max(2, len(sub) // 8))
    coeffs_f, _ = lpc_analyze_channel(
        sub.astype(np.float64), order=order,
        autocorr_len=min(256, len(sub) // 2))
    coeffs_q27, residual = lpc_analyze_int(sub, coeffs_f, order)
    return coeffs_q27, residual, order


def _lpc_inverse(residual: np.ndarray, coeffs_q27: np.ndarray, order: int) -> np.ndarray:
    """LPC synthesis from residual + coefficients.

    Starts prediction from n=0 with zero-padded history, matching
    lpc_analyze_int's forward pass (sig_padded[:order] = 0).
    """
    Q = 27
    out = residual.astype(np.int64).copy()
    for n in range(len(out)):
        pred = np.int64(0)
        for k in range(order):
            idx = n - 1 - k
            if idx >= 0:
                pred += np.int64(coeffs_q27[k]) * out[idx]
        out[n] += pred >> Q
    return out


def encode_window_golomb(window: np.ndarray, n_levels: int = 3):
    """Full lossless encode: lifting → LPC → Golomb-Rice. Returns (bytes, metadata)."""
    C, T = window.shape
    all_bytes = bytearray()
    metadata = []  # [(order, coeffs_q27, sub_len), ...] per channel per subband

    for ch in range(C):
        subs = _decompose_channel(window[ch], n_levels)
        ch_meta = []
        for sub in subs:
            coeffs_q27, residual, order = _lpc_forward(sub, order=8)
            encoded = encode_dense(residual.astype(np.int64))
            all_bytes.extend(encoded)
            ch_meta.append((order, coeffs_q27.copy(), len(sub)))
        metadata.append(ch_meta)

    return bytes(all_bytes), metadata


def decode_window_golomb(data: bytes, metadata, C: int, n_levels: int = 3) -> np.ndarray:
    """Full lossless decode: Golomb-Rice → LPC inverse → lifting inverse."""
    offset = 0
    channels = []

    for ch in range(C):
        subs = []
        for order, coeffs_q27, sub_len in metadata[ch]:
            residual, consumed = decode_dense(data, offset)
            offset += consumed
            reconstructed = _lpc_inverse(residual, coeffs_q27, order)
            subs.append(reconstructed)
        channel = _recompose_channel(subs, n_levels)
        channels.append(channel)

    return np.array(channels, dtype=np.int64)


def encode_window_rans(window: np.ndarray, n_levels: int = 3,
                       shared_freqs: dict = None):
    """Full lossless encode: lifting → LPC → rANS. Returns (bytes, metadata, freqs)."""
    C, T = window.shape
    all_bytes = bytearray()
    metadata = []
    subband_names = [f'sub{i}' for i in range(n_levels + 1)]
    collected_zz = {name: [] for name in subband_names}

    # Pass 1: collect all residuals for freq table building (if not shared)
    all_residuals = []
    for ch in range(C):
        subs = _decompose_channel(window[ch], n_levels)
        ch_meta = []
        for i, sub in enumerate(subs):
            coeffs_q27, residual, order = _lpc_forward(sub, order=8)
            zz = ((residual.astype(np.int64) << 1) ^ (residual.astype(np.int64) >> 63))
            collected_zz[subband_names[i]].append(zz)
            ch_meta.append((order, coeffs_q27.copy(), len(sub), zz))
        metadata.append(ch_meta)

    # Build freq tables (per subband type, shared across channels)
    if shared_freqs is None:
        shared_freqs = {}
        for name in subband_names:
            all_zz = np.concatenate(collected_zz[name])
            shared_freqs[name] = compute_freq(all_zz, total_freq=4096)

    # Pass 2: encode with shared tables
    for ch in range(C):
        for i, (order, coeffs_q27, sub_len, zz) in enumerate(metadata[ch]):
            freq = shared_freqs[subband_names[i]]
            # Clamp symbols to freq table range
            zz_clamped = np.clip(zz, 0, len(freq) - 1)
            rans_bytes = encode_with_freq(zz_clamped, freq, total_freq=4096)
            all_bytes.extend(struct.pack('<H', len(rans_bytes)))
            all_bytes.extend(rans_bytes)

    # Strip zz from metadata for decode
    meta_clean = []
    for ch_meta in metadata:
        meta_clean.append([(o, c, l) for o, c, l, _ in ch_meta])

    return bytes(all_bytes), meta_clean, shared_freqs


def decode_window_rans(data: bytes, metadata, C: int, n_levels: int = 3,
                       shared_freqs: dict = None) -> np.ndarray:
    """Full lossless decode: rANS → LPC inverse → lifting inverse."""
    offset = 0
    subband_names = [f'sub{i}' for i in range(n_levels + 1)]
    channels = []

    for ch in range(C):
        subs = []
        for i, (order, coeffs_q27, sub_len) in enumerate(metadata[ch]):
            # Read rANS payload length
            payload_len = struct.unpack('<H', data[offset:offset+2])[0]
            offset += 2
            rans_bytes = data[offset:offset+payload_len]
            offset += payload_len

            freq = shared_freqs[subband_names[i]]
            zz = rans_decode(rans_bytes, freq, sub_len, total_freq=4096)

            # Zigzag decode
            residual = (zz >> 1) ^ -(zz & 1)

            reconstructed = _lpc_inverse(residual, coeffs_q27, order)
            subs.append(reconstructed)
        channel = _recompose_channel(subs, n_levels)
        channels.append(channel)

    return np.array(channels, dtype=np.int64)


# ============================================================
# Benchmark runner
# ============================================================

def benchmark_file(npz_path: str, max_windows: int = 20,
                   use_native: bool = True) -> dict:
    """Benchmark one file. Returns per-window results."""
    with np.load(npz_path, allow_pickle=True) as d:
        if use_native and 'signal_native' in d:
            data = d['signal_native'].astype(np.int64)
            precision = 'int16'
        else:
            data = d['data'].astype(np.int64)
            precision = 'int32'
        has_seizure = bool(d['seizure_mask'].any()) if 'seizure_mask' in d else False

    C, T = data.shape
    n_windows = min(max_windows, T // 2500)
    if n_windows == 0:
        return {'error': 'too_short', 'file': npz_path}

    raw_bytes = C * 2500 * (2 if precision == 'int16' else 4)
    results = []

    for w in range(n_windows):
        window = data[:, w*2500:(w+1)*2500]

        # --- Golomb-Rice ---
        t0 = time.perf_counter()
        gr_bytes, gr_meta = encode_window_golomb(window)
        t_gr_enc = time.perf_counter() - t0

        t0 = time.perf_counter()
        gr_recon = decode_window_golomb(gr_bytes, gr_meta, C)
        t_gr_dec = time.perf_counter() - t0

        gr_exact = np.array_equal(window, gr_recon)

        # --- rANS ---
        t0 = time.perf_counter()
        rans_bytes, rans_meta, rans_freqs = encode_window_rans(window)
        t_rans_enc = time.perf_counter() - t0

        t0 = time.perf_counter()
        rans_recon = decode_window_rans(rans_bytes, rans_meta, C,
                                         shared_freqs=rans_freqs)
        t_rans_dec = time.perf_counter() - t0

        rans_exact = np.array_equal(window, rans_recon)

        results.append({
            'window': w,
            'raw_bytes': raw_bytes,
            'gr_bytes': len(gr_bytes),
            'gr_cr': raw_bytes / max(1, len(gr_bytes)),
            'gr_exact': gr_exact,
            'gr_enc_ms': t_gr_enc * 1000,
            'gr_dec_ms': t_gr_dec * 1000,
            'rans_bytes': len(rans_bytes),
            'rans_cr': raw_bytes / max(1, len(rans_bytes)),
            'rans_exact': rans_exact,
            'rans_enc_ms': t_rans_enc * 1000,
            'rans_dec_ms': t_rans_dec * 1000,
        })

        if not gr_exact:
            diff = np.abs(window - gr_recon)
            print(f"  [FAIL] Golomb-Rice bit mismatch! win={w}, "
                  f"max_diff={diff.max()}, n_diff={np.count_nonzero(diff)}")

        if not rans_exact:
            diff = np.abs(window - rans_recon)
            print(f"  [FAIL] rANS bit mismatch! win={w}, "
                  f"max_diff={diff.max()}, n_diff={np.count_nonzero(diff)}")

    return {
        'file': os.path.basename(npz_path),
        'precision': precision,
        'has_seizure': has_seizure,
        'n_windows': n_windows,
        'results': results,
    }


def main():
    parser = argparse.ArgumentParser(prog='benchmark_lossless')
    parser.add_argument('--dir', type=str,
                        default=str(_REPO / 'ai_models' / 'dataset_sim' / 'q31_events'))
    parser.add_argument('--max-files', type=int, default=20)
    parser.add_argument('--max-windows', type=int, default=10)
    parser.add_argument('--dataset', type=str, default=None,
                        help='Filter by dataset prefix (chbmit, tuh, tuep, siena)')
    parser.add_argument('--native', action='store_true', default=False,
                        help='Use signal_native (int16) if available')
    args = parser.parse_args()

    # Find files
    pattern = f'{args.dataset}_*.npz' if args.dataset else '*.npz'
    files = sorted(glob.glob(os.path.join(args.dir, pattern)))
    if not files:
        print(f'No files matching {pattern} in {args.dir}')
        return 1

    # Sample diverse files
    if len(files) > args.max_files:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(files), args.max_files, replace=False)
        files = [files[i] for i in sorted(indices)]

    print(f'=== LamQuant Lossless Codec Benchmark ===')
    print(f'Files: {len(files)}, max windows/file: {args.max_windows}')
    print(f'Precision: {"native int16" if args.native else "Q31 int32"}')
    print()

    # Run benchmarks
    all_gr_bytes = []
    all_rans_bytes = []
    all_raw = []
    gr_failures = 0
    rans_failures = 0
    total_windows = 0
    gr_enc_times = []
    gr_dec_times = []
    rans_enc_times = []
    rans_dec_times = []

    for i, f in enumerate(files):
        fname = os.path.basename(f)[:30]
        result = benchmark_file(f, max_windows=args.max_windows,
                                use_native=args.native)

        if 'error' in result:
            continue

        file_gr = []
        file_rans = []
        file_raw = []
        file_gr_ok = True
        file_rans_ok = True

        for r in result['results']:
            file_gr.append(r['gr_bytes'])
            file_rans.append(r['rans_bytes'])
            file_raw.append(r['raw_bytes'])
            gr_enc_times.append(r['gr_enc_ms'])
            gr_dec_times.append(r['gr_dec_ms'])
            rans_enc_times.append(r['rans_enc_ms'])
            rans_dec_times.append(r['rans_dec_ms'])
            if not r['gr_exact']:
                gr_failures += 1
                file_gr_ok = False
            if not r['rans_exact']:
                rans_failures += 1
                file_rans_ok = False
            total_windows += 1

        all_gr_bytes.extend(file_gr)
        all_rans_bytes.extend(file_rans)
        all_raw.extend(file_raw)

        avg_gr_cr = np.mean(file_raw) / np.mean(file_gr)
        avg_rans_cr = np.mean(file_raw) / np.mean(file_rans)
        gr_tag = 'OK' if file_gr_ok else 'FAIL'
        rans_tag = 'OK' if file_rans_ok else 'FAIL'
        sz = 'sz' if result.get('has_seizure') else '  '

        print(f'  [{i+1:>2}/{len(files)}] {fname:<30} {sz} '
              f'GR={avg_gr_cr:>5.2f}:1 [{gr_tag:>4}]  '
              f'rANS={avg_rans_cr:>5.2f}:1 [{rans_tag:>4}]  '
              f'win={result["n_windows"]}')

    # Summary
    print()
    print(f'=== Summary ({total_windows} windows across {len(files)} files) ===')
    print()

    total_raw = sum(all_raw)
    total_gr = sum(all_gr_bytes)
    total_rans = sum(all_rans_bytes)

    print(f'  {"Codec":<20} {"Total":>12} {"CR":>8} {"Bit-exact":>12} '
          f'{"Enc ms":>8} {"Dec ms":>8}')
    print(f'  {"-"*68}')
    print(f'  {"Raw":<20} {total_raw:>10,} B {"1.00:1":>8}')
    print(f'  {"Golomb-Rice":<20} {total_gr:>10,} B '
          f'{total_raw/total_gr:>7.2f}:1 '
          f'{total_windows - gr_failures:>5}/{total_windows:>5} '
          f'{np.mean(gr_enc_times):>7.1f} {np.mean(gr_dec_times):>7.1f}')
    print(f'  {"rANS":<20} {total_rans:>10,} B '
          f'{total_raw/total_rans:>7.2f}:1 '
          f'{total_windows - rans_failures:>5}/{total_windows:>5} '
          f'{np.mean(rans_enc_times):>7.1f} {np.mean(rans_dec_times):>7.1f}')
    print()

    improvement = (1 - total_rans / total_gr) * 100
    print(f'  rANS vs Golomb-Rice: {improvement:+.1f}% size reduction')
    print()

    if gr_failures > 0:
        print(f'  !!! GOLOMB-RICE: {gr_failures} BIT-PARITY FAILURES !!!')
    if rans_failures > 0:
        print(f'  !!! rANS: {rans_failures} BIT-PARITY FAILURES !!!')
    if gr_failures == 0 and rans_failures == 0:
        print(f'  All {total_windows} windows: BIT-PERFECT on both codecs')

    return 1 if (gr_failures + rans_failures) > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
