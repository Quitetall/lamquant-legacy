"""Compression: LatentTokens → CompressedPacket (LMQ1 neural packet format).

The byte layout of a neural packet is defined HERE and only here. Both
the typed pipeline entry (`compress`) and the legacy `SubbandCodec.compress`
delegate to the `_compress_bytes` primitive in this module, so they are
guaranteed to produce identical bytes.

History
-------
LMQ v2 (deprecated): the original SubbandCodec.compress() used a broken
inline rANS — the byte-emission threshold was off by 8 bits (`<< 16`
instead of `<< 8`), so the encoder almost never emitted intermediate
bytes. All symbol information was crammed into the final 4 state bytes,
which is impossible for non-trivial latents. Round-trip recovered ~12% of
symbols. No LMQ v2 file ever decoded correctly and the format is now
rejected at decode time.

LMQ1 (current): uses the verified rANS in `lamquant_codec.ops.rans` with
the standard byte-wise renormalization (RANS_L = 256 * total_freq, byte
emission threshold = (RANS_L // M) * f * 256). Round-trip is bit-exact.
Magic byte is `LMQ1` (defined in `lamquant_codec.ops.constants.MAGIC_LMQ`).

LMQ1 header (30 bytes):
    [0:4]   'LMQ1'                                 — magic
    [4]     quality_mode                           — 0/1/2 (alerting/mon/clin)
    [5]     L (FSQ levels)                         — 8/16/32
    [6:8]   latent_dim (uint16 LE)
    [8:10]  latent_T (uint16 LE)
    [10:14] vmin (float32 LE)
    [14:18] vmax (float32 LE)
    [18:22] rANS payload length (uint32 LE)
    [22:26] LPC payload length (uint32 LE)
    [26:30] detail payload length (uint32 LE)
    [30:]   FSQ frequency table (L × uint16)
            rANS payload
            LPC coefficients (delta-encoded int16 Q15)
            detail subbands (run-length + Golomb-Rice per subband)
"""
import struct
import numpy as np
from typing import Dict, List, Optional

from lamquant_codec.codec_types import LatentTokens, SubbandDecomposition, CompressedPacket
# ops.rans is imported lazily inside _compress_bytes — it pulls numba (~200 ms
# one-time), and we want `import lamquant_codec` to stay near 50 ms for callers
# that only need types/contracts.

from lamquant_codec.ops.constants import (
    MAGIC_LMQ as MAGIC,
    DEFAULT_RANS_TOTAL as _DEFAULT_TOTAL,
    QUALITY_ALERTING, QUALITY_MONITORING, QUALITY_CLINICAL,
    FSQ_LEVELS_BY_MODE,
)


# ============================================================
# Primitive: pure bytes in → bytes out, no dataclass overhead.
# ============================================================

def _compress_bytes(latent: np.ndarray, *,
                    lpc_coeffs: Optional[np.ndarray] = None,
                    subbands_per_ch: Optional[List[Dict[str, np.ndarray]]] = None,
                    quality_mode: int = QUALITY_CLINICAL,
                    fsq_levels: Optional[int] = None,
                    rans_total: int = _DEFAULT_TOTAL) -> bytes:
    """Encode a continuous latent + side info into LMQ1 bytes.

    This is the fast path used by `SubbandCodec.compress()` directly —
    no dataclass allocations, no pipeline wrapping.

    Args:
        latent:           [1, D, T] or [D, T] float array (continuous pre-quant).
        lpc_coeffs:       Optional per-channel LPC coefficients (float, any shape;
                          flattened and Q15-encoded).
        subbands_per_ch:  Optional list of per-channel dicts with keys
                          'l1_detail', 'l2_detail', 'l3_detail'. Encoded only
                          when quality_mode >= MONITORING (l2) or CLINICAL (l1).
        quality_mode:     0/1/2 — alerting/monitoring/clinical.
        fsq_levels:       Override L. Default: FSQ_LEVELS_BY_MODE[quality_mode].
        rans_total:       rANS frequency table precision.

    Returns:
        Raw LMQ1 packet bytes.
    """
    L = fsq_levels if fsq_levels is not None else FSQ_LEVELS_BY_MODE.get(quality_mode, 16)

    # Lazy-import the JIT'd entropy coder so plain `import lamquant_codec`
    # doesn't pull numba (~200 ms one-time cost on first compress() call).
    from lamquant_codec.ops.rans import compute_freq, encode_with_freq

    # Normalize latent shape and dtype.
    lat = np.asarray(latent)
    if lat.ndim == 3:
        lat = lat[0]
    if lat.ndim != 2:
        raise ValueError(
            f"latent must be [D, T] or [1, D, T], got shape {lat.shape}")
    D, T = lat.shape
    lat_flat = lat.astype(np.float32).reshape(-1)

    if np.any(np.isnan(lat_flat)) or np.any(np.isinf(lat_flat)):
        raise ValueError(
            f"latent contains NaN/Inf — refusing to encode garbage "
            f"(NaN={np.isnan(lat_flat).sum()}, Inf={np.isinf(lat_flat).sum()})")

    vmin, vmax = float(lat_flat.min()), float(lat_flat.max())
    span = vmax - vmin
    if span < 1e-12:
        # Constant latent: all symbols map to the center bin.
        symbols = np.full(len(lat_flat), L // 2, dtype=np.int32)
    else:
        normalized = (lat_flat - vmin) / span
        symbols = np.clip((normalized * L).astype(np.int32), 0, L - 1)

    # Build the freq table at the fixed alphabet size L (so the header
    # always carries L entries, regardless of which symbols actually appeared).
    freq = compute_freq(symbols, n_sym=L, total_freq=rans_total)

    # rANS encode using the verified primitive.
    rans_output = encode_with_freq(symbols, freq, total_freq=rans_total)

    # LPC coefficients: delta-encoded int16 (Q15).
    lpc_payload = bytearray()
    if lpc_coeffs is not None:
        lpc_q15 = np.round(np.clip(np.asarray(lpc_coeffs, dtype=np.float64).flatten(),
                                    -1.0, 1.0) * 32767.0).astype(np.int16)
        deltas = np.diff(lpc_q15, prepend=0).astype(np.int16)
        lpc_payload = bytearray(deltas.tobytes())

    # Detail subbands (quality-gated): lazy-import Golomb-Rice.
    detail_payload = bytearray()
    if subbands_per_ch is not None and quality_mode >= QUALITY_MONITORING:
        from lamquant_codec.ops.golomb import encode_detail
        for ch_subs in subbands_per_ch:
            if not isinstance(ch_subs, dict):
                continue
            for key in ('l3_detail', 'l2_detail', 'l1_detail'):
                if key == 'l1_detail' and quality_mode < QUALITY_CLINICAL:
                    continue
                if key == 'l2_detail' and quality_mode < QUALITY_MONITORING:
                    continue
                coeffs = ch_subs.get(key, np.array([]))
                detail_payload.extend(encode_detail(coeffs))

    freq_bytes = freq.astype(np.uint16).tobytes()

    header = struct.pack('<4sBBHHffIII',
                         MAGIC,
                         int(quality_mode),
                         int(L),
                         int(D), int(T),
                         vmin, vmax,
                         len(rans_output),
                         len(lpc_payload),
                         len(detail_payload))

    return (bytes(header) + freq_bytes + bytes(rans_output) +
            bytes(lpc_payload) + bytes(detail_payload))


# ============================================================
# Typed pipeline entry.
# ============================================================

def _compress_adaptive_bytes(latent: np.ndarray, level_schedule,
                             *, lpc_coeffs=None,
                             rans_total: int = _DEFAULT_TOTAL) -> bytes:
    """Encode with per-timestep adaptive FSQ levels (LMQ3 format).

    The SNN classifies each timestep and assigns L=2 (quiet), L=3 (active),
    or L=5 (event). Each segment gets its own FSQ quantization level,
    spending bits only where the clinical signal needs them.

    Args:
        latent:         [D, T] or [1, D, T] float array.
        level_schedule: [T] array/list of FSQ levels (2, 3, or 5).
        lpc_coeffs:     Optional LPC coefficients.
        rans_total:     rANS frequency table precision.

    Returns:
        Raw LMQ3 packet bytes.
    """
    lat = np.asarray(latent, dtype=np.float32)
    if lat.ndim == 3:
        lat = lat[0]
    D, T = lat.shape
    if T == 0:
        raise ValueError("latent has zero timesteps")
    sched = np.asarray(level_schedule, dtype=np.int32)[:T]
    if len(sched) == 0:
        raise ValueError("level_schedule is empty")

    if np.any(np.isnan(lat)) or np.any(np.isinf(lat)):
        raise ValueError(
            f"latent contains NaN/Inf — refusing to encode garbage "
            f"(NaN={np.isnan(lat).sum()}, Inf={np.isinf(lat).sum()})")

    vmin, vmax = float(lat.min()), float(lat.max())
    span = vmax - vmin + 1e-8

    # Run-length encode the schedule
    runs = []
    cur_L, cur_count = int(sched[0]), 1
    for t in range(1, T):
        if int(sched[t]) == cur_L:
            cur_count += 1
        else:
            runs.append((cur_L, cur_count))
            cur_L, cur_count = int(sched[t]), 1
    runs.append((cur_L, cur_count))

    # Quantize each timestep with its assigned L
    all_symbols = []
    for t in range(T):
        L = int(sched[t])
        col = lat[:, t]
        normalized = (col - vmin) / span
        symbols = np.clip((normalized * L).astype(np.int32), 0, L - 1)
        all_symbols.append((L, symbols))

    # Per-level frequency tables
    level_symbols = {}
    for L, syms in all_symbols:
        level_symbols.setdefault(L, []).extend(syms.tolist())

    freq_tables = {}
    for L in sorted(level_symbols.keys()):
        syms = np.array(level_symbols[L], dtype=np.int64)
        counts = np.bincount(syms, minlength=L)
        freq = np.maximum(1, (counts / max(counts.sum(), 1) * rans_total).astype(np.int32))
        freq[np.argmax(freq)] += rans_total - freq.sum()
        start = np.zeros(L, dtype=np.int32)
        for i in range(1, L):
            start[i] = start[i - 1] + freq[i - 1]
        freq_tables[L] = (freq, start)

    # rANS encode (reverse order — LIFO)
    RANS_L = 256 * rans_total
    state = RANS_L
    byte_stream = bytearray()

    flat_symbols = []
    flat_levels = []
    for t in range(T):
        L, syms = all_symbols[t]
        for s in syms:
            flat_symbols.append(int(s))
            flat_levels.append(L)

    for idx in range(len(flat_symbols) - 1, -1, -1):
        sym = flat_symbols[idx]
        L = flat_levels[idx]
        freq, start = freq_tables[L]
        f = int(freq[sym])
        s = int(start[sym])
        threshold = ((RANS_L // rans_total) * f) << 8
        while state >= threshold:
            byte_stream.append(state & 0xFF)
            state >>= 8
        state = (state // f) * rans_total + (state % f) + s

    for _ in range(4):
        byte_stream.append(state & 0xFF)
        state >>= 8

    # LPC payload
    lpc_payload = bytearray()
    if lpc_coeffs is not None:
        lpc_q15 = np.round(np.clip(np.asarray(lpc_coeffs, dtype=np.float64).flatten(),
                                    -1.0, 1.0) * 32767.0).astype(np.int16)
        deltas = np.diff(lpc_q15, prepend=0).astype(np.int16)
        lpc_payload = bytearray(deltas.tobytes())

    # Schedule payload
    schedule_payload = bytearray()
    for L_val, count in runs:
        # Cap run length at 255 to prevent uint8 overflow
        while count > 255:
            schedule_payload.append(L_val & 0xFF)
            schedule_payload.append(255)
            count -= 255
        schedule_payload.append(L_val & 0xFF)
        schedule_payload.append(count & 0xFF)

    # Freq table payload
    freq_payload = bytearray()
    for L in sorted(freq_tables.keys()):
        freq, _ = freq_tables[L]
        freq_payload.append(L & 0xFF)
        freq_payload.extend(freq.astype(np.uint16).tobytes())

    # LMQ3 header
    header = struct.pack('<4sBHHffII',
                         b'LMQ3',
                         len(runs),
                         D, T,
                         vmin, vmax,
                         len(byte_stream),
                         len(lpc_payload))

    return (bytes(header) + bytes(schedule_payload) +
            bytes(freq_payload) + bytes(byte_stream) + bytes(lpc_payload))


def compress(tokens: LatentTokens,
             subband: Optional[SubbandDecomposition] = None,
             *,
             quality_mode: int = QUALITY_CLINICAL) -> CompressedPacket:
    """Typed wrapper around `_compress_bytes`.

    When tokens.fsq_levels is a per-timestep schedule (from SNN adaptive
    mode), this routes to the LMQ3 adaptive path. When it's a single
    value or None, uses the standard LMQ1 uniform path.

    Args:
        tokens:        LatentTokens from encode(). `.latent` is used if set,
                       otherwise `.tokens` (already-quantized path).
        subband:       Optional SubbandDecomposition carrying LPC coefficients
                       and detail subbands. Needed for quality_mode >= 1.
        quality_mode:  0/1/2 — alerting/monitoring/clinical.

    Returns:
        CompressedPacket whose `.data` is a byte-exact LMQ1/LMQ3 packet.
    """
    latent = tokens.latent if tokens.latent is not None else tokens.tokens
    if latent is None:
        raise ValueError(
            "LatentTokens has neither .latent nor .tokens populated.")

    lpc = None
    details = None
    if subband is not None:
        lpc = subband.lpc_coeffs
        if subband.l1_detail is not None and subband.l1_detail.size > 0:
            C = subband.l1_detail.shape[0]
            details = [
                {'l1_detail': subband.l1_detail[c],
                 'l2_detail': subband.l2_detail[c],
                 'l3_detail': subband.l3_detail[c]}
                for c in range(C)
            ]

    # Check if fsq_levels is a per-timestep schedule (adaptive SNN mode)
    # vs a single uniform level or None.
    is_adaptive = (tokens.fsq_levels is not None
                   and len(tokens.fsq_levels) > 1
                   and len(set(tokens.fsq_levels)) > 1)

    if is_adaptive:
        data = _compress_adaptive_bytes(latent, tokens.fsq_levels,
                                        lpc_coeffs=lpc)
        mode_tag = 'neural_adaptive'
    else:
        fsq_level = (tokens.fsq_levels[0] if tokens.fsq_levels else None)
        data = _compress_bytes(latent, lpc_coeffs=lpc, subbands_per_ch=details,
                               quality_mode=quality_mode,
                               fsq_levels=fsq_level)
        mode_tag = 'neural'

    return CompressedPacket(
        data=data,
        mode=mode_tag,
        quality_mode=quality_mode,
        metadata={
            'snac_preset': tokens.snac_preset,
            'latent_shape': tuple(tokens.shape),
            'adaptive': is_adaptive,
        },
    )


__all__ = [
    'compress', '_compress_bytes',
    'QUALITY_ALERTING', 'QUALITY_MONITORING', 'QUALITY_CLINICAL',
    'FSQ_LEVELS_BY_MODE',
]
