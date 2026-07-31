"""Decompression: CompressedPacket → LatentTokens (LMQ1 + LMQ3 packets).

Exact inverse of compress.py. Same byte layout, same rANS variant. Both
`SubbandCodec.decompress()` and the typed `decompress()` wrapper call the
`_decompress_bytes` / `_decompress_adaptive_bytes` primitives — guaranteed
wire-identical with the encoder.

- LMQ1 (uniform FSQ): `_decompress_bytes` — single L for the whole window.
- LMQ3 (adaptive FSQ): `_decompress_adaptive_bytes` — per-timestep level
  schedule from SNN-driven adaptive encode (mirrors
  `SubbandCodec.decompress_adaptive` at codec.py:529).

The deprecated 'LMQ2' format from prior SubbandCodec versions is rejected
with a clear error: those bytes were produced by a broken rANS encoder and
never roundtripped correctly. LMQ4..LMQ9 are reserved for future formats.
"""
import struct
from typing import Tuple

import numpy as np

from lamquant_codec.codec_types import CompressedPacket, LatentTokens
# ops.rans is imported lazily inside _decompress_bytes (numba dependency).

from lamquant_codec.ops.constants import MAGIC_LMQ as MAGIC, DEFAULT_RANS_TOTAL as _DEFAULT_TOTAL

MAGIC_LMQ3 = b'LMQ3'


# ============================================================
# Primitive: pure bytes in → numpy latent out, no dataclass.
# ============================================================

def _decompress_bytes(data: bytes,
                      rans_total: int = _DEFAULT_TOTAL
                      ) -> Tuple[np.ndarray, int, int, int, bytes, bytes]:
    """Parse an LMQ1 packet.

    Returns:
        (latent_np, quality_mode, L, fsq_levels, lpc_bytes, detail_bytes)
        where latent_np is [1, D, T] float32 (matches SubbandCodec.decompress
        shape for back-compat), and lpc_bytes/detail_bytes are the raw
        payload slices (caller decodes on demand).
    """
    header_size = 4 + 1 + 1 + 2 + 2 + 4 + 4 + 4 + 4 + 4  # 30 bytes
    if len(data) < header_size:
        raise ValueError(
            f"Truncated LMQ packet: got {len(data)} bytes, "
            f"need at least {header_size} for header.")

    (magic, quality_mode, L, lat_dim, lat_T, vmin, vmax,
     rans_len, lpc_len, detail_len) = struct.unpack(
        '<4sBBHHffIII', data[:header_size])

    if magic[:3] != b'LMQ':
        raise ValueError(f"Not an LMQ packet (magic: {magic!r}).")
    if magic != MAGIC:
        # LMQ3 has its own primitive — caller should dispatch via the typed
        # `decompress()` wrapper, not call _decompress_bytes directly.
        if magic == MAGIC_LMQ3:
            raise ValueError(
                "LMQ3 packet routed to LMQ1 reader. "
                "Call decompress(packet) or _decompress_adaptive_bytes(data)."
            )
        lmq_ver = magic[3:4]
        if lmq_ver in (b'2', b'4', b'5', b'6', b'7', b'8', b'9'):
            raise ValueError(
                f"LMQ version {lmq_ver.decode()} is newer than this "
                f"reader supports (max version 1, plus adaptive LMQ3). "
                f"Update LamQuant.")
        raise ValueError(f"Invalid LMQ version byte: {magic!r}. File may be corrupt.")

    # Validate header fields before any allocation.
    if L == 0 or L > 256:
        raise ValueError(f"Invalid FSQ levels L={L} in LMQ header.")
    if lat_dim == 0 or lat_dim > 4096:
        raise ValueError(f"Invalid latent_dim={lat_dim} in LMQ header.")
    if lat_T == 0 or lat_T > 100_000:
        raise ValueError(f"Invalid latent_T={lat_T} in LMQ header.")
    if np.isnan(vmin) or np.isnan(vmax) or np.isinf(vmin) or np.isinf(vmax):
        raise ValueError(f"Corrupt vmin/vmax in LMQ header: vmin={vmin}, vmax={vmax}")
    if vmax < vmin:
        raise ValueError(
            f"Invalid quantization range: vmax ({vmax}) < vmin ({vmin}).")

    pos = header_size
    freq_size = L * 2
    total_payload = freq_size + rans_len + lpc_len + detail_len
    if len(data) < header_size + total_payload:
        raise ValueError(
            f"Truncated LMQ packet: header declares {total_payload} bytes "
            f"of payload but only {len(data) - header_size} available.")

    freq = np.frombuffer(data[pos:pos + freq_size], dtype=np.uint16).astype(np.int32)
    pos += freq_size

    rans_payload = data[pos:pos + rans_len]
    pos += rans_len

    # Lazy-import the JIT'd decoder so plain `import lamquant_codec` doesn't
    # pull numba (~200 ms one-time on first decompress() call).
    from lamquant_codec.ops.rans import decode as rans_decode

    n_symbols = lat_dim * lat_T
    symbols = rans_decode(rans_payload, freq, n_symbols, total_freq=rans_total)

    # FSQ dequantize.
    span = vmax - vmin + 1e-8
    lat_flat = vmin + (symbols.astype(np.float32) + 0.5) * span / L
    latent_np = lat_flat.reshape(1, lat_dim, lat_T)

    lpc_bytes = bytes(data[pos:pos + lpc_len])
    pos += lpc_len
    detail_bytes = bytes(data[pos:pos + detail_len])

    return latent_np, int(quality_mode), int(L), int(L), lpc_bytes, detail_bytes


# ============================================================
# LMQ3 (adaptive FSQ) primitive — bytes in → numpy latent + schedule.
# ============================================================

def _decompress_adaptive_bytes(data: bytes,
                               rans_total: int = _DEFAULT_TOTAL
                               ) -> Tuple[np.ndarray, list, bytes]:
    """Parse an LMQ3 (adaptive FSQ) packet.

    Mirrors `SubbandCodec.decompress_adaptive` at codec.py:529 but returns
    numpy + list rather than torch.Tensor + list, so the typed pipeline
    surface stays torch-free.

    Returns:
        (latent_np, level_schedule, lpc_bytes)
        - latent_np: [1, D, T] float32 (matches LMQ1 shape).
        - level_schedule: list[int] of length T, FSQ level per timestep.
        - lpc_bytes: raw LPC payload slice (or b'' if absent).
    """
    hdr_size = 4 + 1 + 2 + 2 + 4 + 4 + 4 + 4  # 25 bytes
    if len(data) < hdr_size:
        raise ValueError(
            f"Truncated LMQ3 packet: got {len(data)} bytes, "
            f"need at least {hdr_size} for header.")

    (magic, n_runs, lat_dim, lat_T, vmin, vmax,
     rans_len, lpc_len) = struct.unpack('<4sBHHffII', data[:hdr_size])

    if magic != MAGIC_LMQ3:
        raise ValueError(
            f"_decompress_adaptive_bytes called with non-LMQ3 magic: {magic!r}")

    if n_runs == 0:
        raise ValueError("LMQ3 header declares zero schedule runs")
    if lat_dim == 0 or lat_dim > 4096:
        raise ValueError(f"Invalid lat_dim={lat_dim} in LMQ3 header.")
    if lat_T == 0 or lat_T > 100_000:
        raise ValueError(f"Invalid lat_T={lat_T} in LMQ3 header.")
    if np.isnan(vmin) or np.isnan(vmax) or np.isinf(vmin) or np.isinf(vmax):
        raise ValueError(f"Corrupt vmin/vmax in LMQ3 header: {vmin=}, {vmax=}")
    if vmax < vmin:
        raise ValueError(f"Invalid quantization range: vmax ({vmax}) < vmin ({vmin}).")

    pos = hdr_size
    span = vmax - vmin + 1e-8

    # Schedule: n_runs × (L_byte, count_byte)
    schedule_size = n_runs * 2
    if len(data) < pos + schedule_size:
        raise ValueError(
            f"Truncated LMQ3 schedule: need {schedule_size} bytes from offset "
            f"{pos}, have {len(data) - pos}.")
    runs = []
    for i in range(n_runs):
        L_val = data[pos + i * 2]
        count = data[pos + i * 2 + 1]
        if L_val == 0 or L_val > 256:
            raise ValueError(f"Invalid schedule level L={L_val} in run {i}.")
        if count == 0:
            raise ValueError(f"Zero-count schedule run at index {i}.")
        runs.append((L_val, count))
    pos += schedule_size

    level_schedule: list[int] = []
    for L_val, count in runs:
        level_schedule.extend([L_val] * count)
    if len(level_schedule) != lat_T:
        raise ValueError(
            f"Schedule length mismatch: expanded {len(level_schedule)} "
            f"timesteps but header declares lat_T={lat_T}.")

    # Freq tables: one per distinct L, in ascending L order (matches encoder).
    # Enforce the order — a crafted packet that omits, duplicates, or
    # reorders L bytes must not survive to the rANS decoder, where it
    # would raise KeyError (DoS). (V4 Pro Finding 1 of 3f147326 review.)
    distinct_levels = sorted(set(level_schedule))
    freq_tables: dict = {}
    for i, expected_L in enumerate(distinct_levels):
        if pos + 1 > len(data):
            raise ValueError(
                f"Truncated LMQ3: ran out of bytes reading freq table {i}.")
        L = data[pos]
        pos += 1
        if L != expected_L:
            raise ValueError(
                f"LMQ3 freq table {i}: got L={L}, expected L={expected_L} "
                f"(must match ascending order of distinct schedule levels)."
            )
        if pos + L * 2 > len(data):
            raise ValueError(
                f"Truncated LMQ3: freq table for L={L} needs {L*2} bytes.")
        freq = np.frombuffer(data[pos:pos + L * 2], dtype=np.uint16).astype(np.int32).copy()
        pos += L * 2
        start = np.zeros(L, dtype=np.int32)
        for k in range(1, L):
            start[k] = start[k - 1] + freq[k - 1]
        total = int(freq.sum())
        if total != rans_total:
            raise ValueError(
                f"LMQ3 freq table for L={L} sums to {total}, expected {rans_total}.")
        cum2sym = np.zeros(total, dtype=np.int32)
        for s in range(L):
            cum2sym[start[s]:start[s] + freq[s]] = s
        freq_tables[L] = (freq, start, cum2sym)

    if pos + rans_len > len(data):
        raise ValueError(
            f"Truncated LMQ3: rans_len={rans_len} but only "
            f"{len(data) - pos} bytes remain at offset {pos}.")
    # Encoder always emits ≥4 bytes of state; a shorter payload is
    # corrupt and would silently mis-decode if we let it through.
    # (V4 Pro Finding 2 of 3f147326 review.)
    if rans_len < 4:
        raise ValueError(
            f"LMQ3 rans_len={rans_len} < 4 — packet is truncated or corrupt "
            f"(rANS state requires 4 bytes minimum).")
    rans_payload = list(data[pos:pos + rans_len])
    pos += rans_len

    # Inline rANS decode — per-symbol freq table varies with the schedule,
    # so the JIT'd `rans_decode` (single-table) can't be reused here.
    RANS_L = 256 * rans_total
    byte_idx = len(rans_payload) - 1
    state = 0
    for _ in range(4):
        # rans_len >= 4 above guarantees byte_idx >= 0 here.
        state = (state << 8) | rans_payload[byte_idx]
        byte_idx -= 1

    # Compute L per symbol inline (no 409M-element Python list).
    # symbol index i maps to timestep t = i // lat_dim. With lat_dim=4096
    # and lat_T=100k, the old `flat_levels` would have been 409M ints =
    # ~3 GB Python list. (V4 Flash Findings 1+5 of 3f147326 review.)
    n_total = lat_dim * lat_T
    symbols = np.zeros(n_total, dtype=np.int32)
    for i in range(n_total):
        t = i // lat_dim
        L = level_schedule[t]
        freq, start, cum2sym = freq_tables[L]
        slot = state % rans_total
        sym = int(cum2sym[slot])
        f = int(freq[sym])
        s_val = int(start[sym])
        state = f * (state // rans_total) + slot - s_val
        while state < RANS_L and byte_idx >= 0:
            state = (state << 8) | rans_payload[byte_idx]
            byte_idx -= 1
        symbols[i] = sym

    latent = np.zeros((lat_dim, lat_T), dtype=np.float32)
    for t in range(lat_T):
        L = level_schedule[t]
        syms = symbols[t * lat_dim:(t + 1) * lat_dim]
        latent[:, t] = vmin + (syms.astype(np.float32) + 0.5) / L * span

    latent_np = latent[np.newaxis, :, :]  # [1, D, T]

    if pos + lpc_len > len(data):
        raise ValueError(
            f"Truncated LMQ3: lpc_len={lpc_len} but only "
            f"{len(data) - pos} bytes remain.")
    lpc_bytes = bytes(data[pos:pos + lpc_len]) if lpc_len > 0 else b''

    return latent_np, level_schedule, lpc_bytes


# ============================================================
# Typed pipeline entry.
# ============================================================

def decompress(packet: CompressedPacket) -> LatentTokens:
    """Parse a CompressedPacket into LatentTokens.

    Dispatches on the 4-byte magic prefix:
      - b'LMQ1' → uniform FSQ. fsq_levels = [L] (1-element list).
      - b'LMQ3' → adaptive FSQ. fsq_levels = full per-timestep schedule
        (list of length T). Existing callers that read `fsq_levels[0]`
        still get a sensible value (the level at t=0).

    The LPC + detail payload bytes (if present) are returned in the
    `side_info` field for downstream synthesis. Callers that need decoded
    LPC coefficients or detail subbands should decode those bytes separately
    (see ops/golomb.decode_detail).
    """
    data = packet.data
    if len(data) < 4:
        raise ValueError(f"Truncated packet: {len(data)} bytes, need ≥4 for magic.")
    magic = bytes(data[:4])

    if magic == MAGIC_LMQ3:
        latent_np, level_schedule, lpc_bytes = _decompress_adaptive_bytes(data)
        latent_2d = latent_np[0] if latent_np.ndim == 3 else latent_np
        return LatentTokens(
            tokens=latent_2d,
            latent=latent_2d,
            fsq_levels=level_schedule,
            shape=tuple(latent_2d.shape),
            vmin=float(latent_2d.min()),
            vmax=float(latent_2d.max()),
            side_info={
                # LMQ3 has no quality_mode byte — adaptive encode is
                # quality-mode agnostic at this layer (lpc/detail
                # decisions are made by the encoder).
                'quality_mode': None,
                'lpc_bytes': lpc_bytes,
                'detail_bytes': b'',
                'adaptive': True,
            },
        )

    # LMQ1 path
    latent_np, quality_mode, L, _, lpc_bytes, detail_bytes = _decompress_bytes(data)
    latent_2d = latent_np[0] if latent_np.ndim == 3 else latent_np
    return LatentTokens(
        tokens=latent_2d,
        latent=latent_2d,
        fsq_levels=[L],
        shape=tuple(latent_2d.shape),
        vmin=float(latent_2d.min()),
        vmax=float(latent_2d.max()),
        side_info={
            'quality_mode': quality_mode,
            'lpc_bytes': lpc_bytes,
            'detail_bytes': detail_bytes,
            'adaptive': False,
        },
    )


__all__ = ['decompress', '_decompress_bytes', '_decompress_adaptive_bytes',
           'MAGIC', 'MAGIC_LMQ3']
