"""Legacy LML iteration decoders (LMQ4 / LMQ5 / LML ) — opt-in only.

Active production decoders live in `lamquant_codec.lossless` and accept
ONLY the current `LML1` magic. This module exists so the older iterations
saved during development can still be read by hand. Nothing in production
imports from here; do not wire anything to these functions.

LMQ4: 18-byte header, no CRC. Bit 0 of klt_flag was a real KLT flag.
LMQ5: 22-byte header, CRC-32, bit 0 of flags was KLT, bits 2-7 noise_bits.
LML : 22-byte header, identical to LMQ5 (a transient renaming).
LML1: current — handled by `lamquant_codec.lossless._decompress_bytes_ref`.
"""
from __future__ import annotations

import struct
import zlib
import numpy as np

from lamquant_codec.ops.bias import (
    restore_jit as _bias_restore_jit,
    BIAS_CTX_LEN as _BIAS_CTX_LEN,
)
from lamquant_codec.ops.golomb import decode_dense

LEGACY_MAGICS = (b'LMQ4', b'LMQ5', b'LML ')


def _decompress_legacy_bytes_ref(data: bytes, *, lifting_rots=None) -> np.ndarray:
    """Decode legacy LMQ4/LMQ5/LML  bytes → [C, T] float64 signal.

    Mirrors the behaviour of the pre-LML1 reference decoder. Reserved-bit
    fail-closed semantics are NOT applied — legacy iterations used bit 0
    as a KLT flag.
    """
    from lamquant_codec.lossless import (
        _lifting_nlevel_inverse,
        apply_lifting_klt_inverse,
    )
    from lamquant_codec.ops.lpc import synthesize_int as lpc_synthesize_int

    if len(data) < 4:
        raise ValueError(f"Truncated LML data: {len(data)} bytes")

    # Skip optional ASCII prefix
    offset = 0
    nl = data.find(b'\n')
    if (0 < nl < 128
            and all(0x20 <= b <= 0x7E for b in data[:nl])
            and len(data) > nl + 4
            and data[nl + 1:nl + 4] in (b'LML', b'LMQ')):
        offset = nl + 1
    data = data[offset:]

    if len(data) < 4:
        raise ValueError("Truncated LML data after prefix")
    magic = data[:4]
    if magic not in LEGACY_MAGICS:
        raise ValueError(
            f"Magic {magic!r} is not a legacy iteration. "
            f"For current LML1 use lamquant_codec.lossless._decompress_bytes_ref.")

    if magic == b'LMQ4':
        # 18-byte header, no CRC, klt_flag is a separate u8 not bit-packed.
        hdr_size = 18
        if len(data) < hdr_size:
            raise ValueError(f"Truncated LMQ4 header")
        (_, n_ch, T, n_levels, klt_flag,
         lpc_len, sub_len) = struct.unpack('<4sHHBBII', data[:hdr_size])
        crc_expected = None
        noise_bits = 0
    else:
        # LMQ5 / LML  — 22-byte header with CRC-32 and bit-packed flags.
        hdr_size = 22
        if len(data) < hdr_size:
            raise ValueError(f"Truncated {magic!r} header")
        (_, n_ch, T, n_levels, flags,
         lpc_len, sub_len, crc_expected) = struct.unpack('<4sHHBBIII', data[:hdr_size])
        klt_flag = flags & 0x01
        noise_bits = (flags >> 2) & 0x3F

    payload = data[hdr_size:hdr_size + lpc_len + sub_len]
    if crc_expected is not None:
        crc_actual = zlib.crc32(payload) & 0xFFFFFFFF
        if crc_actual != crc_expected:
            raise ValueError(
                f"CRC-32 mismatch: expected 0x{crc_expected:08X}, "
                f"got 0x{crc_actual:08X}.")

    expected_len = hdr_size + lpc_len + sub_len
    if len(data) < expected_len:
        raise ValueError(
            f"Truncated legacy data: got {len(data)} bytes, "
            f"header declares {expected_len}.")

    pos = hdr_size
    lpc_data = data[pos:pos + lpc_len]
    pos += lpc_len
    lpc_pos = 0

    subband_keys = [f'l{n_levels}_approx'] + \
                   [f'l{lvl}_detail' for lvl in range(n_levels, 0, -1)]
    signal_out = np.zeros((n_ch, T), dtype=np.int64)
    data_arr = np.frombuffer(data, dtype=np.uint8).copy()

    for ch in range(n_ch):
        subs = {}
        for key in subband_keys:
            order = lpc_data[lpc_pos]
            lpc_pos += 1
            coeffs_q27 = np.frombuffer(
                lpc_data[lpc_pos:lpc_pos + order * 4],
                dtype=np.int32).copy()
            lpc_pos += order * 4
            decoded, bytes_consumed = decode_dense(data_arr, pos)
            pos += bytes_consumed
            residual = _bias_restore_jit(decoded.astype(np.int64), _BIAS_CTX_LEN)
            if order == 0:
                subs[key] = residual
            else:
                subs[key] = lpc_synthesize_int(residual, coeffs_q27, order)
        signal_out[ch] = _lifting_nlevel_inverse(subs, n_levels)

    if klt_flag:
        if lifting_rots is None:
            raise ValueError(
                f"Legacy {magic!r} packet has KLT flag set but no "
                f"lifting_rots was provided to decompress.")
        signal_out = apply_lifting_klt_inverse(signal_out, lifting_rots)
    elif lifting_rots is not None:
        signal_out = apply_lifting_klt_inverse(signal_out, lifting_rots)

    if noise_bits > 0:
        signal_out <<= noise_bits
    return signal_out.astype(np.float64)


def peek_header_legacy(data: bytes):
    """Read a legacy LMQ4/LMQ5/LML  packet header without decompressing."""
    from lamquant_codec.file_info import LQPacketHeader

    if len(data) < 4:
        raise ValueError(f"Truncated: {len(data)} bytes")
    offset = 0
    if data[:3] != b'LMQ' and data[:4] != b'LML ':
        nl = data.find(b'\n')
        if 0 <= nl < 128:
            offset = nl + 1
    data = data[offset:]
    if len(data) < 4:
        raise ValueError("Truncated after prefix")
    magic = data[:4]
    if magic == b'LMQ5' or magic == b'LML ':
        if len(data) < 22:
            raise ValueError(f"Truncated {magic!r} header")
        _, n_ch, T, n_levels, flags, lpc_len, sub_len, crc = \
            struct.unpack('<4sHHBBIII', data[:22])
        nb = (flags >> 2) & 0x3F
        return LQPacketHeader(
            version=magic.decode('ascii').rstrip(),
            n_channels=n_ch, n_samples=T,
            n_levels=n_levels, klt=bool(flags & 0x01),
            noise_bits=nb, lossless=(nb == 0),
            lpc_meta_bytes=lpc_len, payload_bytes=sub_len,
            crc32=crc, total_bytes=22 + lpc_len + sub_len,
        )
    elif magic == b'LMQ4':
        if len(data) < 18:
            raise ValueError("Truncated LMQ4 header")
        _, n_ch, T, n_levels, klt_flag, lpc_len, sub_len = \
            struct.unpack('<4sHHBBII', data[:18])
        return LQPacketHeader(
            version='LMQ4', n_channels=n_ch, n_samples=T,
            n_levels=n_levels, klt=bool(klt_flag),
            noise_bits=0, lossless=True,
            lpc_meta_bytes=lpc_len, payload_bytes=sub_len,
            crc32=0, total_bytes=18 + lpc_len + sub_len,
        )
    raise ValueError(f"Not a legacy iteration magic: {magic!r}")
