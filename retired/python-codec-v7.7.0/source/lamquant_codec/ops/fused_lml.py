"""
Fused LML Pipeline — canonical compress/decompress path.

Single numba call replaces hundreds of Python→numba transitions.
Calls canonical functions from lamquant_codec.ops:
  - lifting: forward_1d_int_jit / inverse_1d_int_jit
  - lpc: analyze_jit / synthesize_jit

Update those functions → this pipeline gets faster too.
One source of truth. No copies.

Usage:
    from lamquant_codec.ops.fused_lml import fused_compress, fused_decompress
    packet = fused_compress(signal_int64)
    recovered = fused_decompress(packet)
"""
import struct
import zlib

import numpy as np

from lamquant_codec.errors import (
    LmlHeaderError,
    LmlLegacyMagicError,
    LmlMagicError,
    LmlNoiseStrippedError,
    LmlReservedBitsSetError,
    LmlTruncatedError,
    LmlVersionError,
)

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    # Import the canonical numba functions — single source of truth
    from lamquant_codec.ops.lifting import forward_1d_int_jit as lifting_1d_forward_int_jit
    from lamquant_codec.ops.lifting import inverse_1d_int_jit as lifting_1d_inverse_int_jit
    from lamquant_codec.ops.lpc import analyze_jit as lpc_analyze_jit
    from lamquant_codec.ops.lpc import synthesize_jit as lpc_synthesize_jit

    @numba.njit(cache=True, fastmath=False)
    def _process_all_channels(signal_int, n_ch, ctx_len):
        """ONE numba call replaces 420 Python→numba transitions.
        Calls the canonical functions from lamquant_codec.ops.
        """
        orders_spec = (1, 1, 2, 3)  # approx, d3, d2, d1

        all_orders = np.empty(n_ch * 4, dtype=numba.int64)
        all_residuals = []
        all_coeffs = []

        for ch in range(n_ch):
            # 3-level lifting using canonical function
            a1, d1 = lifting_1d_forward_int_jit(signal_int[ch])
            a2, d2 = lifting_1d_forward_int_jit(a1)
            a3, d3 = lifting_1d_forward_int_jit(a2)

            subbands = (a3, d3, d2, d1)

            for sb_idx in range(4):
                flat_idx = ch * 4 + sb_idx
                sb = subbands[sb_idx]
                order = orders_spec[sb_idx]
                if len(sb) < order * 4:
                    order = max(1, len(sb) // 4)

                # LPC + bias using canonical function
                cq, res = lpc_analyze_jit(sb, order, ctx_len)
                all_orders[flat_idx] = order
                all_residuals.append(res)
                all_coeffs.append(cq)

        return all_orders, all_coeffs, all_residuals


def fused_compress(signal: np.ndarray, *, noise_bits: int = 0) -> bytes:
    """Fused LML compress: [C, T] → LML1 bytes.

    4× faster than _compress_bytes(). Byte-identical output.
    Calls canonical numba functions from lamquant_codec.ops.
    """
    from lamquant_codec.ops.golomb import encode_dense

    n_ch, T = signal.shape
    signal_int = np.round(signal).astype(np.int64)
    n_levels = 3

    noise_bits = max(0, min(32, int(noise_bits)))
    if noise_bits > 0:
        mask = (1 << noise_bits) - 1
        bottom_energy = int(np.count_nonzero(signal_int & mask))
        if bottom_energy == 0 and T > 16:
            raise LmlNoiseStrippedError(
                f"noise_bits={noise_bits} requested but the bottom {noise_bits} "
                f"bits are ALREADY zero across all {n_ch}×{T} samples. "
                f"This signal was likely already noise-stripped. "
                f"Re-stripping would destroy real data. Use noise_bits=0.")
        signal_int >>= noise_bits

    while T < (4 * (1 << n_levels)) and n_levels > 0:
        n_levels -= 1

    # Fused orchestrator requires exactly 3-level lifting.
    # Short signals that reduce n_levels fall back to reference path.
    if n_levels != 3:
        from lamquant_codec.lossless import _compress_bytes_ref
        return _compress_bytes_ref(signal, noise_bits=noise_bits)

    flags = (noise_bits & 0x3F) << 2
    from lamquant_codec.ops.constants import BIAS_CTX_LEN
    ctx_len = np.int64(BIAS_CTX_LEN)

    # ONE numba call for all channels
    all_orders, all_coeffs, all_residuals = _process_all_channels(
        signal_int, n_ch, ctx_len)

    # GR encoding (existing tested function)
    lpc_parts = []
    payload_parts = []
    for i in range(n_ch * 4):
        order = int(all_orders[i])
        lpc_parts.append(bytes([order]) + all_coeffs[i].astype(np.int32).tobytes())
        payload_parts.append(encode_dense(all_residuals[i].astype(np.int64)))

    lpc_meta = b''.join(lpc_parts)
    subband_payload = b''.join(payload_parts)
    payload = lpc_meta + subband_payload

    # CRC-32 covers header_var (n_ch..sub_len) || lpc_meta || subband_payload.
    # Magic and CRC field excluded. Matches lamquant-core::lml::compress and
    # lossless._compress_bytes — without header coverage, a single-byte flip
    # in any header field escapes detection.
    header_var = struct.pack('<HHBBII',
                             n_ch, T,
                             n_levels, flags,
                             len(lpc_meta), len(subband_payload))
    crc = zlib.crc32(header_var + payload) & 0xFFFFFFFF
    header = b'LML1' + header_var + struct.pack('<I', crc)

    nb = (flags >> 2) & 0x3F
    mode = "lossless" if nb == 0 else f"noise_bits={nb}"
    prefix = f"LML | {n_ch}ch | {mode} | CRC-32\n".encode("ascii")

    return prefix + bytes(header) + payload


if HAS_NUMBA:
    @numba.njit(cache=True, fastmath=False)
    def _process_all_channels_inverse(all_orders, all_coeffs, all_residuals,
                                      n_ch, ctx_len):
        """ONE numba call for the entire inverse pipeline.
        Calls canonical functions from lamquant_codec.ops.
        """
        signal_out = []
        for ch in range(n_ch):
            # Reconstruct subbands: bias restore + LPC synthesis
            subbands = []
            for sb_idx in range(4):
                flat_idx = ch * 4 + sb_idx
                order = all_orders[flat_idx]
                res = all_residuals[flat_idx]
                coeffs = all_coeffs[flat_idx]

                # Always run synthesize — encoder applies bias_cancel even for order=0
                subbands.append(lpc_synthesize_jit(res, coeffs, order, ctx_len))

            # 3-level inverse lifting: a3,d3 → a2; a2,d2 → a1; a1,d1 → signal
            a3, d3, d2, d1 = subbands[0], subbands[1], subbands[2], subbands[3]
            a2 = lifting_1d_inverse_int_jit(a3, d3)
            a1 = lifting_1d_inverse_int_jit(a2, d2)
            sig = lifting_1d_inverse_int_jit(a1, d1)
            signal_out.append(sig)

        return signal_out


def fused_decompress(data: bytes) -> np.ndarray:
    """Fused LML decompress: bytes → [C, T] int64.

    Byte-identical output to _decompress_bytes(). Faster due to
    single numba call for all channels.
    """
    from lamquant_codec.ops.golomb import decode_dense

    if len(data) < 4:
        raise LmlTruncatedError(f"Truncated LML data: {len(data)} bytes")

    # Scan past ASCII prefix: printable text + \n + 'LML' binary magic.
    offset = 0
    nl = data.find(b'\n')
    if (0 < nl < 128
            and all(0x20 <= b <= 0x7E for b in data[:nl])
            and len(data) > nl + 4
            and data[nl + 1:nl + 4] in (b'LML', b'LMQ')):
        offset = nl + 1
    data = data[offset:]

    if len(data) < 4:
        raise LmlTruncatedError("Truncated LML data after prefix")
    magic = data[:4]

    noise_bits = 0

    if magic == b'LML1':
        hdr_size = 22
        if len(data) < hdr_size:
            raise LmlTruncatedError(f"Truncated LML header: {len(data)} < {hdr_size}")
        (_, n_ch, T, n_levels, flags,
         lpc_len, sub_len, crc_expected) = struct.unpack('<4sHHBBIII', data[:hdr_size])

        # Reserved bits 0-1 must be zero per LML1 spec (docs/lml-format-v1.md §3.2).
        # Fail-closed: a non-zero reserved bit indicates either a corrupt header or
        # a future format extension this reader does not understand.
        if flags & 0x03:
            raise LmlReservedBitsSetError(
                f"LML1 reserved flag bits set (flags=0x{flags:02X}). "
                f"Header is either corrupt or written by an incompatible encoder.")

        noise_bits = (flags >> 2) & 0x3F

        # Header validation — prevent garbage dimensions from reaching numba
        if n_ch == 0 or n_ch > 1024:
            raise LmlHeaderError(f"Invalid channel count {n_ch}. Data is corrupt.")
        if T == 0 or T > 100_000_000:
            raise LmlHeaderError(f"Invalid sample count {T}. Data is corrupt.")
        if lpc_len + sub_len > len(data) - hdr_size:
            raise LmlTruncatedError(
                f"Payload length ({lpc_len}+{sub_len}) exceeds file size. "
                f"Data is truncated or corrupt.")

        # CRC-32 verification — covers header_var (4..18) + payload.
        # Mirrors encoder; magic and CRC field excluded. Shared helper tries
        # the modern scope first, then the legacy payload-only scope so
        # pre-a81cd04 artefacts still decode (DECODE-ONLY back-compat).
        from lamquant_codec.lossless import _verify_packet_crc
        payload = data[hdr_size:hdr_size + lpc_len + sub_len]
        header_var = data[4:18]
        _verify_packet_crc(header_var, payload, crc_expected)
        # Fused orchestrator requires exactly 3-level lifting
        if n_levels != 3:
            from lamquant_codec.lossless import _decompress_bytes_ref
            return _decompress_bytes_ref(data)

    elif magic[:3] == b'LML' and magic[3:4] in b'23456789':
        raise LmlVersionError(
            f"LML version {magic[3:4]!r} is newer than this reader "
            f"supports (max version 1). Update LamQuant.")
    elif magic in (b'LMQ4', b'LMQ5', b'LML '):
        raise LmlLegacyMagicError(
            f"Magic {magic!r} is from an earlier development iteration. "
            f"Use lamquant_codec.legacy.lossless_legacy.fused_decompress_legacy "
            f"to read it. The active production decoder accepts only LML1.")
    else:
        raise LmlMagicError(
            f"Not an LML packet (magic: {magic!r}). File may be corrupt.")

    pos = hdr_size
    lpc_data = data[pos:pos + lpc_len]
    pos += lpc_len
    lpc_pos = 0

    # Pre-copy payload for decode_dense
    data_arr = np.frombuffer(data, dtype=np.uint8).copy()

    # Parse all LPC metadata + decode all subbands (Python side)
    all_orders = np.empty(n_ch * 4, dtype=np.int64)
    all_coeffs = []
    all_residuals = []

    for i in range(n_ch * 4):
        order = lpc_data[lpc_pos]
        lpc_pos += 1
        coeffs = np.frombuffer(
            lpc_data[lpc_pos:lpc_pos + order * 4],
            dtype=np.int32).copy()
        lpc_pos += order * 4

        decoded, bytes_consumed = decode_dense(data_arr, pos)
        pos += bytes_consumed

        all_orders[i] = order
        all_coeffs.append(coeffs)
        all_residuals.append(decoded.astype(np.int64))

    # ONE numba call for all channels: bias restore + LPC synthesis + inverse lifting
    from lamquant_codec.ops.constants import BIAS_CTX_LEN
    ctx_len = np.int64(BIAS_CTX_LEN)
    channels = _process_all_channels_inverse(
        all_orders, all_coeffs, all_residuals, n_ch, ctx_len)

    signal_out = np.zeros((n_ch, T), dtype=np.int64)
    for ch in range(n_ch):
        sig = channels[ch]
        signal_out[ch, :len(sig)] = sig

    if noise_bits > 0:
        signal_out <<= noise_bits

    return signal_out.astype(np.float64)
