"""LamQuant Lossless (LML) — bit-exact EEG compression.

Integer lifting DWT + per-subband LPC + Golomb-Rice entropy coding.
No neural network involved.

LML per-window packet (22-byte header):
    [0:4]   'LML1' magic
    [4:6]   n_channels (uint16 LE)
    [6:8]   T samples per channel (uint16 LE)
    [8]     n_levels (uint8, lifting depth)
    [9]     flags (uint8: bits 0-1 reserved (MUST be 0), bits 2-7 = noise_bits)
    [10:14] LPC metadata length (uint32 LE)
    [14:18] subband payload length (uint32 LE)
    [18:22] CRC-32 of payload (uint32 LE)
    [22:]   LPC metadata + Golomb-Rice encoded residuals

Human-readable ASCII prefix before binary header:
    'LML | 21ch | lossless | CRC-32\\n'
"""
import struct
import numpy as np
from typing import List, Optional, Tuple

from lamquant_codec.codec_types import SubbandDecomposition, CompressedPacket
from lamquant_codec.errors import (
    LmlChannelCountError,
    LmlCrcError,
    LmlEmptySignalError,
    LmlLegacyMagicError,
    LmlMagicError,
    LmlNoiseStrippedError,
    LmlReservedBitsSetError,
    LmlSignalShapeError,
    LmlTruncatedError,
    LmlVersionError,
)

# Pure mechanism: Golomb-Rice primitives live in ops/.
from lamquant_codec.ops.golomb import encode_dense, decode_dense

# Bias cancellation (context-adaptive, +6% CR, Sriraam-style).
from lamquant_codec.ops.bias import cancel_jit as _bias_cancel_jit, restore_jit as _bias_restore_jit, BIAS_CTX_LEN as _BIAS_CTX_LEN

# Lifting primitives.
from lamquant_codec.ops.lifting import forward_nlevel_int as _lifting_nlevel_forward_raw
from lamquant_codec.ops.lifting import inverse_nlevel_int as _lifting_nlevel_inverse_raw


LIFT_PREC = 20       # Q20 fixed-point for lifting coefficients
LIFT_HALF = 1 << 19  # rounding bias = 2^(PREC-1)


# ---------------------------------------------------------------------------
# Legacy (pre-a81cd04) CRC back-compat — DECODE ONLY.
#
# ROOT CAUSE: commit a81cd04 (2026-05-11, "fix(lml): CRC covers packet header
# to detect single-byte header corruption") widened the LML1 per-window CRC-32
# scope from `crc32(lpc_meta || payload)` (legacy, payload-only) to
# `crc32(header[4..18] || lpc_meta || payload)` (modern) on BOTH encode and
# decode, with no version field in the LML1 header and no back-compat read
# path. Every file written before a81cd04 therefore fails CRC under the current
# reader even though its bytes are perfectly intact.
#
# This helper is the Python parity of `lamquant_core::lml::verify_packet_crc`:
# on a miss against the modern scope, recompute the legacy payload-only scope;
# if THAT matches, the packet is a valid pre-a81cd04 packet — accept it, latch
# `SAW_LEGACY_CRC`, and warn once. If both scopes miss, raise `LmlCrcError`.
#
# DECODE-ONLY: the encoder (`_compress_bytes`, `fused_compress`) is untouched
# and keeps writing the modern scope. The legacy scope is only ever recomputed
# on the read side, exactly mirroring the Rust fix.

# Monotonic latch: set True the first time any decode in this process accepts a
# packet via the legacy payload-only CRC scope. Never cleared automatically
# (matches the Rust `AtomicBool` — a process that has seen one legacy packet has
# seen one, for the life of the run). Tests may reset it explicitly to observe
# their own effect. Read it to detect that legacy artefacts are still in use.
SAW_LEGACY_CRC = False

# Warn-once guard so a multi-window legacy file emits a single notice, not one
# per window. Like `SAW_LEGACY_CRC`, deliberately never auto-cleared.
_WARNED_LEGACY_CRC = False


def _verify_packet_crc(header_var: bytes, payload: bytes, crc_expected: int) -> None:
    """Verify a per-window LML1 packet CRC, accepting the legacy scope on miss.

    Parameters mirror the wire layout:
      * ``header_var`` — the variable-fields header slice ``data[4:18]``
        (n_ch, T, n_levels, flags, lpc_len, sub_len). The constant 'LML1' magic
        (data[0:4]) and the CRC field itself (data[18:22]) are excluded.
      * ``payload`` — ``lpc_meta || subband_payload`` (the contiguous bytes the
        legacy encoder hashed payload-only).
      * ``crc_expected`` — the u32 stored in the packet header.

    Order of checks (fast path first, identical cost to the pre-fix decoder on
    the common case):
      1. MODERN scope ``crc32(header_var || payload)`` — return on match.
      2. LEGACY scope ``crc32(payload)`` — on match, latch ``SAW_LEGACY_CRC``,
         warn once, return. This is exactly the pre-a81cd04 payload-only scope
         (``reference_implementations/.../legacy/lossless_legacy.py:83``).
      3. Both miss → ``LmlCrcError`` reporting the MODERN actual (the scope the
         current encoder writes), so the genuine-corruption message only fires
         when the legacy scope ALSO failed.
    """
    global SAW_LEGACY_CRC, _WARNED_LEGACY_CRC
    import zlib

    crc_modern = zlib.crc32(header_var + bytes(payload)) & 0xFFFFFFFF
    if crc_modern == crc_expected:
        return

    crc_legacy = zlib.crc32(bytes(payload)) & 0xFFFFFFFF
    if crc_legacy == crc_expected:
        SAW_LEGACY_CRC = True
        if not _WARNED_LEGACY_CRC:
            _WARNED_LEGACY_CRC = True
            import warnings
            warnings.warn(
                "LML packet uses the legacy pre-2026-05-11 payload-only CRC "
                "scope; decoding via back-compat fallback. Re-encode to adopt "
                "the current header+payload CRC.",
                stacklevel=2,
            )
        return

    # Both scopes miss → genuine corruption. Report the modern actual: that is
    # the scope the current encoder writes, so it is the meaningful diagnostic.
    raise LmlCrcError(
        f"CRC-32 mismatch: expected 0x{crc_expected:08X}, got "
        f"0x{crc_modern:08X}. Data is corrupted (neither the current "
        f"header+payload CRC scope nor the legacy pre-2026-05-11 "
        f"payload-only scope matched).")


# ============================================================
# KLT / lifting-rotation primitives (pure, no torch).
# ============================================================

def compute_klt(training_signals) -> np.ndarray:
    """Compute a KLT matrix from training data [N, C, T]."""
    all_data = np.concatenate([s for s in training_signals], axis=1)
    cov = np.cov(all_data)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, idx].T.astype(np.float64)


def compute_lifting_rotations(klt_matrix: np.ndarray) -> List[Tuple[int, int, int, int, int]]:
    """Decompose KLT into lifting-based Givens rotations (integer-exact).

    Each 2D rotation R(θ) factors into 3 lifting steps:
      R(θ) = [[1,a],[0,1]] × [[1,0],[b,1]] × [[1,a],[0,1]]
    where a = -tan(θ/2), b = sin(θ). Q20 fixed-point coefficients make each
    step perfectly invertible (forward adds, inverse subtracts — no accumulated
    rounding error regardless of rotation count).

    Returns: list of (i, j, a_q20, b_q20, mode=0) tuples.
    """
    PREC = LIFT_PREC
    n = klt_matrix.shape[0]
    Q = klt_matrix.copy()
    rotations: List[Tuple[int, int, int, int, int]] = []

    for j in range(n):
        for i in range(n - 1, j, -1):
            a, b = Q[i - 1, j], Q[i, j]
            if abs(b) < 1e-15:
                continue
            r = np.sqrt(a * a + b * b)
            c, s = a / r, -b / r
            theta = np.arctan2(s, c)

            def _emit_rotation(angle, ri, rj):
                if abs(angle) < 1e-10:
                    return
                if abs(angle) <= np.pi / 3:
                    a_q = int(np.round(-np.tan(angle / 2.0) * (1 << PREC)))
                    b_q = int(np.round(np.sin(angle) * (1 << PREC)))
                    if a_q != 0 or b_q != 0:
                        rotations.append((ri, rj, a_q, b_q, 0))
                else:
                    _emit_rotation(angle / 2, ri, rj)
                    _emit_rotation(angle / 2, ri, rj)

            _emit_rotation(theta, i - 1, i)
            for k in range(n):
                q1, q2 = Q[i - 1, k], Q[i, k]
                Q[i - 1, k] = c * q1 - s * q2
                Q[i, k] = s * q1 + c * q2

    return rotations


# ---- Reference implementations (Python, slow, the SPEC) ----

def _lifting_forward_one_pyref(out, i, j, a_q, b_q, P):
    """Apply one lifting rotation forward (3 steps). Reference impl."""
    T = len(out[i])
    for t in range(T): out[i][t] += (a_q * out[j][t]) >> P
    for t in range(T): out[j][t] += (b_q * out[i][t]) >> P
    for t in range(T): out[i][t] += (a_q * out[j][t]) >> P


def _lifting_inverse_one_pyref(out, i, j, a_q, b_q, P):
    """Undo one lifting rotation. Reference impl."""
    T = len(out[i])
    for t in range(T): out[i][t] -= (a_q * out[j][t]) >> P
    for t in range(T): out[j][t] -= (b_q * out[i][t]) >> P
    for t in range(T): out[i][t] -= (a_q * out[j][t]) >> P


# ---- Vectorised implementations (numpy, ~100× faster) ----
# Replaces the Python triple-loop over T=2500 with numpy array ops.
# Bit-identical because numpy >> on int64 uses C-level arithmetic shift.

def _lifting_forward_one(out, i, j, a_q, b_q, P):
    """Apply one lifting rotation forward. Vectorised numpy."""
    out[i] += (np.int64(a_q) * out[j]) >> P
    out[j] += (np.int64(b_q) * out[i]) >> P
    out[i] += (np.int64(a_q) * out[j]) >> P


def _lifting_inverse_one(out, i, j, a_q, b_q, P):
    """Undo one lifting rotation. Vectorised numpy."""
    out[i] -= (np.int64(a_q) * out[j]) >> P
    out[j] -= (np.int64(b_q) * out[i]) >> P
    out[i] -= (np.int64(a_q) * out[j]) >> P


def apply_lifting_klt_forward(signal_int, rotations) -> np.ndarray:
    """Apply lifting-based KLT forward. Bit-exact invertible.

    Vectorised: each rotation is 3 numpy array ops over T samples
    instead of 3 Python loops. ~100× faster than the _pyref version
    for T=2500 × 210 rotations (1.575M ops → ~15 ms vs ~1.5 s).
    """
    P = LIFT_PREC
    out = signal_int.astype(np.int64).copy()
    for i, j, a_q, b_q, _ in rotations:
        _lifting_forward_one(out, i, j, a_q, b_q, P)
    return out


def apply_lifting_klt_inverse(signal_int, rotations) -> np.ndarray:
    """Apply inverse lifting-based KLT. Bit-exact inverse of forward."""
    P = LIFT_PREC
    out = signal_int.astype(np.int64).copy()
    for i, j, a_q, b_q, mode in reversed(rotations):
        _lifting_inverse_one(out, i, j, a_q, b_q, P)
    return out


# ============================================================
# N-level lifting (delegates to the tested implementation in the
# training codebase — avoids duplicating the reference algorithm).
# ============================================================

def _lifting_nlevel_forward(signal_int, n_levels):
    """N-level integer lifting. Returns dict of subbands."""
    return _lifting_nlevel_forward_raw(signal_int, n_levels)


def _lifting_nlevel_inverse(subbands, n_levels):
    """N-level integer inverse lifting."""
    return _lifting_nlevel_inverse_raw(subbands, n_levels)


# ============================================================
# Byte primitives — the wire format lives here.
# ============================================================

def _sub_lpc_schedule(n_levels: int) -> dict:
    """Per-subband LPC orders from the v7.6.1 empirical sweep:
      L_n approx: order 1 (nearly white after fullband LPC)
      L_n detail: order 1 (white noise)
      L_{n-1} detail: order 2 (mild spectral structure)
      L_k<n-1 detail: order 3 (more temporal correlation at higher freq)
    """
    sched = {f'l{n_levels}_approx': 1}
    for lvl in range(n_levels, 0, -1):
        if lvl == n_levels:
            sched[f'l{lvl}_detail'] = 1
        elif lvl == n_levels - 1:
            sched[f'l{lvl}_detail'] = 2
        else:
            sched[f'l{lvl}_detail'] = 3
    return sched


def _compress_bytes(signal: np.ndarray, *,
                    lifting_rots=None,
                    n_levels: int = 3,
                    noise_bits: int = 0) -> bytes:
    """Compress integer signal [C, T] → LML1 bytes.

    Uses fused pipeline (single numba call) when available and no KLT.
    Falls back to reference path for KLT, short signals, or no numba.
    """
    if lifting_rots is None:
        try:
            from lamquant_codec.ops.fused_lml import fused_compress, HAS_NUMBA
            if HAS_NUMBA:
                return fused_compress(signal, noise_bits=noise_bits)
        except ImportError:
            pass

    return _compress_bytes_ref(signal, lifting_rots=lifting_rots,
                               n_levels=n_levels, noise_bits=noise_bits)


def _compress_bytes_ref(signal: np.ndarray, *,
                        lifting_rots=None,
                        n_levels: int = 3,
                        noise_bits: int = 0) -> bytes:
    """Reference compress path. Handles KLT, variable n_levels, all edge cases."""
    from lamquant_codec.ops.lpc import analyze_channel as lpc_analyze_channel, analyze_int as lpc_analyze_int

    noise_bits = max(0, min(32, int(noise_bits)))
    if signal.ndim != 2:
        raise LmlSignalShapeError(
            f"Expected 2D signal [C, T], got {signal.ndim}D shape {signal.shape}")
    n_ch, T = signal.shape
    if n_ch > 1024:
        raise LmlChannelCountError(
            f"Signal has {n_ch} channels (shape {signal.shape}). "
            f"Expected [channels, time] with channels <= 1024. "
            f"Did you transpose the signal?")
    if n_ch == 0 or T == 0:
        raise LmlEmptySignalError(f"Cannot compress empty signal (shape {signal.shape})")
    signal_int = np.round(signal).astype(np.int64)

    # Strip noise bits: right-shift removes ADC noise floor.
    # Decoder left-shifts to restore scale. Bottom N bits become zero.
    if noise_bits > 0:
        # SAFETY: detect if data was already stripped (bottom bits all zero).
        # If the bottom noise_bits are ALREADY zero for the entire signal,
        # someone likely already stripped them. Refuse to strip again —
        # double-stripping silently destroys real data.
        mask = (1 << noise_bits) - 1
        bottom_energy = int(np.count_nonzero(signal_int & mask))
        if bottom_energy == 0 and T > 16:
            raise LmlNoiseStrippedError(
                f"noise_bits={noise_bits} requested but the bottom {noise_bits} "
                f"bits are ALREADY zero across all {n_ch}×{T} samples. "
                f"This signal was likely already noise-stripped. "
                f"Re-stripping would destroy real data. Use noise_bits=0 "
                f"to compress as-is, or check the provenance of this data.")
        signal_int >>= noise_bits

    # Adaptive n_levels: reduce lifting depth for short signals.
    # Each level halves the sample count. Need at least 4 samples
    # at the deepest level for LPC to work.
    while T < (4 * (1 << n_levels)) and n_levels > 0:
        n_levels -= 1

    # Flags byte: bits 0-1 reserved (MUST be 0 per LML1 spec), bits 2-7 = noise_bits (0-63).
    # KLT is an in-memory transform — caller is responsible for passing the same
    # lifting_rots to decode. The wire format does not signal KLT presence.
    if lifting_rots is not None:
        signal_int = apply_lifting_klt_forward(signal_int, lifting_rots)
    flags = (noise_bits & 0x3F) << 2

    sub_lpc = _sub_lpc_schedule(n_levels)
    subband_keys = [f'l{n_levels}_approx'] + \
                   [f'l{lvl}_detail' for lvl in range(n_levels, 0, -1)]

    # Sequential channel processing — threads are slower due to GIL
    lpc_parts = []
    payload_parts = []

    for ch in range(n_ch):
        subs = _lifting_nlevel_forward(signal_int[ch], n_levels)
        for key in subband_keys:
            sub_data = subs[key]
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
            corrected = _bias_cancel_jit(residual.astype(np.int64), _BIAS_CTX_LEN)
            lpc_parts.append(bytes([order]) + coeffs_q27.astype(np.int32).tobytes())
            payload_parts.append(encode_dense(corrected.astype(np.int64)))

    lpc_meta = b''.join(lpc_parts)
    subband_payload = b''.join(payload_parts)
    payload = lpc_meta + subband_payload

    # CRC-32 covers: variable-fields header (n_ch, T, n_levels, flags,
    # lpc_len, sub_len) || lpc_meta || subband_payload. Magic ('LML1') is
    # constant so excluded; CRC field itself is excluded to avoid self-
    # reference. Header coverage matters: without it, a one-byte flip in
    # any header field escapes detection (regression caught by
    # `lml::tests::crc_catches_header_corruption` in lamquant-core).
    header_var = struct.pack('<HHBBII',
                             n_ch, T,
                             n_levels, flags,
                             len(lpc_meta),
                             len(subband_payload))
    import zlib
    crc = zlib.crc32(header_var + payload) & 0xFFFFFFFF

    header = b'LML1' + header_var + struct.pack('<I', crc)

    nb = (flags >> 2) & 0x3F
    mode = "lossless" if nb == 0 else f"noise_bits={nb}"
    prefix = f"LML | {n_ch}ch | {mode} | CRC-32\n".encode("ascii")

    return prefix + bytes(header) + payload


def _decompress_bytes(data: bytes, *, lifting_rots=None) -> np.ndarray:
    """Lossless decompress: LML1 bytes → [C, T] float64 signal. Bit-exact.

    Uses fused pipeline (single numba call) when available and no KLT.
    Falls back to reference path for KLT or when numba is unavailable.

    Active path accepts ONLY LML1 magic. To read bytes from earlier
    development iterations (LMQ4, LMQ5, LML ), call
    `_decompress_legacy_bytes_ref` directly — it is not wired into any
    production code path.
    """
    if lifting_rots is None:
        try:
            from lamquant_codec.ops.fused_lml import fused_decompress, HAS_NUMBA
            if HAS_NUMBA:
                return fused_decompress(data)
        except ImportError:
            pass

    return _decompress_bytes_ref(data, lifting_rots=lifting_rots)


def _decompress_bytes_ref(data: bytes, *, lifting_rots=None) -> np.ndarray:
    """Reference decompress path — LML1 only.

    Accepts only the current LML1 magic. Legacy iterations (LMQ4, LMQ5, LML )
    are NOT handled here. To decode a legacy artefact, call
    `_decompress_legacy_bytes_ref` explicitly. The legacy decoder is not wired
    into any production path; it exists only so that historical bytes saved
    during development can still be read by hand.
    """
    from lamquant_codec.ops.lpc import synthesize_int as lpc_synthesize_int

    if len(data) < 4:
        raise LmlTruncatedError(f"Truncated LML data: {len(data)} bytes")

    # Scan past human-readable ASCII prefix.
    # Prefix: printable ASCII ending with \n, followed by 'LML' binary magic.
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

    if magic != b'LML1':
        if magic[:3] == b'LML' and magic[3:4] in b'23456789':
            raise LmlVersionError(
                f"LML version {magic[3:4].decode()} is newer than this "
                f"reader (LML1). Update LamQuant.")
        if magic in (b'LMQ4', b'LMQ5', b'LML '):
            raise LmlLegacyMagicError(
                f"Magic {magic!r} is from an earlier development iteration. "
                f"Use _decompress_legacy_bytes_ref to read it. The active "
                f"production decoder accepts only LML1.")
        raise LmlMagicError(f"Invalid LML magic: {magic!r}. File is corrupt.")

    hdr_size = 22  # magic(4) + n_ch(2) + T(2) + n_levels(1) + flags(1) + lpc(4) + sub(4) + crc(4)
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

    # CRC-32 verification — covers variable-fields header (4..18) plus
    # lpc_meta + subband payload. Mirrors the encoder side; magic and the
    # CRC field itself are excluded. `_verify_packet_crc` tries the modern
    # scope first, then falls back to the legacy payload-only scope so
    # pre-a81cd04 artefacts still decode (DECODE-ONLY back-compat).
    payload = data[hdr_size:hdr_size + lpc_len + sub_len]
    header_var = data[4:18]
    _verify_packet_crc(header_var, payload, crc_expected)

    # Integrity check: verify total data length matches header
    expected_len = hdr_size + lpc_len + sub_len
    if len(data) < expected_len:
        raise LmlTruncatedError(
            f"Truncated LML data: got {len(data)} bytes, "
            f"header declares {expected_len} (hdr={hdr_size} + "
            f"lpc={lpc_len} + payload={sub_len})")

    pos = hdr_size
    lpc_data = data[pos:pos + lpc_len]
    pos += lpc_len
    lpc_pos = 0

    subband_keys = [f'l{n_levels}_approx'] + \
                   [f'l{lvl}_detail' for lvl in range(n_levels, 0, -1)]

    signal_out = np.zeros((n_ch, T), dtype=np.int64)

    # Single copy of payload for numba decode (avoids per-subband copy)
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

            # Always run bias_restore — encoder applies bias_cancel even for order=0
            residual = _bias_restore_jit(decoded.astype(np.int64), _BIAS_CTX_LEN)
            if order == 0:
                subs[key] = residual
            else:
                subs[key] = lpc_synthesize_int(residual, coeffs_q27, order)

        signal_out[ch] = _lifting_nlevel_inverse(subs, n_levels)

    # KLT inverse: applied if caller passed lifting_rots. The wire format
    # does not signal KLT presence in LML1 — the caller is responsible for
    # passing the same lifting_rots on both sides.
    if lifting_rots is not None:
        signal_out = apply_lifting_klt_inverse(signal_out, lifting_rots)

    # Restore noise bits: left-shift to recover original scale.
    # Bottom N bits are zeroed — this is the only data change in noise_bits mode.
    if noise_bits > 0:
        signal_out <<= noise_bits

    return signal_out.astype(np.float64)


def peek_header(data: bytes):
    """Read an LML1 packet header without decompressing.

    Active production peek — accepts only LML1 magic. For legacy iterations
    (LMQ4, LMQ5), call `peek_header_legacy` explicitly.

    Returns LQPacketHeader (typed dataclass).
    """
    from lamquant_codec.file_info import LQPacketHeader

    if len(data) < 4:
        raise LmlTruncatedError(f"Truncated: {len(data)} bytes")
    offset = 0
    nl = data.find(b'\n')
    if (0 < nl < 128
            and all(0x20 <= b <= 0x7E for b in data[:nl])
            and len(data) > nl + 4
            and data[nl + 1:nl + 4] == b'LML'):
        offset = nl + 1
    data = data[offset:]
    if len(data) < 4:
        raise LmlTruncatedError(f"Truncated after prefix")
    magic = data[:4]
    if magic != b'LML1':
        if magic in (b'LMQ4', b'LMQ5', b'LML '):
            raise LmlLegacyMagicError(
                f"Magic {magic!r} is a legacy iteration. "
                f"Use peek_header_legacy to read it.")
        raise LmlMagicError(f"Invalid magic: {magic!r}")
    if len(data) < 22:
        raise LmlTruncatedError(f"Truncated LML1 header")
    _, n_ch, T, n_levels, flags, lpc_len, sub_len, crc = \
        struct.unpack('<4sHHBBIII', data[:22])
    if flags & 0x03:
        raise LmlReservedBitsSetError(
            f"LML1 reserved flag bits set (flags=0x{flags:02X}).")
    nb = (flags >> 2) & 0x3F
    return LQPacketHeader(
        version='LML1', n_channels=n_ch, n_samples=T,
        n_levels=n_levels, klt=False,
        noise_bits=nb, lossless=(nb == 0),
        lpc_meta_bytes=lpc_len, payload_bytes=sub_len,
        crc32=crc, total_bytes=22 + lpc_len + sub_len,
    )


# ============================================================
# Stateful wrapper (back-compat) — used to live in codec.py.
# ============================================================

class LosslessCodec:
    """Mode 3: Lossless EEG compression using only DSP.

    Pipeline:
      Encode: KLT spatial decorrelation (optional) → integer Le Gall 5/3
              lifting (N-level) → adaptive-order LPC per subband →
              Golomb-Rice entropy coding
      Decode: inverse GR → inverse LPC → inverse lifting → inverse KLT
    """

    # Expose the Q20 constants used by external callers / tests.
    LIFT_PREC = LIFT_PREC
    LIFT_HALF = LIFT_HALF

    def __init__(self, klt_matrix=None, lpc_order=8, n_levels=3):
        self.lpc_order = lpc_order
        self.n_levels = n_levels
        self.klt = klt_matrix
        self.klt_inv = klt_matrix.T if klt_matrix is not None else None
        self.lifting_rots = (compute_lifting_rotations(klt_matrix)
                             if klt_matrix is not None else None)

    # Static helpers kept for back-compat with callers that used the class.
    compute_klt = staticmethod(compute_klt)
    compute_lifting_rotations = staticmethod(compute_lifting_rotations)
    apply_lifting_klt_forward = staticmethod(apply_lifting_klt_forward)
    apply_lifting_klt_inverse = staticmethod(apply_lifting_klt_inverse)
    _lifting_forward_one = staticmethod(_lifting_forward_one)
    _lifting_inverse_one = staticmethod(_lifting_inverse_one)
    _lifting_nlevel_forward = staticmethod(_lifting_nlevel_forward)
    _lifting_nlevel_inverse = staticmethod(_lifting_nlevel_inverse)

    def compress(self, signal):
        """Lossless compress: signal [C, T] → bytes."""
        return _compress_bytes(signal, lifting_rots=self.lifting_rots,
                               n_levels=self.n_levels)

    def decompress(self, data):
        """Lossless decompress: bytes → signal [C, T]. Bit-perfect."""
        return _decompress_bytes(data, lifting_rots=self.lifting_rots)

    def compress_to_packet(self, signal):
        """Compress + decompress + wrap in an EEGPacket (convenience)."""
        # Local import avoids pulling EEGPacket at module load time.
        from lamquant_codec.codec_types import EEGPacket
        compressed = self.compress(signal)
        recon = self.decompress(compressed)
        return EEGPacket.from_lossless(
            signal=recon,
            compressed_bytes=len(compressed),
            metadata={'n_levels': self.n_levels, 'lpc_order': self.lpc_order},
        )


# ============================================================
# Typed pipeline entry (no class, no state — for Unix-style callers).
# ============================================================

def compress(subband: SubbandDecomposition, *, klt_matrix=None,
             lpc_order: int = 8, n_levels: int = 3) -> CompressedPacket:
    """Encode a SubbandDecomposition losslessly (bit-exact).

    Args:
        subband:     Must have `source_signal` populated (the raw [C, T]
                     signal before decompose() was called).
        klt_matrix:  Optional KLT for spatial decorrelation.
        lpc_order:   LPC order hint (only used as fallback in schedule).
        n_levels:    Lifting decomposition depth (default 3).

    Returns:
        CompressedPacket with mode='lossless' (LML1 bytes).
    """
    if subband.source_signal is None:
        raise ValueError("lossless.compress requires subband.source_signal "
                         "(the original [C, T] signal before decompose).")

    rots = compute_lifting_rotations(klt_matrix) if klt_matrix is not None else None
    data = _compress_bytes(np.asarray(subband.source_signal, dtype=np.float64),
                           lifting_rots=rots, n_levels=n_levels)

    C, T = subband.source_signal.shape
    return CompressedPacket(
        data=data,
        mode='lossless',
        raw_bytes=C * T * 2,
        quality_mode=2,
        metadata={'lpc_order': lpc_order, 'n_levels': n_levels},
    )


def decompress(packet: CompressedPacket, *, klt_matrix=None) -> np.ndarray:
    """Decode a lossless CompressedPacket back to the raw signal."""
    if packet.mode != 'lossless':
        raise ValueError(f"lossless.decompress got packet.mode={packet.mode!r}")
    rots = compute_lifting_rotations(klt_matrix) if klt_matrix is not None else None
    return _decompress_bytes(packet.data, lifting_rots=rots)


__all__ = [
    'compress', 'decompress',
    'LosslessCodec',
    '_compress_bytes', '_decompress_bytes',
    'compute_klt', 'compute_lifting_rotations',
    'apply_lifting_klt_forward', 'apply_lifting_klt_inverse',
    'LIFT_PREC', 'LIFT_HALF',
]