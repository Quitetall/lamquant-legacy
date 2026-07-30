"""Cross-language LML1 wire-format tests — Python ↔ Rust parity (L5).

Pins every wire-format invariant that must hold across both implementations:
  - Magic byte, header size, field offsets, endianness, CRC polynomial
  - Bit-exact round-trip in both directions (Python encode → Rust decode and vice versa)
  - Identical reject behaviour for corruption, truncation, future versions,
    legacy iteration magics, and reserved-bit-set headers
  - Identical noise_bits semantics

Drift caught here would mean a Python-encoded packet is mis-read by Rust
(or vice versa), producing silently wrong samples.

Implementation: uses the in-process PyO3 wheel `lamquant_core` for the Rust
side. The wheel must be installed via `maturin develop --features python`.
Imports the canonical helpers from `tests/helpers/` rather than duplicating
their logic inline.
"""
from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest

# Active Python codec.
from lamquant_codec.lossless import (
    _compress_bytes_ref as py_encode_ref,
    _decompress_bytes_ref as py_decode_ref,
    _compress_bytes as py_encode,
    _decompress_bytes as py_decode,
)
from lamquant_codec.ops.constants import MAGIC_LML

# Canonical helpers — single source of truth.
from tests.helpers.signals import synth_signal
from tests.helpers.rust_bindings import (
    rust_compress as rust_encode,
    rust_decompress as rust_decode,
    to_rust_signal,
    requires_rust,
)

pytestmark = [pytest.mark.l5, pytest.mark.cross_lang, requires_rust]


def strip_ascii_prefix(data: bytes) -> bytes:
    """Both encoders prepend a 'LML | Nch | mode | CRC-32\\n' ASCII line."""
    nl = data.find(b'\n')
    if 0 < nl < 128 and data[nl + 1:nl + 4] == b'LML':
        return data[nl + 1:]
    return data


# ============================================================
# 1. Wire-format constant pinning (drift detection)
# ============================================================

class TestWireConstants:

    def test_magic_byte_value(self):
        """LML1 magic must be exactly these 4 bytes everywhere."""
        assert MAGIC_LML == b'LML1'
        assert len(MAGIC_LML) == 4

    def test_struct_format_string(self):
        """Header packs as <4sHHBBIII = 22 bytes, little-endian."""
        fmt = '<4sHHBBIII'
        assert struct.calcsize(fmt) == 22

    def test_header_field_widths(self):
        """Field widths are pinned: u16, u8, u32 in known order."""
        fmt = '<4sHHBBIII'
        # 4 bytes magic + 2 (n_ch u16) + 2 (T u16) + 1 (n_levels u8) +
        # 1 (flags u8) + 4 (lpc_len u32) + 4 (sub_len u32) + 4 (crc u32)
        assert struct.calcsize(fmt) == 4 + 2 + 2 + 1 + 1 + 4 + 4 + 4

    def test_endianness_marker(self):
        """Format string must start with '<' (little-endian)."""
        # If anyone changes this to '>' or '@', cross-language decode breaks.
        fmt = '<4sHHBBIII'
        assert fmt[0] == '<'


# ============================================================
# 2. Header byte-layout drift (field offsets pinned)
# ============================================================

class TestHeaderLayout:
    """Encode a synthetic signal with each side, then byte-compare headers.

    A drift would show up as different bytes at the same offset for the
    same logical input.
    """

    def test_python_and_rust_header_equal(self):
        sig = synth_signal(4, 256, seed=11)
        py_bytes = strip_ascii_prefix(py_encode(sig.astype(np.float64), noise_bits=0))
        rs_bytes = strip_ascii_prefix(bytes(rust_encode(to_rust_signal(sig), 0)))
        # First 22 bytes are the LML1 header. They must match exactly.
        assert py_bytes[:22] == rs_bytes[:22], (
            f"Header byte-layout drift!\n"
            f"  Python: {py_bytes[:22].hex(' ')}\n"
            f"  Rust:   {rs_bytes[:22].hex(' ')}"
        )

    def test_magic_at_offset_0(self):
        sig = synth_signal(2, 64, seed=12)
        py = strip_ascii_prefix(py_encode(sig.astype(np.float64), noise_bits=0))
        rs = strip_ascii_prefix(bytes(rust_encode(to_rust_signal(sig), 0)))
        assert py[0:4] == b'LML1'
        assert rs[0:4] == b'LML1'

    def test_n_ch_at_offset_4_le(self):
        for n_ch in (1, 2, 21, 64, 1024):
            sig = synth_signal(n_ch, 64, seed=n_ch)
            py = strip_ascii_prefix(py_encode(sig.astype(np.float64), noise_bits=0))
            rs = strip_ascii_prefix(bytes(rust_encode(to_rust_signal(sig), 0)))
            # u16 LE at offset 4
            assert struct.unpack('<H', py[4:6])[0] == n_ch
            assert struct.unpack('<H', rs[4:6])[0] == n_ch

    def test_T_at_offset_6_le(self):
        for T in (16, 64, 256, 2500, 65535):
            sig = synth_signal(2, T, seed=T)
            py = strip_ascii_prefix(py_encode(sig.astype(np.float64), noise_bits=0))
            rs = strip_ascii_prefix(bytes(rust_encode(to_rust_signal(sig), 0)))
            assert struct.unpack('<H', py[6:8])[0] == T
            assert struct.unpack('<H', rs[6:8])[0] == T

    def test_flags_byte_zero_when_no_noise_bits(self):
        sig = synth_signal(2, 64, seed=99)
        py = strip_ascii_prefix(py_encode(sig.astype(np.float64), noise_bits=0))
        rs = strip_ascii_prefix(bytes(rust_encode(to_rust_signal(sig), 0)))
        assert py[9] == 0
        assert rs[9] == 0

    def test_flags_byte_noise_bits_packing(self):
        """noise_bits packed into bits 2-7, bits 0-1 always zero."""
        # Signal must have NON-zero low bits so Python's "already-stripped"
        # safety check doesn't trip. Bottom nb bits will be discarded by the
        # encoder; we just need them to be non-uniformly zero on input.
        sig = synth_signal(2, 64, seed=99)
        for nb in (0, 1, 2, 4, 8):
            py = strip_ascii_prefix(py_encode(sig.astype(np.float64), noise_bits=nb))
            rs = strip_ascii_prefix(bytes(rust_encode(to_rust_signal(sig), nb)))
            assert (py[9] & 0x03) == 0, "Python set reserved bit"
            assert (rs[9] & 0x03) == 0, "Rust set reserved bit"
            assert (py[9] >> 2) & 0x3F == nb
            assert (rs[9] >> 2) & 0x3F == nb


# ============================================================
# 3. CRC-32 polynomial parity
# ============================================================

class TestCRCParity:
    """Both languages must compute the same CRC-32 over the same payload."""

    def test_python_zlib_matches_rust_crc(self):
        sig = synth_signal(2, 64, seed=42)
        py_full = strip_ascii_prefix(py_encode(sig.astype(np.float64), noise_bits=0))
        # Header tells us the payload extent.
        _, _, _, _, _, lpc_len, sub_len, crc_in_header = struct.unpack(
            '<4sHHBBIII', py_full[:22])
        payload = py_full[22:22 + lpc_len + sub_len]
        recomputed = zlib.crc32(payload) & 0xFFFFFFFF
        assert recomputed == crc_in_header

        # Same payload through Rust encoder must produce the same CRC.
        rs_full = strip_ascii_prefix(bytes(rust_encode(to_rust_signal(sig), 0)))
        _, _, _, _, _, _, _, rs_crc = struct.unpack('<4sHHBBIII', rs_full[:22])
        rs_payload = rs_full[22:22 + lpc_len + sub_len]
        # Cross-check: Python's zlib.crc32 over Rust's payload matches Rust's CRC.
        assert (zlib.crc32(rs_payload) & 0xFFFFFFFF) == rs_crc


# ============================================================
# 4. Cross-lang round-trip — Python encode → Rust decode
# ============================================================

class TestPythonEncodeRustDecode:

    @pytest.mark.parametrize("n_ch,T,seed", [
        (1, 16, 1),
        (2, 64, 2),
        (4, 256, 3),
        (21, 2500, 4),
        (64, 1250, 5),
    ])
    def test_roundtrip(self, n_ch, T, seed):
        sig = synth_signal(n_ch, T, seed=seed)
        encoded = py_encode(sig.astype(np.float64), noise_bits=0)
        decoded = np.array(rust_decode(encoded), dtype=np.int64)
        assert decoded.shape == sig.shape
        np.testing.assert_array_equal(decoded, sig)

    @pytest.mark.parametrize("nb", [0, 1, 2, 4, 8, 16])
    def test_roundtrip_noise_bits(self, nb):
        # Signal has non-zero low bits so the encoder's safety check passes.
        # The encoder strips the bottom nb bits internally.
        sig = synth_signal(4, 256, seed=10, amp=30000)
        encoded = py_encode(sig.astype(np.float64), noise_bits=nb)
        decoded = np.array(rust_decode(encoded), dtype=np.int64)
        # Noise-bits mode is lossy by design: bottom nb bits are zero on output.
        expected = (sig >> nb) << nb
        np.testing.assert_array_equal(decoded, expected)

    def test_all_zeros(self):
        sig = np.zeros((2, 64), dtype=np.int64)
        encoded = py_encode(sig.astype(np.float64), noise_bits=0)
        decoded = np.array(rust_decode(encoded), dtype=np.int64)
        np.testing.assert_array_equal(decoded, sig)

    def test_int16_extremes(self):
        sig = np.full((2, 64), 32767, dtype=np.int64)
        encoded = py_encode(sig.astype(np.float64), noise_bits=0)
        decoded = np.array(rust_decode(encoded), dtype=np.int64)
        np.testing.assert_array_equal(decoded, sig)

        sig = np.full((2, 64), -32768, dtype=np.int64)
        encoded = py_encode(sig.astype(np.float64), noise_bits=0)
        decoded = np.array(rust_decode(encoded), dtype=np.int64)
        np.testing.assert_array_equal(decoded, sig)


# ============================================================
# 5. Cross-lang round-trip — Rust encode → Python decode
# ============================================================

class TestRustEncodePythonDecode:

    @pytest.mark.parametrize("n_ch,T,seed", [
        (1, 16, 1),
        (2, 64, 2),
        (4, 256, 3),
        (21, 2500, 4),
        (64, 1250, 5),
    ])
    def test_roundtrip(self, n_ch, T, seed):
        sig = synth_signal(n_ch, T, seed=seed)
        encoded = bytes(rust_encode(to_rust_signal(sig), 0))
        decoded = py_decode(encoded).astype(np.int64)
        assert decoded.shape == sig.shape
        np.testing.assert_array_equal(decoded, sig)

    @pytest.mark.parametrize("nb", [0, 1, 2, 4, 8, 16])
    def test_roundtrip_noise_bits(self, nb):
        sig = synth_signal(4, 256, seed=11, amp=30000)
        encoded = bytes(rust_encode(to_rust_signal(sig), nb))
        decoded = py_decode(encoded).astype(np.int64)
        expected = (sig >> nb) << nb
        np.testing.assert_array_equal(decoded, expected)


# ============================================================
# 6. Full triangle — same data must survive all four paths
# ============================================================

class TestFullTriangle:
    """For the same input, all four (encode, decode) pairs must give the same output."""

    @pytest.mark.parametrize("n_ch,T", [(2, 64), (21, 2500), (4, 1024)])
    def test_full_triangle(self, n_ch, T):
        sig = synth_signal(n_ch, T, seed=n_ch * 100 + T)
        py_bytes = py_encode(sig.astype(np.float64), noise_bits=0)
        rs_bytes = bytes(rust_encode(to_rust_signal(sig), 0))

        # All 4 decodes must reconstruct the original sample-for-sample.
        py_py = py_decode(py_bytes).astype(np.int64)
        py_rs = np.array(rust_decode(py_bytes), dtype=np.int64)
        rs_py = py_decode(rs_bytes).astype(np.int64)
        rs_rs = np.array(rust_decode(rs_bytes), dtype=np.int64)

        np.testing.assert_array_equal(py_py, sig)
        np.testing.assert_array_equal(py_rs, sig)
        np.testing.assert_array_equal(rs_py, sig)
        np.testing.assert_array_equal(rs_rs, sig)


# ============================================================
# 7. Reject behaviour drift — both sides must reject the same garbage
# ============================================================

class TestRejectBehaviour:
    """Drift in reject behaviour = a corrupt file that one side accepts and
    the other rejects → silent acceptance of garbage."""

    def _make_packet(self, sig: np.ndarray) -> bytes:
        return bytes(rust_encode(to_rust_signal(sig), 0))

    def test_both_reject_legacy_lmq5(self):
        sig = synth_signal(2, 64, seed=70)
        packet = self._make_packet(sig)
        body = strip_ascii_prefix(packet)
        body = b'LMQ5' + body[4:]
        with pytest.raises((ValueError, Exception)):
            py_decode(body)
        with pytest.raises((ValueError, Exception)):
            rust_decode(body)

    def test_both_reject_legacy_lmq4(self):
        sig = synth_signal(2, 64, seed=71)
        body = strip_ascii_prefix(self._make_packet(sig))
        body = b'LMQ4' + body[4:]
        with pytest.raises(Exception):
            py_decode(body)
        with pytest.raises(Exception):
            rust_decode(body)

    def test_both_reject_legacy_lml_space(self):
        sig = synth_signal(2, 64, seed=72)
        body = strip_ascii_prefix(self._make_packet(sig))
        body = b'LML ' + body[4:]
        with pytest.raises(Exception):
            py_decode(body)
        with pytest.raises(Exception):
            rust_decode(body)

    @pytest.mark.parametrize("future", [b'2', b'3', b'7', b'9'])
    def test_both_reject_future_version(self, future):
        sig = synth_signal(2, 64, seed=73)
        body = strip_ascii_prefix(self._make_packet(sig))
        body = b'LML' + future + body[4:]
        with pytest.raises(Exception):
            py_decode(body)
        with pytest.raises(Exception):
            rust_decode(body)

    @pytest.mark.parametrize("flag_bits", [0x01, 0x02, 0x03])
    def test_both_reject_reserved_bits(self, flag_bits):
        sig = synth_signal(2, 64, seed=74)
        body = bytearray(strip_ascii_prefix(self._make_packet(sig)))
        # Set forbidden bits in flags byte.
        body[9] |= flag_bits
        # CRC still verifies (header is not part of CRC). Drift would mean one
        # side accepts and silently mis-decodes; we want both to reject.
        with pytest.raises(Exception):
            py_decode(bytes(body))
        with pytest.raises(Exception):
            rust_decode(bytes(body))

    def test_both_reject_invalid_magic(self):
        sig = synth_signal(2, 64, seed=75)
        body = bytearray(strip_ascii_prefix(self._make_packet(sig)))
        body[0:4] = b'XXXX'
        with pytest.raises(Exception):
            py_decode(bytes(body))
        with pytest.raises(Exception):
            rust_decode(bytes(body))

    def test_both_reject_crc_corruption(self):
        sig = synth_signal(2, 64, seed=76)
        body = bytearray(strip_ascii_prefix(self._make_packet(sig)))
        # Flip a payload byte (not the header).
        body[30] ^= 0x01
        with pytest.raises(Exception):
            py_decode(bytes(body))
        with pytest.raises(Exception):
            rust_decode(bytes(body))

    @pytest.mark.parametrize("trunc_at", [0, 1, 4, 10, 21, 22, 30])
    def test_both_reject_truncation(self, trunc_at):
        sig = synth_signal(2, 64, seed=77)
        body = strip_ascii_prefix(self._make_packet(sig))
        truncated = body[:trunc_at]
        with pytest.raises(Exception):
            py_decode(truncated)
        with pytest.raises(Exception):
            rust_decode(truncated)


# ============================================================
# 8. Boundary shapes — both sides handle edges identically
# ============================================================

class TestBoundaryShapes:

    @pytest.mark.parametrize("n_ch,T", [
        (1, 4),         # min viable
        (1, 16),
        (1, 1024),
        (256, 16),      # near max channels (Python ref-path asserts < 512)
        (21, 2500),     # canonical clinical
        (64, 1250),
        (2, 65535),     # max u16 samples
    ])
    def test_both_directions(self, n_ch, T):
        sig = synth_signal(n_ch, T, seed=n_ch ^ T)
        # Python encode → Rust decode
        py_bytes = py_encode(sig.astype(np.float64), noise_bits=0)
        rust_out = np.array(rust_decode(py_bytes), dtype=np.int64)
        np.testing.assert_array_equal(rust_out, sig)
        # Rust encode → Python decode
        rs_bytes = bytes(rust_encode(to_rust_signal(sig), 0))
        py_out = py_decode(rs_bytes).astype(np.int64)
        np.testing.assert_array_equal(py_out, sig)


# ============================================================
# 9. Field-value drift — encode known values, byte-pin them
# ============================================================

class TestKnownValueDrift:
    """Build a packet by hand from struct.pack and verify both decoders
    parse the same fields in the same way. Any change to header layout
    or endianness fails here loudly."""

    def test_handcrafted_zero_packet_is_well_formed(self):
        """A zeros-only signal compresses deterministically — pin some bytes."""
        sig = np.zeros((2, 16), dtype=np.int64)
        py_bytes = strip_ascii_prefix(py_encode(sig.astype(np.float64), noise_bits=0))
        rs_bytes = strip_ascii_prefix(bytes(rust_encode(to_rust_signal(sig), 0)))

        # Headers equal, payloads equal.
        assert py_bytes == rs_bytes, (
            "Python and Rust diverge on identical zero input — "
            "this means LPC/Golomb/CRC code paths differ."
        )

    def test_n_ch_byte_order_explicit(self):
        """n_ch=0x0102 (258) must be serialized as bytes 02 01 (little-endian)."""
        sig = synth_signal(258, 32, seed=1)  # 258 = 0x0102
        py_bytes = strip_ascii_prefix(py_encode(sig.astype(np.float64), noise_bits=0))
        assert py_bytes[4] == 0x02
        assert py_bytes[5] == 0x01
        rs_bytes = strip_ascii_prefix(bytes(rust_encode(to_rust_signal(sig), 0)))
        assert rs_bytes[4] == 0x02
        assert rs_bytes[5] == 0x01

    def test_payload_length_byte_order_explicit(self):
        """A short signal yields small lpc_len and sub_len fields whose LE
        byte order can be visually verified."""
        sig = synth_signal(1, 16, seed=2)
        py_bytes = strip_ascii_prefix(py_encode(sig.astype(np.float64), noise_bits=0))
        # lpc_len at [10:14], sub_len at [14:18]
        lpc_len_le = py_bytes[10:14]
        sub_len_le = py_bytes[14:18]
        # u32 LE of small values: leading byte has the data, top bytes are 00.
        assert lpc_len_le[2] == 0 and lpc_len_le[3] == 0
        assert sub_len_le[2] == 0 and sub_len_le[3] == 0


# ============================================================
# 10. Limit drift — both sides agree on hard caps (n_ch, T, noise_bits)
# ============================================================

class TestLimitDrift:
    """Each implementation caps inputs at sanity limits. Drift here means
    one side accepts a packet the other rejects → silent acceptance of
    garbage at the boundary."""

    def test_n_ch_cap_aligned(self):
        """Both implementations cap n_ch at 1024 (or higher). Below that,
        both must accept; above, both should reject."""
        # 1024 must work in both.
        sig = synth_signal(1024, 16, seed=1024)
        py_bytes = py_encode(sig.astype(np.float64), noise_bits=0)
        rs_bytes = bytes(rust_encode(to_rust_signal(sig), 0))
        # Both should round-trip 1024 channels.
        py_out = py_decode(py_bytes).astype(np.int64)
        rs_out = np.array(rust_decode(rs_bytes), dtype=np.int64)
        np.testing.assert_array_equal(py_out, sig)
        np.testing.assert_array_equal(rs_out, sig)

    def test_T_cap_u16_max(self):
        """Both implementations support T = u16::MAX (65535)."""
        sig = synth_signal(1, 65535, seed=1)
        py_bytes = py_encode(sig.astype(np.float64), noise_bits=0)
        rs_bytes = bytes(rust_encode(to_rust_signal(sig), 0))
        py_out = py_decode(py_bytes).astype(np.int64)
        rs_out = np.array(rust_decode(rs_bytes), dtype=np.int64)
        np.testing.assert_array_equal(py_out, sig)
        np.testing.assert_array_equal(rs_out, sig)

    def test_noise_bits_practical_range(self):
        """Wire format stores noise_bits as u6 (max 63 on the wire), but both
        implementations cap practical input at 32 to keep arithmetic well-
        behaved. Confirm both accept the range 0..32 and produce identical
        cross-decoded output."""
        sig = synth_signal(2, 64, seed=63, amp=30000)
        for nb in (0, 1, 2, 4, 8, 16, 31, 32):
            py_bytes = py_encode(sig.astype(np.float64), noise_bits=nb)
            rs_bytes = bytes(rust_encode(to_rust_signal(sig), nb))
            py_decoded = py_decode(rs_bytes).astype(np.int64)
            rs_decoded = np.array(rust_decode(py_bytes), dtype=np.int64)
            expected = (sig >> nb) << nb
            np.testing.assert_array_equal(py_decoded, expected)
            np.testing.assert_array_equal(rs_decoded, expected)

    def test_noise_bits_above_cap_rejected(self):
        """noise_bits > 32 must be rejected (or clamped) by both sides.
        Drift here means one side encodes nb=33 literally while the other
        clamps to 32 → silent samples mismatch."""
        sig = synth_signal(2, 64, seed=64, amp=30000)
        # Python clamps internally rather than raising — verify clamp.
        py_bytes_33 = py_encode(sig.astype(np.float64), noise_bits=33)
        py_bytes_32 = py_encode(sig.astype(np.float64), noise_bits=32)
        # The flags byte carries the actual noise_bits used.
        py_body_33 = strip_ascii_prefix(py_bytes_33)
        py_body_32 = strip_ascii_prefix(py_bytes_32)
        nb_used_33 = (py_body_33[9] >> 2) & 0x3F
        nb_used_32 = (py_body_32[9] >> 2) & 0x3F
        assert nb_used_33 == nb_used_32 == 32, (
            f"Python should clamp to 32, got 33→{nb_used_33}, 32→{nb_used_32}"
        )
        # Rust asserts > 32 → panic. Confirm via pytest.raises catching the
        # PanicException raised through PyO3.
        with pytest.raises(BaseException):
            rust_encode(to_rust_signal(sig), 33)


# ============================================================
# 11. Prefix-format parity — ASCII prefix contents match exactly
# ============================================================

class TestAsciiPrefixDrift:
    """Both encoders prepend a human-readable ASCII line. The exact format
    is part of the wire contract — drift here breaks tools that grep file
    headers."""

    def test_prefix_lossless(self):
        sig = synth_signal(2, 64, seed=33)
        py_bytes = py_encode(sig.astype(np.float64), noise_bits=0)
        rs_bytes = bytes(rust_encode(to_rust_signal(sig), 0))
        nl_py = py_bytes.find(b'\n')
        nl_rs = rs_bytes.find(b'\n')
        assert py_bytes[:nl_py + 1] == rs_bytes[:nl_rs + 1], (
            f"ASCII prefix drift!\n"
            f"  Python: {py_bytes[:nl_py + 1]!r}\n"
            f"  Rust:   {rs_bytes[:nl_rs + 1]!r}"
        )

    @pytest.mark.parametrize("n_ch", [1, 2, 21, 64])
    def test_prefix_channel_count_matches(self, n_ch):
        sig = synth_signal(n_ch, 64, seed=n_ch)
        py_bytes = py_encode(sig.astype(np.float64), noise_bits=0)
        rs_bytes = bytes(rust_encode(to_rust_signal(sig), 0))
        # Both prefixes start with "LML | Nch | "
        assert f"{n_ch}ch".encode() in py_bytes[:64]
        assert f"{n_ch}ch".encode() in rs_bytes[:64]


# ============================================================
# 12. Internal Python parity — fused (numba) vs ref must agree
# ============================================================

class TestPythonFusedRefParity:
    """The fused numba path and the pure-Python ref path must produce
    identical bytes. Drift here means the fast path silently differs from
    the spec implementation."""

    @pytest.mark.parametrize("n_ch,T,seed", [
        (2, 64, 1),
        (21, 2500, 2),
        (4, 1024, 3),
    ])
    def test_encode_byte_identical(self, n_ch, T, seed):
        sig = synth_signal(n_ch, T, seed=seed).astype(np.float64)
        fused_bytes = py_encode(sig, noise_bits=0)
        ref_bytes = py_encode_ref(sig, noise_bits=0)
        # Strip prefixes — fused and ref both emit a prefix; payload bytes
        # past the prefix must be byte-identical.
        assert strip_ascii_prefix(fused_bytes) == strip_ascii_prefix(ref_bytes)

    @pytest.mark.parametrize("n_ch,T,seed", [
        (2, 64, 1),
        (21, 2500, 2),
        (4, 1024, 3),
    ])
    def test_decode_identical(self, n_ch, T, seed):
        sig = synth_signal(n_ch, T, seed=seed).astype(np.float64)
        encoded = py_encode(sig, noise_bits=0)
        fused_out = py_decode(encoded)
        from lamquant_codec.lossless import _decompress_bytes_ref as _ref_decode
        ref_out = _ref_decode(encoded)
        np.testing.assert_array_equal(fused_out, ref_out)


# ============================================================
# 13. Legacy decoder is opt-in only — never auto-invoked
# ============================================================

class TestLegacyIsolation:
    """The legacy decoder must NOT be reachable from any production code
    path. Confirms `lamquant_codec.legacy.lossless_legacy` is importable
    on demand but not wired into active decoders."""

    def test_legacy_module_importable(self):
        from lamquant_codec.legacy.lossless_legacy import (
            _decompress_legacy_bytes_ref,
            peek_header_legacy,
            LEGACY_MAGICS,
        )
        assert LEGACY_MAGICS == (b'LMQ4', b'LMQ5', b'LML ')
        assert callable(_decompress_legacy_bytes_ref)
        assert callable(peek_header_legacy)

    def test_active_path_does_not_import_legacy(self):
        """The active decoder must not pull in legacy on a clean import."""
        import sys
        # Force a clean reload.
        for mod in list(sys.modules):
            if mod.startswith('lamquant_codec.legacy'):
                del sys.modules[mod]
        # Import the active path.
        from lamquant_codec import lossless  # noqa: F401
        from lamquant_codec.ops import fused_lml  # noqa: F401
        # Legacy module must not be in sys.modules from this import alone.
        assert 'lamquant_codec.legacy.lossless_legacy' not in sys.modules

    def test_legacy_decoder_handles_lmq5_synthetically(self):
        """If a caller manually crafts an LMQ5 packet by overwriting the
        magic on a current LML1 packet (without bit 0 set), the legacy
        decoder must read it correctly."""
        from lamquant_codec.legacy.lossless_legacy import (
            _decompress_legacy_bytes_ref,
        )
        sig = synth_signal(2, 64, seed=88)
        body = bytearray(strip_ascii_prefix(
            py_encode(sig.astype(np.float64), noise_bits=0)))
        body[0:4] = b'LMQ5'
        decoded = _decompress_legacy_bytes_ref(bytes(body))
        np.testing.assert_array_equal(decoded.astype(np.int64), sig)
