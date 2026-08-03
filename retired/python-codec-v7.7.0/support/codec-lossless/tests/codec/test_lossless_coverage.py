"""Coverage tests for `lamquant_codec.lossless`.

Targets the still-uncovered branches:

  - compute_klt(): KLT matrix from multi-window training signals
  - compute_lifting_rotations(): orthogonal Givens decomposition with the
    >pi/3 recursion branch
  - _lifting_forward_one / _lifting_inverse_one (numpy variants)
  - _lifting_forward_one_pyref / _lifting_inverse_one_pyref (slow ref)
  - apply_lifting_klt_forward + inverse round-trip
  - _compress_bytes_ref guards: shape/channels/empty/double-stripped
  - _decompress_bytes_ref guards: truncated/version/legacy/magic/reserved/CRC
  - peek_header() happy path + every typed-error path
  - LosslessCodec instance: compress/decompress/compress_to_packet
  - Typed pipeline entries: compress(SubbandDecomposition) and decompress
  - lpc_order=8 schedule fallback (n_levels==1 short-signal adaptive path)

Uses np.random math fixtures — no synthetic EEG semantics.
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

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
from lamquant_codec.lossless import (
    LIFT_HALF,
    LIFT_PREC,
    LosslessCodec,
    _compress_bytes,
    _compress_bytes_ref,
    _decompress_bytes,
    _decompress_bytes_ref,
    _lifting_forward_one,
    _lifting_forward_one_pyref,
    _lifting_inverse_one,
    _lifting_inverse_one_pyref,
    apply_lifting_klt_forward,
    apply_lifting_klt_inverse,
    compress,
    compute_klt,
    compute_lifting_rotations,
    decompress,
    peek_header,
)
from lamquant_codec.codec_types import (
    CompressedPacket,
    SubbandDecomposition,
)

pytestmark = [pytest.mark.l3]


# ============================================================
# 1. compute_klt() — diagonalises sample covariance
# ============================================================


class TestComputeKLT:

    def test_shape_and_orthogonality(self):
        """KLT matrix is [C, C] and rows are orthonormal."""
        rng = np.random.default_rng(0)
        # Two windows of 5-channel × 250-sample data.
        sigs = [rng.standard_normal((5, 250)) * 50 for _ in range(2)]
        K = compute_klt(sigs)
        assert K.shape == (5, 5)
        # Row-orthonormal: K @ K.T ~ I.
        I = K @ K.T
        np.testing.assert_allclose(I, np.eye(5), atol=1e-9)

    def test_dtype_is_float64(self):
        rng = np.random.default_rng(1)
        sigs = [rng.standard_normal((3, 100))]
        K = compute_klt(sigs)
        assert K.dtype == np.float64


# ============================================================
# 2. compute_lifting_rotations() — Givens factorization
# ============================================================


class TestLiftingRotations:

    def test_emits_finite_list_on_random_orthonormal(self):
        rng = np.random.default_rng(2)
        # Random orthonormal 4x4 from QR.
        A = rng.standard_normal((4, 4))
        Q, _ = np.linalg.qr(A)
        rots = compute_lifting_rotations(Q.astype(np.float64))
        assert isinstance(rots, list)
        # Each tuple is (i, j, a_q, b_q, mode=0).
        for tup in rots:
            assert len(tup) == 5
            i, j, a_q, b_q, mode = tup
            assert 0 <= i < 4 and 0 <= j < 4
            assert isinstance(a_q, int) and isinstance(b_q, int)
            assert mode == 0

    def test_identity_emits_no_rotations(self):
        """KLT = I → no lifting steps needed."""
        I = np.eye(3, dtype=np.float64)
        rots = compute_lifting_rotations(I)
        assert rots == []

    def test_large_rotation_recurses(self):
        """A rotation with abs(theta) > pi/3 must trigger the recursion branch."""
        # Build a 2x2 rotation by ~ 2*pi/5 (> pi/3).
        theta = 2 * np.pi / 5
        c, s = np.cos(theta), np.sin(theta)
        K = np.array([[c, -s], [s, c]], dtype=np.float64)
        rots = compute_lifting_rotations(K)
        # Recursion halves the angle until small enough — emits 2+ steps.
        assert len(rots) >= 2


# ============================================================
# 3. Per-rotation primitives (vectorised + reference)
# ============================================================


class TestLiftingPrimitives:

    def test_numpy_forward_inverse_roundtrip_int(self):
        rng = np.random.default_rng(3)
        out = rng.integers(-1000, 1000, size=(2, 64)).astype(np.int64)
        before = out.copy()
        a_q, b_q = 12345, -6789
        _lifting_forward_one(out, 0, 1, a_q, b_q, LIFT_PREC)
        _lifting_inverse_one(out, 0, 1, a_q, b_q, LIFT_PREC)
        np.testing.assert_array_equal(out, before)

    def test_pyref_matches_vectorised(self):
        """Slow Python loop must match the numpy implementation bit-for-bit."""
        rng = np.random.default_rng(4)
        out_np = rng.integers(-1000, 1000, size=(2, 16)).astype(np.int64)
        out_py = out_np.copy()
        # The pyref helper expects nested int lists (uses [t] indexing).
        list_py = [out_py[0].tolist(), out_py[1].tolist()]
        a_q, b_q = 4321, -5678
        _lifting_forward_one(out_np, 0, 1, a_q, b_q, LIFT_PREC)
        _lifting_forward_one_pyref(list_py, 0, 1, a_q, b_q, LIFT_PREC)
        # The pyref accumulates in Python ints; compare element-by-element.
        for c in range(2):
            for t in range(16):
                assert list_py[c][t] == int(out_np[c, t]), \
                    f"mismatch at ch={c} t={t}"

    def test_pyref_inverse_undoes_pyref_forward(self):
        rng = np.random.default_rng(5)
        sig = rng.integers(-200, 200, size=(2, 12)).astype(np.int64)
        before = [sig[0].tolist(), sig[1].tolist()]
        rolling = [sig[0].tolist(), sig[1].tolist()]
        a_q, b_q = 99, -77
        _lifting_forward_one_pyref(rolling, 0, 1, a_q, b_q, LIFT_PREC)
        _lifting_inverse_one_pyref(rolling, 0, 1, a_q, b_q, LIFT_PREC)
        assert rolling == before


class TestKLTApply:

    def test_forward_inverse_roundtrip_with_identity_rotations(self):
        rng = np.random.default_rng(6)
        sig = rng.integers(-5000, 5000, size=(3, 64)).astype(np.int64)
        rots = []  # empty → forward/inverse are identity
        out = apply_lifting_klt_forward(sig, rots)
        np.testing.assert_array_equal(out, sig)
        back = apply_lifting_klt_inverse(out, rots)
        np.testing.assert_array_equal(back, sig)

    def test_forward_inverse_roundtrip_on_klt(self):
        rng = np.random.default_rng(7)
        # Build a real KLT from random data.
        sigs = [rng.standard_normal((3, 250)) * 100 for _ in range(2)]
        K = compute_klt(sigs)
        rots = compute_lifting_rotations(K)
        sig = rng.integers(-5000, 5000, size=(3, 64)).astype(np.int64)
        fwd = apply_lifting_klt_forward(sig, rots)
        back = apply_lifting_klt_inverse(fwd, rots)
        np.testing.assert_array_equal(back, sig)


# ============================================================
# 4. _compress_bytes_ref — every guard
# ============================================================


class TestCompressBytesGuards:

    def test_rejects_1d_signal(self):
        with pytest.raises(LmlSignalShapeError):
            _compress_bytes_ref(np.zeros(64, dtype=np.float64))

    def test_rejects_3d_signal(self):
        with pytest.raises(LmlSignalShapeError):
            _compress_bytes_ref(np.zeros((2, 4, 64), dtype=np.float64))

    def test_rejects_too_many_channels(self):
        # 1025 > 1024 channel ceiling.
        sig = np.zeros((1025, 16), dtype=np.float64)
        with pytest.raises(LmlChannelCountError):
            _compress_bytes_ref(sig)

    def test_rejects_zero_channels(self):
        sig = np.zeros((0, 16), dtype=np.float64)
        with pytest.raises(LmlEmptySignalError):
            _compress_bytes_ref(sig)

    def test_rejects_zero_samples(self):
        sig = np.zeros((2, 0), dtype=np.float64)
        with pytest.raises(LmlEmptySignalError):
            _compress_bytes_ref(sig)

    def test_rejects_double_strip(self):
        """noise_bits>0 with already-zero bottom bits → typed error."""
        rng = np.random.default_rng(8)
        sig = rng.integers(-256, 256, size=(2, 64)).astype(np.float64)
        # Strip the bottom 4 bits ourselves — now the bottom 4 are zero.
        sig = (sig.astype(np.int64) >> 4 << 4).astype(np.float64)
        with pytest.raises(LmlNoiseStrippedError):
            _compress_bytes_ref(sig, noise_bits=4)

    def test_short_signal_adapts_nlevels(self):
        """T < 4*2^n_levels reduces n_levels until it fits."""
        # T=8 → forces n_levels down to 1 (4*2 = 8 fits).
        rng = np.random.default_rng(9)
        sig = rng.integers(-100, 100, size=(2, 8)).astype(np.float64)
        # Should NOT crash; produce a valid LML1 packet.
        packet = _compress_bytes_ref(sig, n_levels=3)
        # Decode it back.
        recon = _decompress_bytes_ref(packet)
        np.testing.assert_array_equal(recon.astype(np.int64), sig.astype(np.int64))


# ============================================================
# 5. _decompress_bytes_ref — every typed guard
# ============================================================


def _make_ref_packet(seed: int = 12, n_ch: int = 2, T: int = 64) -> bytes:
    """Build a tiny LML1 packet via the encoder for use in mutation tests."""
    rng = np.random.default_rng(seed)
    sig = rng.integers(-1000, 1000, size=(n_ch, T)).astype(np.float64)
    return _compress_bytes_ref(sig, n_levels=2)


class TestDecompressBytesGuards:

    def test_truncated_data_too_short(self):
        with pytest.raises(LmlTruncatedError):
            _decompress_bytes_ref(b"AB")

    def test_unknown_magic_after_prefix(self):
        with pytest.raises(LmlMagicError):
            _decompress_bytes_ref(b"BOGUS_MAGIC_HEADER_AT_LEAST_22_BYTES_LONG__")

    def test_legacy_magic_routes_to_typed_error(self):
        # LMQ4 legacy magic at offset 0 → LmlLegacyMagicError.
        body = b"LMQ4" + b"\x00" * 60
        with pytest.raises(LmlLegacyMagicError):
            _decompress_bytes_ref(body)

    def test_future_lml_version_typed(self):
        # LML9 magic → LmlVersionError ("newer than reader").
        body = b"LML9" + b"\x00" * 60
        with pytest.raises(LmlVersionError):
            _decompress_bytes_ref(body)

    def test_truncated_after_magic(self):
        # Valid magic but only 4 bytes total.
        body = b"LML1"
        with pytest.raises(LmlTruncatedError):
            _decompress_bytes_ref(body)

    def test_reserved_flag_bits_typed(self):
        pkt = _make_ref_packet()
        # Find binary header (after ASCII prefix).
        nl = pkt.find(b"\n")
        bin_start = nl + 1 if 0 < nl < 128 else 0
        # Flag byte is at offset bin_start + 9 (magic 4 + n_ch 2 + T 2 + nlev 1).
        flag_off = bin_start + 9
        mutated = bytearray(pkt)
        mutated[flag_off] = mutated[flag_off] | 0x01  # set reserved bit
        with pytest.raises(LmlReservedBitsSetError):
            _decompress_bytes_ref(bytes(mutated))

    def test_crc_mismatch_typed(self):
        pkt = _make_ref_packet()
        # Tamper a byte well inside the payload — CRC must catch it.
        mutated = bytearray(pkt)
        # Last 4 bytes are part of payload; flip a high-order bit.
        mutated[-3] ^= 0x40
        with pytest.raises(LmlCrcError):
            _decompress_bytes_ref(bytes(mutated))

    def test_truncated_payload_detected(self):
        pkt = _make_ref_packet()
        # Chop the last 8 bytes — declared payload length now exceeds file.
        with pytest.raises((LmlTruncatedError, LmlCrcError)):
            _decompress_bytes_ref(pkt[:-8])


# ============================================================
# 6. peek_header — happy path + each guard
# ============================================================


class TestPeekHeader:

    def test_happy_path_returns_header_dataclass(self):
        pkt = _make_ref_packet(n_ch=3, T=64)
        hdr = peek_header(pkt)
        assert hdr.version == "LML1"
        assert hdr.n_channels == 3
        assert hdr.n_samples == 64
        assert hdr.lossless is True  # noise_bits=0
        assert hdr.noise_bits == 0
        # total_bytes equals header + lpc_meta + payload.
        assert hdr.total_bytes == 22 + hdr.lpc_meta_bytes + hdr.payload_bytes

    def test_rejects_too_short(self):
        with pytest.raises(LmlTruncatedError):
            peek_header(b"AB")

    def test_rejects_invalid_magic(self):
        with pytest.raises(LmlMagicError):
            peek_header(b"NOPEHEADER" + b"\x00" * 30)

    def test_rejects_legacy_magic(self):
        with pytest.raises(LmlLegacyMagicError):
            peek_header(b"LMQ4" + b"\x00" * 30)

    def test_rejects_short_after_magic(self):
        # Valid magic but missing rest of header.
        with pytest.raises(LmlTruncatedError):
            peek_header(b"LML1" + b"\x00" * 5)

    def test_rejects_reserved_bits_set(self):
        pkt = _make_ref_packet()
        nl = pkt.find(b"\n")
        bin_start = nl + 1 if 0 < nl < 128 else 0
        flag_off = bin_start + 9
        mutated = bytearray(pkt)
        mutated[flag_off] = mutated[flag_off] | 0x02
        with pytest.raises(LmlReservedBitsSetError):
            peek_header(bytes(mutated))


# ============================================================
# 7. LosslessCodec class — stateful wrapper
# ============================================================


class TestLosslessCodec:

    def test_roundtrip_no_klt(self):
        rng = np.random.default_rng(20)
        sig = rng.integers(-1000, 1000, size=(2, 128)).astype(np.float64)
        codec = LosslessCodec(klt_matrix=None, lpc_order=8, n_levels=2)
        packet = codec.compress(sig)
        recon = codec.decompress(packet)
        np.testing.assert_array_equal(recon.astype(np.int64), sig.astype(np.int64))

    def test_roundtrip_with_klt(self):
        rng = np.random.default_rng(21)
        sigs = [rng.standard_normal((3, 250)) * 100 for _ in range(2)]
        K = compute_klt(sigs)
        codec = LosslessCodec(klt_matrix=K, lpc_order=8, n_levels=2)
        # The codec.lifting_rots field is auto-populated from K.
        assert codec.lifting_rots is not None
        sig = rng.integers(-2000, 2000, size=(3, 128)).astype(np.float64)
        packet = codec.compress(sig)
        recon = codec.decompress(packet)
        np.testing.assert_array_equal(recon.astype(np.int64), sig.astype(np.int64))

    def test_compress_to_packet_wraps_eegpacket(self):
        rng = np.random.default_rng(22)
        sig = rng.integers(-200, 200, size=(2, 64)).astype(np.float64)
        codec = LosslessCodec(n_levels=2)
        eeg_packet = codec.compress_to_packet(sig)
        # Returned object has the signal + lossless mode + metadata.
        assert eeg_packet.mode == 'lossless'
        assert eeg_packet.signal.shape == sig.shape
        assert eeg_packet.metadata['n_levels'] == 2

    def test_lift_constants_exposed(self):
        assert LosslessCodec.LIFT_PREC == LIFT_PREC
        assert LosslessCodec.LIFT_HALF == LIFT_HALF


# ============================================================
# 8. Typed pipeline entry points: compress / decompress
# ============================================================


class TestTypedPipelineEntries:

    def _make_subband(self, sig: np.ndarray) -> SubbandDecomposition:
        # We only need source_signal populated; the other fields aren't
        # used by the typed compress() entry.
        empty = np.zeros((sig.shape[0], 1), dtype=np.float64)
        empty_lpc = np.zeros((sig.shape[0], 1), dtype=np.float64)
        return SubbandDecomposition(
            l3_approx=empty, l3_detail=empty,
            l2_detail=empty, l1_detail=empty,
            lpc_coeffs=empty_lpc, lpc_order=8,
            source_signal=sig.astype(np.float64),
        )

    def test_compress_requires_source_signal(self):
        empty = np.zeros((2, 1), dtype=np.float64)
        empty_lpc = np.zeros((2, 1), dtype=np.float64)
        sub = SubbandDecomposition(
            l3_approx=empty, l3_detail=empty,
            l2_detail=empty, l1_detail=empty,
            lpc_coeffs=empty_lpc, lpc_order=8,
            source_signal=None,
        )
        with pytest.raises(ValueError, match="source_signal"):
            compress(sub)

    def test_typed_compress_decompress_roundtrip(self):
        rng = np.random.default_rng(30)
        sig = rng.integers(-500, 500, size=(2, 128)).astype(np.float64)
        sub = self._make_subband(sig)
        packet = compress(sub, n_levels=2)
        assert isinstance(packet, CompressedPacket)
        assert packet.mode == 'lossless'
        recon = decompress(packet)
        np.testing.assert_array_equal(recon.astype(np.int64), sig.astype(np.int64))

    def test_decompress_rejects_non_lossless_packet(self):
        # Packet wired with mode='neural' must be rejected.
        fake = CompressedPacket(data=b"LML1", mode='neural')
        with pytest.raises(ValueError, match="mode"):
            decompress(fake)


# ============================================================
# 9. Noise-bits and KLT-on-decode coverage
# ============================================================


class TestNoiseBitsAndKLT:

    def test_noise_bits_round_trips_through_ref_path(self):
        rng = np.random.default_rng(40)
        # Signal where low bits are scrambled — noise_bits safety check
        # requires the bottom N bits to NOT all be zero.
        sig = rng.integers(-1000, 1000, size=(2, 64)).astype(np.float64)
        sig[0, 0] = 7  # ensure low bit non-zero
        packet = _compress_bytes_ref(sig, n_levels=2, noise_bits=2)
        recon = _decompress_bytes_ref(packet).astype(np.int64)
        # Expected = (sig >> 2) << 2
        expected = (sig.astype(np.int64) >> 2) << 2
        np.testing.assert_array_equal(recon, expected)

    def test_klt_roundtrip_via_decompress_ref(self):
        rng = np.random.default_rng(41)
        sigs = [rng.standard_normal((3, 250)) * 100 for _ in range(2)]
        K = compute_klt(sigs)
        rots = compute_lifting_rotations(K)
        sig = rng.integers(-1000, 1000, size=(3, 128)).astype(np.float64)
        packet = _compress_bytes_ref(sig, lifting_rots=rots, n_levels=2)
        recon = _decompress_bytes_ref(packet, lifting_rots=rots)
        # Bit-exact lossless even with KLT.
        np.testing.assert_array_equal(recon.astype(np.int64),
                                      sig.astype(np.int64))


# ============================================================
# 10. _compress_bytes / _decompress_bytes wrapper paths
# ============================================================


class TestWrapperPaths:

    def test_compress_bytes_wrapper_round_trip(self):
        """Public _compress_bytes routes through fused (if available) or ref."""
        rng = np.random.default_rng(50)
        sig = rng.integers(-500, 500, size=(2, 256)).astype(np.float64)
        packet = _compress_bytes(sig, n_levels=2)
        recon = _decompress_bytes(packet)
        np.testing.assert_array_equal(recon.astype(np.int64),
                                      sig.astype(np.int64))

    def test_compress_bytes_with_lifting_rots_uses_ref(self):
        """When lifting_rots is non-None, fused path is skipped — ref runs."""
        rng = np.random.default_rng(51)
        sigs = [rng.standard_normal((3, 250)) * 100 for _ in range(2)]
        K = compute_klt(sigs)
        rots = compute_lifting_rotations(K)
        sig = rng.integers(-500, 500, size=(3, 128)).astype(np.float64)
        # lifting_rots != None → fused path bypassed, ref handles it.
        packet = _compress_bytes(sig, lifting_rots=rots, n_levels=2)
        recon = _decompress_bytes(packet, lifting_rots=rots)
        np.testing.assert_array_equal(recon.astype(np.int64),
                                      sig.astype(np.int64))
