"""Coverage tests for ``lamquant_codec.ops.golomb``.

Pins shape + invariant contracts, not exact bitstream bytes:
  - zigzag_encode / zigzag_decode round-trip (signed ints)
  - BitWriter / BitReader symmetric: write_bits/read_bits, write_unary/read_unary
  - encode_dense / decode_dense round-trip on math fixtures
  - encode_detail / decode_detail round-trip on sparse arrays
  - compute_adaptive_k returns documented int >= 0
  - Edge cases: empty input, single value, k=0
  - encode / decode aliases route to dense path

Math fixtures (np.random) — NOT synthetic EEG data.
"""
from __future__ import annotations

import numpy as np
import pytest

from lamquant_codec.ops.golomb import (
    BitReader,
    BitWriter,
    _decode_dense_pyref,
    _encode_dense_pyref,
    compute_adaptive_k,
    decode,
    decode_dense,
    decode_detail,
    encode,
    encode_dense,
    encode_detail,
    zigzag_decode,
    zigzag_encode,
)


def _math_int_array(n: int = 256, scale: int = 10, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return (rng.randn(n) * scale).astype(np.int64)


def _math_sparse_array(n: int = 256, density: float = 0.1,
                       scale: int = 10, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    mask = rng.rand(n) < density
    values = (rng.randn(n) * scale).astype(np.int64) * mask.astype(np.int64)
    return values


# ============================================================
# Zigzag mapping (signed ↔ unsigned)
# ============================================================


class TestZigzag:
    @pytest.mark.parametrize("v", [0, 1, -1, 2, -2, 127, -128, 65535, -65536])
    def test_roundtrip(self, v: int) -> None:
        assert zigzag_decode(zigzag_encode(v)) == v

    def test_encoded_values_unsigned(self) -> None:
        for v in [-100, -1, 0, 1, 100]:
            zz = zigzag_encode(v)
            assert zz >= 0

    def test_known_small_values(self) -> None:
        # Contract: 0→0, -1→1, 1→2, -2→3, 2→4 ... (spec from the docstring)
        assert zigzag_encode(0) == 0
        assert zigzag_encode(-1) == 1
        assert zigzag_encode(1) == 2
        assert zigzag_encode(-2) == 3
        assert zigzag_encode(2) == 4


# ============================================================
# BitWriter / BitReader low-level primitives
# ============================================================


class TestBitWriter:
    def test_write_bits_roundtrip_through_reader(self) -> None:
        w = BitWriter()
        values_and_widths = [(0b101, 3), (0b1111, 4), (0xAB, 8), (0, 5)]
        for v, n in values_and_widths:
            w.write_bits(v, n)
        data = w.to_bytes()
        assert isinstance(data, bytes)
        r = BitReader(data)
        for v, n in values_and_widths:
            assert r.read_bits(n) == v

    def test_write_unary_then_read_unary(self) -> None:
        w = BitWriter()
        unary_values = [0, 1, 5, 56, 113, 200]
        for v in unary_values:
            w.write_unary(v)
        r = BitReader(w.to_bytes())
        for v in unary_values:
            assert r.read_unary() == v

    def test_write_zero_width_noop(self) -> None:
        w = BitWriter()
        w.write_bits(0xFF, 0)  # no-op
        w.write_bits(0b1, 1)
        data = w.to_bytes()
        assert BitReader(data).read_bits(1) == 1

    def test_bit_count_tracks_total(self) -> None:
        w = BitWriter()
        w.write_bits(0, 3)
        w.write_bits(0, 5)
        assert w.bit_count() == 8

    def test_golomb_rice_roundtrip(self) -> None:
        w = BitWriter()
        values_k = [(0, 4), (1, 0), (5, 2), (100, 5), (1000, 3)]
        for v, k in values_k:
            w.write_golomb_rice(v, k)
        r = BitReader(w.to_bytes())
        for v, k in values_k:
            assert r.read_golomb_rice(k) == v


class TestBitReader:
    def test_read_bit_drains_buffer(self) -> None:
        w = BitWriter()
        w.write_bits(0b1010, 4)
        r = BitReader(w.to_bytes())
        assert r.read_bit() == 1
        assert r.read_bit() == 0
        assert r.read_bit() == 1
        assert r.read_bit() == 0

    def test_read_bits_zero_returns_zero(self) -> None:
        r = BitReader(b'\x00')
        assert r.read_bits(0) == 0

    def test_empty_stream_returns_zero_bits(self) -> None:
        r = BitReader(b'')
        assert r.read_bit() == 0

    def test_accepts_bytearray_and_memoryview(self) -> None:
        r1 = BitReader(bytearray(b'\xAB'))
        r2 = BitReader(memoryview(b'\xAB'))
        assert r1.read_bits(8) == 0xAB
        assert r2.read_bits(8) == 0xAB


# ============================================================
# compute_adaptive_k
# ============================================================


class TestComputeAdaptiveK:
    def test_returns_nonneg_int(self) -> None:
        k = compute_adaptive_k(_math_int_array(256, scale=20))
        assert isinstance(k, int)
        assert k >= 0

    def test_zero_array_yields_zero_k(self) -> None:
        k = compute_adaptive_k(np.zeros(64, dtype=np.int64))
        assert k == 0

    def test_small_values_yield_zero_k(self) -> None:
        # Values < 1 in mean abs => k = 0 by spec.
        assert compute_adaptive_k(np.array([0, 0, 0, 1])) == 0

    def test_large_values_yield_larger_k(self) -> None:
        # mean abs >> 1 -> k > 0
        large = np.full(64, 1024, dtype=np.int64)
        assert compute_adaptive_k(large) > 0


# ============================================================
# Dense encode/decode
# ============================================================


class TestDenseRoundtrip:
    @pytest.mark.parametrize("n,scale,seed",
                             [(64, 5, 0), (256, 10, 1), (1024, 50, 2)])
    def test_roundtrip(self, n: int, scale: int, seed: int) -> None:
        arr = _math_int_array(n, scale=scale, seed=seed)
        encoded = encode_dense(arr)
        assert isinstance(encoded, (bytes, bytearray))
        decoded, consumed = decode_dense(encoded)
        assert decoded.dtype == np.int64
        assert decoded.shape == arr.shape
        np.testing.assert_array_equal(decoded, arr)
        assert consumed == len(encoded)

    def test_empty_input_yields_3byte_header(self) -> None:
        encoded = encode_dense(np.array([], dtype=np.int64))
        # Header: <BH = 3 bytes (k=0, n_total=0)
        assert len(encoded) == 3
        decoded, consumed = decode_dense(encoded)
        assert decoded.shape == (0,)
        assert consumed == 3

    def test_single_zero_value(self) -> None:
        arr = np.array([0], dtype=np.int64)
        encoded = encode_dense(arr)
        decoded, _ = decode_dense(encoded)
        np.testing.assert_array_equal(decoded, arr)

    def test_negative_values_roundtrip(self) -> None:
        arr = np.array([-100, -1, 0, 1, 100], dtype=np.int64)
        decoded, _ = decode_dense(encode_dense(arr))
        np.testing.assert_array_equal(decoded, arr)

    def test_offset_decode(self) -> None:
        """decode_dense honours offset for embedded payloads."""
        prefix = b'\xDE\xAD\xBE\xEF'
        arr = _math_int_array(64, scale=4, seed=3)
        encoded = bytes(encode_dense(arr))
        blob = prefix + encoded
        decoded, consumed = decode_dense(blob, offset=len(prefix))
        np.testing.assert_array_equal(decoded, arr)
        assert consumed == len(encoded)

    def test_truncated_data_raises(self) -> None:
        with pytest.raises(ValueError):
            decode_dense(b'\x00\x00')  # 2 bytes < 3 header

    def test_alias_encode_decode(self) -> None:
        """`encode` / `decode` are aliases for the dense path."""
        arr = _math_int_array(128, scale=8, seed=4)
        out = encode(arr)
        decoded, _ = decode(out)
        np.testing.assert_array_equal(decoded, arr)


class TestDensePyref:
    """The pyref encoder + decoder is the SPEC; round-trip them."""

    def test_pyref_roundtrip(self) -> None:
        arr = _math_int_array(64, scale=5, seed=10)
        encoded = _encode_dense_pyref(arr.astype(np.float64))
        decoded, _ = _decode_dense_pyref(bytes(encoded))
        np.testing.assert_array_equal(decoded, arr)

    def test_pyref_empty(self) -> None:
        encoded = _encode_dense_pyref(np.array([]))
        decoded, _ = _decode_dense_pyref(bytes(encoded))
        assert decoded.shape == (0,)


# ============================================================
# Detail (sparse) encode/decode
# ============================================================


class TestDetailRoundtrip:
    def test_sparse_roundtrip(self) -> None:
        arr = _math_sparse_array(n=256, density=0.1, scale=10, seed=0)
        encoded = encode_detail(arr)
        assert isinstance(encoded, bytearray)
        decoded, consumed = decode_detail(bytes(encoded))
        assert decoded.shape == (256,)
        # Detail uses int32 round + float32; check integer values match.
        np.testing.assert_array_equal(decoded.astype(np.int64), arr)
        assert consumed > 0

    def test_all_zero_input(self) -> None:
        arr = np.zeros(64, dtype=np.int64)
        encoded = encode_detail(arr)
        # n_nz==0 -> 6-byte header only.
        assert len(encoded) == 6
        decoded, consumed = decode_detail(bytes(encoded))
        assert decoded.shape == (64,)
        np.testing.assert_array_equal(decoded.astype(np.int64), arr)
        assert consumed == 6

    def test_empty_input(self) -> None:
        encoded = encode_detail(np.array([], dtype=np.int64))
        # n_total==0 -> 6-byte header.
        assert len(encoded) == 6
        decoded, consumed = decode_detail(bytes(encoded))
        assert decoded.shape == (0,)

    def test_truncated_data_returns_empty(self) -> None:
        decoded, consumed = decode_detail(b'\x00')
        assert decoded.shape == (0,)
        assert consumed == 0
