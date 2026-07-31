"""Entropy coder tests: Golomb-Rice and rANS.

Verifies Rust (lamquant_core) produces byte-identical output to
Python reference implementations.
"""
import pytest
import numpy as np


class TestGolombRice:
    """Golomb-Rice encode/decode roundtrip."""

    def _roundtrip(self, data):
        from lamquant_codec.ops.golomb import encode_dense, decode_dense
        data = np.asarray(data, dtype=np.int64)
        encoded = bytes(encode_dense(data))
        decoded, consumed = decode_dense(encoded, 0)
        assert np.array_equal(data, decoded)

    def test_zeros(self):
        self._roundtrip(np.zeros(200, dtype=np.int64))

    def test_small(self):
        self._roundtrip(np.arange(-50, 50, dtype=np.int64))

    def test_large(self):
        rng = np.random.default_rng(42)
        self._roundtrip(rng.integers(-50000, 50000, 1250))

    def test_single(self):
        self._roundtrip(np.array([42], dtype=np.int64))

    def test_empty(self):
        self._roundtrip(np.array([], dtype=np.int64))

    def test_rust_byte_identical(self):
        """Rust output must be byte-identical to Python reference."""
        try:
            import lamquant_core
        except ImportError:
            pytest.skip("lamquant_core not installed")
        from lamquant_codec.ops.golomb import _encode_dense_pyref

        rng = np.random.default_rng(42)
        data = rng.integers(-3000, 3000, 625).astype(np.int64)
        py_bytes = bytes(_encode_dense_pyref(data))
        rs_bytes = bytes(lamquant_core.golomb_encode_dense(
            np.ascontiguousarray(data)))
        assert py_bytes == rs_bytes


class TestRANS:
    """rANS encode/decode roundtrip."""

    def _roundtrip(self, symbols, total_freq=4096):
        from lamquant_codec.ops.rans import compute_freq, encode_with_freq, decode
        symbols = np.asarray(symbols, dtype=np.int64)
        freq = compute_freq(symbols, total_freq=total_freq)
        M = int(freq.sum())
        encoded = encode_with_freq(symbols, freq, total_freq=M)
        decoded = decode(encoded, freq, len(symbols), total_freq=M)
        assert np.array_equal(symbols, decoded)

    def test_uniform(self):
        rng = np.random.default_rng(42)
        self._roundtrip(rng.integers(0, 8, 500))

    def test_binary(self):
        rng = np.random.default_rng(42)
        self._roundtrip(rng.integers(0, 2, 1000))

    def test_skewed(self):
        syms = np.concatenate([np.zeros(400), np.arange(1, 5).repeat(25)])
        self._roundtrip(syms.astype(np.int64))

    def test_wide_alphabet(self):
        rng = np.random.default_rng(42)
        self._roundtrip(rng.integers(0, 32, 300))
