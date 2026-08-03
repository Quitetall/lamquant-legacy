"""LML per-window codec roundtrip tests.

Tests _compress_bytes / _decompress_bytes at the packet level.
Every signal that goes in must come out bit-identical.
"""
import pytest
import numpy as np


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestRoundtrip:
    """Bit-exact roundtrip: compress → decompress = original."""

    def _check(self, sig):
        from lamquant_codec.lossless import _compress_bytes, _decompress_bytes
        sig = sig.astype(np.float64)
        sig_int = np.round(sig).astype(np.int64)
        compressed = _compress_bytes(sig)
        recovered = np.round(_decompress_bytes(compressed)).astype(np.int64)
        assert np.array_equal(sig_int, recovered), \
            f"max_diff={np.abs(sig_int - recovered).max()}"

    def test_normal_21ch(self, rng):
        self._check(rng.integers(-5000, 5000, (21, 2500)))

    def test_1ch(self, rng):
        self._check(rng.integers(-500, 500, (1, 2500)))

    def test_64ch(self, rng):
        self._check(rng.integers(-3000, 3000, (64, 1250)))

    def test_256ch(self, rng):
        self._check(rng.integers(-1000, 1000, (256, 625)))

    def test_zeros(self):
        self._check(np.zeros((21, 2500)))

    def test_ones(self):
        self._check(np.ones((21, 2500)))

    def test_neg_ones(self):
        self._check(-np.ones((21, 2500)))

    def test_dc_offset(self):
        self._check(np.full((21, 2500), 32767))

    def test_alternating(self):
        self._check(np.tile([1, -1], (21, 1250)))

    def test_impulse(self):
        sig = np.zeros((21, 2500), dtype=np.int64)
        sig[10, 1250] = 30000
        self._check(sig)

    def test_ramp(self):
        self._check(np.arange(21 * 2500, dtype=np.int64).reshape(21, 2500))

    def test_sawtooth(self):
        self._check(np.tile(np.arange(250), (21, 10)))

    def test_single_sample(self):
        self._check(np.array([[42]]))

    def test_two_samples(self):
        self._check(np.array([[1, -1]]))

    def test_three_samples(self):
        self._check(np.array([[100, -200, 300]]))

    def test_four_samples(self):
        self._check(np.array([[1, 2, 3, 4]]))

    def test_odd_length(self, rng):
        self._check(rng.integers(-5000, 5000, (21, 2501)))


class TestNoiseBits:
    """noise_bits parameter: strip LSBs, restore on decode."""

    def test_noise_bits_4(self):
        from lamquant_codec.lossless import _compress_bytes, _decompress_bytes
        rng = np.random.default_rng(42)
        sig = rng.integers(-50000, 50000, (21, 2500)).astype(np.float64)
        sig_int = np.round(sig).astype(np.int64)
        expected = (sig_int >> 4) << 4
        compressed = _compress_bytes(sig, noise_bits=4)
        recovered = np.round(_decompress_bytes(compressed)).astype(np.int64)
        assert np.array_equal(expected, recovered)

    @pytest.mark.parametrize("nb", [1, 2, 3, 5, 6, 7, 8])
    def test_noise_bits_range(self, nb):
        from lamquant_codec.lossless import _compress_bytes, _decompress_bytes
        rng = np.random.default_rng(nb)
        sig = rng.integers(-50000, 50000, (21, 2500)).astype(np.float64)
        sig_int = np.round(sig).astype(np.int64)
        expected = (sig_int >> nb) << nb
        compressed = _compress_bytes(sig, noise_bits=nb)
        recovered = np.round(_decompress_bytes(compressed)).astype(np.int64)
        assert np.array_equal(expected, recovered)


class TestStress:
    """Random signal stress tests."""

    def test_200_random(self, rng):
        from lamquant_codec.lossless import _compress_bytes, _decompress_bytes
        for i in range(200):
            n_ch = int(rng.integers(1, 65))
            T = int(rng.integers(4, 5001))
            amp = int(rng.integers(1, 32768))
            sig = rng.integers(-amp, max(amp, 2), (n_ch, T)).astype(np.float64)
            sig_int = np.round(sig).astype(np.int64)
            c = _compress_bytes(sig)
            r = np.round(_decompress_bytes(c)).astype(np.int64)
            assert np.array_equal(sig_int, r), \
                f"Fail at i={i}: {n_ch}ch x {T}"


class TestDeterminism:
    """Same input must produce same output."""

    def test_deterministic(self, rng):
        from lamquant_codec.lossless import _compress_bytes, _decompress_bytes
        sig = rng.integers(-5000, 5000, (21, 2500)).astype(np.float64)
        r1 = np.round(_decompress_bytes(_compress_bytes(sig))).astype(np.int64)
        r2 = np.round(_decompress_bytes(_compress_bytes(sig))).astype(np.int64)
        assert np.array_equal(r1, r2)
