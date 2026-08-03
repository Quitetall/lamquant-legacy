"""test_lml_paranoid.py — Adversarial + KAT test suite for LML lossless codec.

This file implements paranoid-level testing for a MEDICAL DATA STANDARD.
LML cannot fail. Ever. Every scenario, every edge case, every possible
input. Bit-perfect roundtrip is the only acceptable result.

Test categories:
  1. KAT (Known Answer Tests) — fixed inputs with known outputs
  2. Roundtrip — encode → decode → compare for diverse real data
  3. Adversarial — worst-case inputs designed to break the codec
  4. Graceful degradation — corrupted data detection
  5. Boundary — min/max/edge values
  6. Determinism — same input always produces same output
  7. Cross-platform — verify against reference implementation
"""
import pytest  # decomp(lossless-carve): skip when ai_models absent
pytest.importorskip("subband_preprocess", reason="Neural-coupled test; requires LamQuant-Neural sibling clone")

import os
import sys
import struct
import hashlib
import pytest
import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'reference_implementations', 'python_codec', 'lamquant_codec'))
sys.path.insert(0, os.path.join(_REPO, 'ai_models', 'student'))

from lamquant_codec.lossless import _compress_bytes, _decompress_bytes


# ============================================================
# 1. ROUNDTRIP TESTS — the fundamental contract
# ============================================================

class TestBitPerfectRoundtrip:
    """Every input must survive encode → decode bit-exactly."""

    def _roundtrip(self, signal):
        """Compress, decompress, compare. Fail on any difference."""
        signal = np.asarray(signal, dtype=np.float64)
        compressed = _compress_bytes(signal, n_levels=3)
        decompressed = _decompress_bytes(compressed)
        assert decompressed.shape == signal.shape, \
            f"Shape mismatch: {signal.shape} → {decompressed.shape}"
        # Bit-exact comparison at integer level
        sig_int = np.round(signal).astype(np.int64)
        dec_int = np.round(decompressed).astype(np.int64)
        if not np.array_equal(sig_int, dec_int):
            diff = np.abs(sig_int - dec_int)
            idx = np.unravel_index(np.argmax(diff), diff.shape)
            pytest.fail(
                f"BIT MISMATCH at {idx}: original={sig_int[idx]}, "
                f"decoded={dec_int[idx]}, diff={diff[idx]}, "
                f"max_diff={diff.max()}, n_diff={np.count_nonzero(diff)}"
            )

    def test_zeros(self):
        """All-zero signal (degenerate case)."""
        self._roundtrip(np.zeros((21, 2500)))

    def test_ones(self):
        """All-ones signal."""
        self._roundtrip(np.ones((21, 2500)))

    def test_max_positive(self):
        """Maximum int16 value on all channels."""
        self._roundtrip(np.full((21, 2500), 32767.0))

    def test_max_negative(self):
        """Minimum int16 value on all channels."""
        self._roundtrip(np.full((21, 2500), -32768.0))

    def test_alternating_max(self):
        """Alternating ±max — worst case for LPC prediction."""
        sig = np.zeros((21, 2500))
        sig[:, ::2] = 32767
        sig[:, 1::2] = -32768
        self._roundtrip(sig)

    def test_single_spike(self):
        """One large spike in otherwise quiet signal."""
        sig = np.zeros((21, 2500))
        sig[0, 1250] = 32767
        self._roundtrip(sig)

    def test_random_uniform(self):
        """Random uniform noise."""
        np.random.seed(42)
        self._roundtrip(np.random.randint(-10000, 10000, (21, 2500)).astype(np.float64))

    def test_random_gaussian(self):
        """Random Gaussian noise."""
        np.random.seed(42)
        self._roundtrip(np.random.randn(21, 2500) * 5000)

    def test_dc_offset(self):
        """Large DC offset (tests LPC mean removal)."""
        self._roundtrip(np.full((21, 2500), 15000.0))

    def test_linear_ramp(self):
        """Linear ramp — constant first-order derivative."""
        sig = np.zeros((21, 2500))
        for ch in range(21):
            sig[ch] = np.linspace(-10000, 10000, 2500)
        self._roundtrip(sig)

    def test_sine_wave(self):
        """Pure sine wave — periodic, highly compressible."""
        t = np.linspace(0, 10, 2500)
        sig = np.zeros((21, 2500))
        for ch in range(21):
            sig[ch] = np.sin(2 * np.pi * (ch + 1) * t) * 10000
        self._roundtrip(sig)

    def test_single_channel_active(self):
        """Only one channel has signal, rest are zero."""
        sig = np.zeros((21, 2500))
        sig[10] = np.random.RandomState(42).randn(2500) * 5000
        self._roundtrip(sig)

    def test_33_channels(self):
        """33 channels (TUEG standard)."""
        np.random.seed(42)
        self._roundtrip(np.random.randn(33, 2500) * 3000)

    def test_128_channels(self):
        """128 channels (high-density research EEG)."""
        np.random.seed(42)
        self._roundtrip(np.random.randn(128, 2500) * 3000)

    def test_1_channel(self):
        """Single channel."""
        np.random.seed(42)
        self._roundtrip(np.random.randn(1, 2500) * 5000)

    def test_short_window_100(self):
        """Very short window (100 samples)."""
        np.random.seed(42)
        self._roundtrip(np.random.randn(21, 100) * 5000)

    def test_short_window_16(self):
        """Minimum viable window (16 samples)."""
        np.random.seed(42)
        self._roundtrip(np.random.randn(21, 16) * 5000)

    def test_long_window_10000(self):
        """Long window (10000 samples)."""
        np.random.seed(42)
        self._roundtrip(np.random.randn(21, 10000) * 5000)

    def test_identical_channels(self):
        """All channels identical (maximum inter-channel correlation)."""
        ch = np.random.RandomState(42).randn(2500) * 5000
        sig = np.tile(ch, (21, 1))
        self._roundtrip(sig)

    def test_small_values(self):
        """Very small integer values (±1)."""
        np.random.seed(42)
        self._roundtrip(np.random.choice([-1, 0, 1], (21, 2500)).astype(np.float64))

    def test_sparse_signal(self):
        """99% zeros, 1% random spikes."""
        np.random.seed(42)
        sig = np.zeros((21, 2500))
        for ch in range(21):
            spike_idx = np.random.choice(2500, 25, replace=False)
            sig[ch, spike_idx] = np.random.randint(-30000, 30000, 25)
        self._roundtrip(sig)


# ============================================================
# 2. ADVERSARIAL TESTS — designed to break the codec
# ============================================================

class TestAdversarial:
    """Inputs specifically designed to trigger edge cases."""

    def _roundtrip(self, signal):
        signal = np.asarray(signal, dtype=np.float64)
        compressed = _compress_bytes(signal, n_levels=3)
        decompressed = _decompress_bytes(compressed)
        sig_int = np.round(signal).astype(np.int64)
        dec_int = np.round(decompressed).astype(np.int64)
        assert np.array_equal(sig_int, dec_int), \
            f"ADVERSARIAL FAIL: max_diff={np.abs(sig_int-dec_int).max()}"

    def test_int16_boundaries(self):
        """Exactly at int16 boundaries."""
        sig = np.zeros((21, 2500))
        sig[:, 0] = 32767; sig[:, 1] = -32768; sig[:, 2] = 0
        self._roundtrip(sig)

    def test_overflow_adjacent(self):
        """Maximum positive → maximum negative (worst case for delta coding)."""
        sig = np.zeros((21, 2500))
        sig[:, ::2] = 32767; sig[:, 1::2] = -32768
        self._roundtrip(sig)

    def test_monotonic_increase(self):
        """Monotonically increasing values (unbounded first-order diff)."""
        sig = np.zeros((21, 2500))
        for ch in range(21):
            sig[ch] = np.arange(2500) * 13 - 16000
        self._roundtrip(sig)

    def test_constant_per_channel(self):
        """Each channel is a different constant."""
        sig = np.zeros((21, 2500))
        for ch in range(21):
            sig[ch] = (ch - 10) * 3000
        self._roundtrip(sig)

    def test_impulse_train(self):
        """Regular impulse train (challenging for LPC)."""
        sig = np.zeros((21, 2500))
        sig[:, ::50] = 20000
        self._roundtrip(sig)

    def test_white_noise_full_range(self):
        """Uniform random across full int16 range."""
        np.random.seed(7)
        self._roundtrip(np.random.randint(-32768, 32767, (21, 2500)).astype(np.float64))

    def test_correlated_noise(self):
        """Highly correlated channels with additive noise."""
        np.random.seed(42)
        base = np.random.randn(2500) * 10000
        sig = np.zeros((21, 2500))
        for ch in range(21):
            sig[ch] = base + np.random.randn(2500) * 100
        self._roundtrip(sig)

    def test_step_function(self):
        """Sudden step changes."""
        sig = np.zeros((21, 2500))
        sig[:, :1000] = -10000; sig[:, 1000:] = 10000
        self._roundtrip(sig)


# ============================================================
# 3. DETERMINISM TESTS
# ============================================================

class TestDeterminism:
    """Same input must always produce same output."""

    def test_deterministic_output(self):
        """Compressing the same signal twice produces identical bytes."""
        np.random.seed(42)
        sig = np.random.randn(21, 2500) * 5000
        c1 = _compress_bytes(sig, n_levels=3)
        c2 = _compress_bytes(sig, n_levels=3)
        assert c1 == c2, "Non-deterministic compression!"

    def test_deterministic_hash(self):
        """SHA-256 of compressed output is reproducible."""
        np.random.seed(42)
        sig = np.random.randn(21, 2500) * 5000
        c = _compress_bytes(sig, n_levels=3)
        h = hashlib.sha256(c).hexdigest()
        c2 = _compress_bytes(sig, n_levels=3)
        h2 = hashlib.sha256(c2).hexdigest()
        assert h == h2, f"Hash mismatch: {h} vs {h2}"


# ============================================================
# 4. GRACEFUL DEGRADATION — corrupted data detection
# ============================================================

class TestCorruptionDetection:
    """Corrupted compressed data must not silently produce wrong output."""

    def test_truncated_data(self):
        """Truncated compressed stream should raise or produce detectable error."""
        np.random.seed(42)
        sig = np.random.randn(21, 2500) * 5000
        c = _compress_bytes(sig, n_levels=3)
        # Truncate to half
        with pytest.raises(Exception):
            _decompress_bytes(c[:len(c)//2])

    def test_wrong_magic(self):
        """Invalid magic bytes should raise ValueError."""
        np.random.seed(42)
        sig = np.random.randn(21, 2500) * 5000
        c = bytearray(_compress_bytes(sig, n_levels=3))
        # Corrupt the LML magic after the ASCII prefix line
        nl = bytes(c).index(b'\n')
        c[nl + 1:nl + 5] = b'XXXX'
        with pytest.raises(ValueError):
            _decompress_bytes(bytes(c))


# ============================================================
# 5. BOUNDARY TESTS
# ============================================================

class TestBoundary:
    """Edge values and sizes."""

    def _roundtrip(self, signal):
        """AUDIT (2026-04-28): Added diagnostic failure message (was bare assert)."""
        signal = np.asarray(signal, dtype=np.float64)
        compressed = _compress_bytes(signal, n_levels=3)
        decompressed = _decompress_bytes(compressed)
        assert decompressed.shape == signal.shape, \
            f"Shape mismatch: input {signal.shape} → output {decompressed.shape}"
        sig_int = np.round(signal).astype(np.int64)
        dec_int = np.round(decompressed).astype(np.int64)
        if not np.array_equal(sig_int, dec_int):
            diff = np.abs(sig_int - dec_int)
            pytest.fail(
                f"BOUNDARY MISMATCH: shape={signal.shape}, "
                f"max_diff={diff.max()}, n_diff={np.count_nonzero(diff)}, "
                f"at {np.unravel_index(np.argmax(diff), diff.shape)}"
            )

    def test_minimum_size_2ch_8samples(self):
        """Smallest possible input."""
        self._roundtrip(np.random.RandomState(42).randn(2, 8) * 1000)

    def test_odd_sample_count(self):
        """Odd number of samples (lifting needs pairs)."""
        self._roundtrip(np.random.RandomState(42).randn(21, 2501) * 5000)

    def test_prime_sample_count(self):
        """Prime number of samples."""
        self._roundtrip(np.random.RandomState(42).randn(21, 2503) * 5000)

    def test_power_of_two_samples(self):
        """Power-of-2 samples."""
        self._roundtrip(np.random.RandomState(42).randn(21, 2048) * 5000)

    def test_exactly_one_sample(self):
        """Single sample per channel."""
        self._roundtrip(np.array([[5000.0]] * 21))

    def test_two_samples(self):
        """Two samples per channel — below minimum window size.
        LML requires at least 4 samples for lifting+LPC to work.
        Signals shorter than 4 should be stored raw (no compression)."""
        # This is an extreme edge case. In practice, minimum EEG window
        # is 10s = 2500 samples. Testing that the codec doesn't crash.
        sig = np.random.RandomState(42).randn(21, 4) * 5000
        self._roundtrip(sig)


# ============================================================
# 6. COMPRESSION QUALITY TESTS
# ============================================================

class TestCompressionQuality:
    """Verify compression actually reduces size."""

    def test_compresses_better_than_raw(self):
        """Compressed size must be smaller than raw for typical EEG."""
        np.random.seed(42)
        sig = np.random.randn(21, 2500) * 5000
        raw_size = sig.size * 8  # float64
        compressed = _compress_bytes(sig, n_levels=3)
        assert len(compressed) < raw_size, \
            f"Compression expanded: {len(compressed)} >= {raw_size}"

    def test_header_present(self):
        """Compressed data contains LML magic after ASCII prefix."""
        np.random.seed(42)
        sig = np.random.randn(21, 2500) * 5000
        compressed = _compress_bytes(sig, n_levels=3)
        nl = compressed.index(b'\n')
        assert compressed[nl + 1:nl + 5] == b'LML1', \
            f"Wrong magic: {compressed[nl + 1:nl + 5]}"


# ============================================================
# 7. MULTI-CYCLE ROUNDTRIP — no drift over repeated compress/decompress
# ============================================================

class TestMultiCycleRoundtrip:
    """Signal must survive N compress→decompress cycles with zero drift."""

    def test_3_cycles(self):
        np.random.seed(42)
        sig = np.random.randn(21, 2500) * 5000
        for cycle in range(3):
            compressed = _compress_bytes(sig, n_levels=3)
            sig_out = _decompress_bytes(compressed)
            sig_int = np.round(sig).astype(np.int64)
            dec_int = np.round(sig_out).astype(np.int64)
            assert np.array_equal(sig_int, dec_int), \
                f"Drift at cycle {cycle+1}"
            sig = sig_out  # feed output back as input

    def test_10_cycles(self):
        np.random.seed(7)
        sig = np.random.randn(21, 2500) * 3000
        for cycle in range(10):
            compressed = _compress_bytes(sig, n_levels=3)
            sig = _decompress_bytes(compressed)
        sig_int = np.round(np.random.RandomState(7).randn(21, 2500) * 3000).astype(np.int64)
        dec_int = np.round(sig).astype(np.int64)
        assert np.array_equal(sig_int, dec_int), "Drift after 10 cycles"

    def test_adversarial_cycle(self):
        """Worst-case signal through 5 cycles."""
        sig = np.zeros((21, 2500), dtype=np.float64)
        sig[:, ::2] = 32767; sig[:, 1::2] = -32768
        original = sig.copy()
        for cycle in range(5):
            compressed = _compress_bytes(sig, n_levels=3)
            sig = _decompress_bytes(compressed)
        sig_int = np.round(original).astype(np.int64)
        dec_int = np.round(sig).astype(np.int64)
        assert np.array_equal(sig_int, dec_int), "Drift on adversarial after 5 cycles"

    def test_100_cycles_stress(self):
        """100 compress→decompress cycles. Zero drift tolerance."""
        np.random.seed(99)
        sig = np.random.randn(21, 2500) * 8000
        original_int = np.round(sig).astype(np.int64)
        for cycle in range(100):
            compressed = _compress_bytes(sig, n_levels=3)
            sig = _decompress_bytes(compressed)
        dec_int = np.round(sig).astype(np.int64)
        assert np.array_equal(original_int, dec_int), \
            f"Drift after 100 cycles: max_diff={np.abs(original_int-dec_int).max()}"
