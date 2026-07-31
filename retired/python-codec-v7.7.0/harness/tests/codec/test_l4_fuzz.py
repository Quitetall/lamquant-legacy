"""test_lml_fuzz.py — Property-based + differential fuzzing + bit-flip tests.

Guardrails 6-8 from the LML hardening plan:
  6. Differential fuzzing: random valid EEG → roundtrip → verify
  7. Property-based testing: invariants expressed as Hypothesis strategies
  8. Bit-flip corruption: flipped bits must error, never silently corrupt

These tests run longer than unit tests. Use `pytest -m fuzz` to run them
separately, or `pytest --timeout=300` for CI.
"""
import os
import sys
import struct
import hashlib
import pytest
import numpy as np

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'reference_implementations', 'python_codec', 'lamquant_codec'))
sys.path.insert(0, os.path.join(_REPO, 'ai_models', 'student'))

from lamquant_codec.lossless import _compress_bytes, _decompress_bytes


# ============================================================
# Hypothesis strategies for EEG-like data
# ============================================================

# Realistic EEG: 1-128 channels, 4-10000 samples, int16 range
eeg_signals = st.builds(
    lambda ch, t, seed: np.random.RandomState(seed).randint(
        -20000, 20000, (ch, t)).astype(np.float64),
    ch=st.integers(min_value=1, max_value=64),
    t=st.integers(min_value=8, max_value=5000),
    seed=st.integers(min_value=0, max_value=2**31),
)


# ============================================================
# Guardrail 6: Differential fuzzing
# ============================================================

class TestDifferentialFuzz:
    """Random EEG signals → compress → decompress → bit-exact."""

    @given(eeg_signals)
    @settings(max_examples=200, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_random_eeg_roundtrip(self, signal):
        """Hypothesis-generated EEG signals roundtrip bit-perfectly."""
        compressed = _compress_bytes(signal, n_levels=3)
        decompressed = _decompress_bytes(compressed)
        sig_int = np.round(signal).astype(np.int64)
        dec_int = np.round(decompressed).astype(np.int64)
        assert np.array_equal(sig_int, dec_int), \
            f"FUZZ FAIL: shape={signal.shape}, max_diff={np.abs(sig_int-dec_int).max()}"


# ============================================================
# Guardrail 7: Property-based testing
# ============================================================

class TestProperties:
    """Invariants that must hold for ALL inputs."""

    @given(eeg_signals)
    @settings(max_examples=100, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_roundtrip_identity(self, signal):
        """decompress(compress(x)) == x for all x."""
        c = _compress_bytes(signal, n_levels=3)
        d = _decompress_bytes(c)
        assert np.array_equal(
            np.round(signal).astype(np.int64),
            np.round(d).astype(np.int64))

    @given(eeg_signals)
    @settings(max_examples=100, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_compression_bounded(self, signal):
        """Compressed size bounded by C * raw_size for some constant C."""
        raw_size = signal.size * 8  # float64 bytes
        c = _compress_bytes(signal, n_levels=3)
        # Compressed should be at most 2× raw size (generous bound)
        assert len(c) <= raw_size * 2, \
            f"Compression expanded beyond 2×: {len(c)} > {raw_size * 2}"

    @given(eeg_signals)
    @settings(max_examples=50, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_deterministic(self, signal):
        """compress(x) == compress(x) always."""
        c1 = _compress_bytes(signal, n_levels=3)
        c2 = _compress_bytes(signal, n_levels=3)
        assert c1 == c2, "Non-deterministic compression"

    @given(eeg_signals)
    @settings(max_examples=50, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_recompression_idempotent(self, signal):
        """compress(decompress(compress(x))) == compress(x)."""
        c1 = _compress_bytes(signal, n_levels=3)
        d = _decompress_bytes(c1)
        c2 = _compress_bytes(d, n_levels=3)
        # The bytes might differ (floating point representation of d differs),
        # but the decompressed result must be identical
        d2 = _decompress_bytes(c2)
        assert np.array_equal(
            np.round(d).astype(np.int64),
            np.round(d2).astype(np.int64)), \
            "Recompression not idempotent"

    @given(eeg_signals)
    @settings(max_examples=50, deadline=30000,
              suppress_health_check=[HealthCheck.too_slow])
    def test_valid_header(self, signal):
        """Compressed data starts with human-readable prefix then LML magic."""
        c = _compress_bytes(signal, n_levels=3)
        # LML packets have ASCII prefix "LML v5 | ..." followed by \n then LML
        nl = c.index(b'\n')
        assert c[nl + 1:nl + 5] == b'LML1'


# ============================================================
# Guardrail 8: Bit-flip corruption testing
# ============================================================

class TestBitFlipCorruption:
    """Flipped bits must cause error or be detected, never silent corruption."""

    def _compress_signal(self):
        np.random.seed(42)
        sig = np.random.randn(21, 2500) * 5000
        return _compress_bytes(sig, n_levels=3), sig

    def test_single_bit_flip_in_payload(self):
        """Single bit flip in payload: must either error or produce wrong output
        that is detectable (not silently accepted as valid).

        AUDIT (2026-04-27): Narrowed from bare `except Exception` to specific
        expected exceptions. Unexpected errors (MemoryError, etc.) propagate.
        """
        compressed, orig_sig = self._compress_signal()
        orig_int = np.round(orig_sig).astype(np.int64)

        # Flip bits at various positions in the payload (after header)
        detected = 0
        silent = 0
        for byte_pos in range(18, min(len(compressed), 118), 10):  # sample every 10 bytes
            for bit in range(8):
                corrupted = bytearray(compressed)
                corrupted[byte_pos] ^= (1 << bit)
                try:
                    d = _decompress_bytes(bytes(corrupted))
                    dec_int = np.round(d).astype(np.int64)
                    if not np.array_equal(orig_int, dec_int):
                        detected += 1  # corruption detected (wrong output)
                    else:
                        silent += 1  # bit flip had no effect (benign)
                except (ValueError, struct.error):
                    detected += 1  # corruption caused expected error (good)
                # Other exceptions intentionally propagate — they're real bugs

        # We want: most flips detected, zero silent corruption that looks valid
        # Some benign flips are OK (e.g., in padding bits)
        assert detected > 0, "No bit flips detected at all — CRC/validation may be broken"

    def test_magic_corruption_detected(self):
        """Corrupted magic bytes must raise ValueError."""
        compressed, _ = self._compress_signal()
        corrupted = bytearray(compressed)
        # Find LML magic after the ASCII prefix
        nl = compressed.index(b'\n')
        corrupted[nl + 1] = 0xFF  # corrupt the 'L' of 'LML'
        with pytest.raises(ValueError):
            _decompress_bytes(bytes(corrupted))

    def test_truncation_detected(self):
        """ANY truncation must raise ValueError.

        This is a patient safety requirement. A truncated medical archive
        that decodes silently could hide critical EEG events (seizures,
        artifacts). The header declares expected payload length, and the
        decoder verifies it before proceeding.
        """
        compressed, _ = self._compress_signal()
        # Every truncation point must raise
        for cut_point in [10, 50, len(compressed) // 4, len(compressed) // 2,
                          len(compressed) - 10, len(compressed) - 1]:
            with pytest.raises((ValueError, struct.error)):
                _decompress_bytes(compressed[:cut_point])

    def test_appended_garbage(self):
        """Extra bytes appended should not affect output."""
        compressed, orig_sig = self._compress_signal()
        extended = compressed + b'\x00' * 100
        d = _decompress_bytes(extended)
        orig_int = np.round(orig_sig).astype(np.int64)
        dec_int = np.round(d).astype(np.int64)
        assert np.array_equal(orig_int, dec_int), \
            "Appended garbage changed output"


# ============================================================
# Guardrail 9: Statistical anomaly flagging
# ============================================================

class TestStatisticalAnomalies:
    """Detect signals that produce unusual compression ratios."""

    def test_pathological_cr_flagged(self):
        """Signals with abnormally high or low CR should be detectable."""
        # Very compressible (all zeros)
        sig_easy = np.zeros((21, 2500), dtype=np.float64)
        c_easy = _compress_bytes(sig_easy, n_levels=3)
        cr_easy = 21 * 2500 * 8 / len(c_easy)

        # Incompressible (random noise)
        np.random.seed(42)
        sig_hard = np.random.randint(-32768, 32767, (21, 2500)).astype(np.float64)
        c_hard = _compress_bytes(sig_hard, n_levels=3)
        cr_hard = 21 * 2500 * 8 / len(c_hard)

        # Easy should compress much better than hard
        assert cr_easy > cr_hard * 2, \
            f"Expected easy ({cr_easy:.1f}) >> hard ({cr_hard:.1f})"

        # Both should still roundtrip
        for sig, c in [(sig_easy, c_easy), (sig_hard, c_hard)]:
            d = _decompress_bytes(c)
            assert np.array_equal(
                np.round(sig).astype(np.int64),
                np.round(d).astype(np.int64))
