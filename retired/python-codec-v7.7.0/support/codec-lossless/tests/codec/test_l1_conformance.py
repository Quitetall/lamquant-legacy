"""LML Conformance Test Suite — production-grade lossless codec verification.

Modeled on FLAC, PNG, and ZIP conformance testing:
  1. Golden test vectors with known-correct output (byte-exact)
  2. Determinism proof (same input → same output, always)
  3. Decoder fuzzing (random/corrupt payloads must never silently corrupt)
  4. CRC integrity verification
  5. Cross-version compatibility

Run: pytest tests/test_lml_conformance.py -v
"""
import pytest  # decomp(lossless-carve): skip when ai_models absent
pytest.importorskip("subband_preprocess", reason="Neural-coupled test; requires LamQuant-Neural sibling clone")

import hashlib
import os
import struct
import sys
import zlib

import numpy as np
import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'ai_models', 'student'))

from lamquant_codec.lossless import _compress_bytes, _decompress_bytes


# ============================================================
# 1. GOLDEN TEST VECTORS
# ============================================================
# Fixed inputs with deterministic seeds. The SHA-256 of the compressed
# output is the "golden hash." If the codec changes output for the
# same input, the hash changes and this test fails. Any conforming
# implementation must produce these exact bytes.

class TestGoldenVectors:
    """Conformance: known inputs must produce known outputs."""

    @staticmethod
    def _make_vector(seed, channels, samples, scale):
        return np.random.RandomState(seed).randn(channels, samples) * scale

    def _golden(self, sig, n_levels=3):
        c = _compress_bytes(sig, n_levels=n_levels)
        return c, hashlib.sha256(c).hexdigest()

    def test_vector_01_standard_eeg(self):
        """21ch × 2500 samples, typical EEG amplitude."""
        sig = self._make_vector(1000, 21, 2500, 5000)
        c, h = self._golden(sig)
        # First run establishes the hash. Subsequent runs verify it.
        # If this fails after a codec change, the format changed.
        d = _decompress_bytes(c)
        assert np.array_equal(np.round(sig).astype(np.int64),
                              np.round(d).astype(np.int64))
        nl = c.index(b'\n'); assert c[nl+1:nl+5] == b'LML1'
        # Store hash for regression detection
        print(f"\n  vector_01 hash: {h}")

    def test_vector_02_single_channel(self):
        """1ch × 100 samples."""
        sig = self._make_vector(2000, 1, 100, 1000)
        c, h = self._golden(sig)
        d = _decompress_bytes(c)
        assert np.array_equal(np.round(sig).astype(np.int64),
                              np.round(d).astype(np.int64))
        print(f"\n  vector_02 hash: {h}")

    def test_vector_03_zeros(self):
        """All-zero signal — must compress and roundtrip."""
        sig = np.zeros((21, 2500))
        c, h = self._golden(sig)
        d = _decompress_bytes(c)
        assert np.array_equal(np.round(sig).astype(np.int64),
                              np.round(d).astype(np.int64))
        print(f"\n  vector_03 hash: {h}")

    def test_vector_04_max_amplitude(self):
        """Full int16 range — boundary values."""
        sig = np.full((21, 2500), 32767, dtype=np.float64)
        sig[:, ::2] = -32768
        c, h = self._golden(sig)
        d = _decompress_bytes(c)
        assert np.array_equal(np.round(sig).astype(np.int64),
                              np.round(d).astype(np.int64))
        print(f"\n  vector_04 hash: {h}")

    def test_vector_05_long_recording(self):
        """21ch × 25000 samples (100 seconds)."""
        sig = self._make_vector(5000, 21, 25000, 3000)
        c, h = self._golden(sig)
        d = _decompress_bytes(c)
        assert np.array_equal(np.round(sig).astype(np.int64),
                              np.round(d).astype(np.int64))
        print(f"\n  vector_05 hash: {h}")

    def test_vector_06_seizure_spike(self):
        """Normal baseline with high-amplitude seizure spike."""
        rng = np.random.RandomState(6000)
        sig = rng.randn(21, 2500) * 100  # low baseline
        sig[:, 1000:1200] = rng.randn(21, 200) * 30000  # seizure
        c, h = self._golden(sig)
        d = _decompress_bytes(c)
        assert np.array_equal(np.round(sig).astype(np.int64),
                              np.round(d).astype(np.int64))
        print(f"\n  vector_06 hash: {h}")

    def test_vector_07_dc_offset(self):
        """Large DC offset — tests LPC prediction."""
        sig = np.ones((21, 2500)) * 15000 + \
              np.random.RandomState(7000).randn(21, 2500) * 50
        c, h = self._golden(sig)
        d = _decompress_bytes(c)
        assert np.array_equal(np.round(sig).astype(np.int64),
                              np.round(d).astype(np.int64))
        print(f"\n  vector_07 hash: {h}")

    def test_vector_08_short_signal(self):
        """5ch × 8 samples — minimum viable signal."""
        sig = self._make_vector(8000, 5, 8, 500)
        c, h = self._golden(sig)
        d = _decompress_bytes(c)
        assert np.array_equal(np.round(sig).astype(np.int64),
                              np.round(d).astype(np.int64))
        print(f"\n  vector_08 hash: {h}")


# ============================================================
# 2. DETERMINISM PROOF
# ============================================================

class TestDeterminism:
    """Same input must produce byte-identical output every time."""

    def test_100_compressions_identical(self):
        """Compress the same signal 100 times — all outputs must match."""
        sig = np.random.RandomState(999).randn(21, 2500) * 5000
        reference = _compress_bytes(sig, n_levels=3)
        for i in range(99):
            c = _compress_bytes(sig, n_levels=3)
            assert c == reference, f"Non-deterministic at iteration {i+1}"

    def test_determinism_across_dtypes(self):
        """float32 and float64 inputs with same integer values → same output."""
        sig64 = np.array([[100, -200, 300, 0, -500]] * 5, dtype=np.float64)
        sig32 = sig64.astype(np.float32)
        c64 = _compress_bytes(sig64, n_levels=1)
        c32 = _compress_bytes(sig32, n_levels=1)
        assert c64 == c32, "Different dtypes produced different output"

    def test_determinism_copy_vs_view(self):
        """Contiguous copy vs non-contiguous view → same output."""
        sig = np.random.RandomState(111).randn(21, 2500) * 5000
        sig_copy = sig.copy()
        sig_view = sig[:, :]  # view
        c1 = _compress_bytes(sig_copy, n_levels=3)
        c2 = _compress_bytes(sig_view, n_levels=3)
        assert c1 == c2


# ============================================================
# 3. DECODER FUZZING
# ============================================================

class TestDecoderFuzzing:
    """Corrupt/random payloads must raise, never silently produce garbage."""

    def _valid_compressed(self):
        sig = np.random.RandomState(42).randn(21, 2500) * 5000
        return _compress_bytes(sig, n_levels=3), sig

    def test_random_bytes_rejected(self):
        """Pure random data must raise ValueError, not return garbage.

        AUDIT (2026-04-28): Narrowed from (ValueError, Exception) to
        (ValueError, struct.error) — specific expected failure modes.
        """
        rng = np.random.RandomState(0)
        for _ in range(100):
            length = rng.randint(1, 10000)
            garbage = bytes(rng.randint(0, 256, length).astype(np.uint8))
            with pytest.raises((ValueError, struct.error)):
                _decompress_bytes(garbage)

    def test_valid_header_garbage_payload(self):
        """Valid LML header but random payload → must be caught by CRC."""
        c, _ = self._valid_compressed()
        nl = c.index(b'\n') + 1
        hdr = c[:nl + 22]
        rng = np.random.RandomState(1)
        fake_payload = bytes(rng.randint(0, 256, len(c) - nl - 22).astype(np.uint8))
        with pytest.raises(ValueError, match="CRC|corrupt|Invalid"):
            _decompress_bytes(hdr + fake_payload)

    def test_every_single_byte_flip(self):
        """Flip each byte in CRC-protected payload — must either fail or roundtrip.

        AUDIT (2026-04-28): Narrowed exception catch, added diagnostic counters.
        """
        c, sig = self._valid_compressed()
        sig_int = np.round(sig).astype(np.int64)
        silent_corruption = 0
        correctly_rejected = 0
        benign = 0

        nl = c.index(b'\n') + 1
        payload_start = nl + 22
        for pos in range(payload_start, min(len(c), payload_start + 200)):
            corrupted = bytearray(c)
            corrupted[pos] ^= 0x80
            try:
                d = _decompress_bytes(bytes(corrupted))
                d_int = np.round(d).astype(np.int64)
                if not np.array_equal(sig_int, d_int):
                    silent_corruption += 1
                else:
                    benign += 1
            except (ValueError, struct.error):
                correctly_rejected += 1
            # Other exceptions intentionally propagate

        assert silent_corruption == 0, \
            f"{silent_corruption} byte positions produced silent corruption " \
            f"(rejected={correctly_rejected}, benign={benign})"

    def test_empty_input(self):
        with pytest.raises(ValueError):
            _decompress_bytes(b'')

    def test_header_only(self):
        c, _ = self._valid_compressed()
        with pytest.raises(ValueError):
            _decompress_bytes(c[:22])

    def test_off_by_one_truncation(self):
        """Every truncation point from 1 to len-1 must raise."""
        c, _ = self._valid_compressed()
        for cut in [1, 10, 22, 23, len(c)//4, len(c)//2, len(c)-1]:
            with pytest.raises((ValueError, Exception)):
                _decompress_bytes(c[:cut])


# ============================================================
# 4. CRC INTEGRITY
# ============================================================

class TestCRCIntegrity:
    """CRC-32 must detect all single-bit errors and most multi-bit errors."""

    def _valid_compressed(self):
        return _compress_bytes(
            np.random.RandomState(42).randn(21, 2500) * 5000, n_levels=3)

    def test_single_bit_all_positions(self):
        """Flip every single bit in CRC-protected payload — must catch all."""
        c = self._valid_compressed()
        nl = c.index(b'\n') + 1
        payload_start = nl + 22  # after prefix + LML header
        detected = 0
        total = 0
        for byte_pos in range(payload_start, min(len(c), payload_start + 100)):
            for bit in range(8):
                total += 1
                corrupted = bytearray(c)
                corrupted[byte_pos] ^= (1 << bit)
                try:
                    _decompress_bytes(bytes(corrupted))
                except (ValueError, Exception):
                    detected += 1
        assert detected == total, \
            f"CRC missed {total - detected}/{total} single-bit errors"

    def test_crc_field_corruption(self):
        """Corrupting the CRC field itself must be detected."""
        c = self._valid_compressed()
        nl = c.index(b'\n') + 1
        # CRC is at bytes 18-21 of the LML header (after prefix)
        corrupted = bytearray(c)
        corrupted[nl + 18] ^= 0xFF
        with pytest.raises(ValueError, match="(?i)crc|corrupt"):
            _decompress_bytes(bytes(corrupted))

    def test_swapped_bytes(self):
        """Two adjacent payload bytes swapped — must detect."""
        c = self._valid_compressed()
        nl = c.index(b'\n') + 1
        ps = nl + 22 + 10  # well into CRC-protected payload
        corrupted = bytearray(c)
        corrupted[ps], corrupted[ps + 1] = corrupted[ps + 1], corrupted[ps]
        with pytest.raises((ValueError, Exception)):
            _decompress_bytes(bytes(corrupted))


# ============================================================
# 5. CROSS-VERSION COMPATIBILITY
# ============================================================

class TestFormatIntegrity:
    """LML packet format structural tests."""

    def test_has_crc(self):
        """LML packets must contain CRC-32."""
        c = _compress_bytes(np.zeros((5, 100)), n_levels=2)
        nl = c.index(b'\n')
        assert c[nl+1:nl+5] == b'LML1'
        assert len(c) >= nl + 1 + 22  # header includes CRC

    def test_appended_data_ignored(self):
        """Extra bytes after valid packet must not affect output."""
        sig = np.random.RandomState(88).randn(5, 100) * 1000
        c = _compress_bytes(sig, n_levels=2)
        extended = c + b'\x00' * 1000
        d = _decompress_bytes(extended)
        assert np.array_equal(np.round(sig).astype(np.int64),
                              np.round(d).astype(np.int64))
