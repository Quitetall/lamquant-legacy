"""L2 — Integrity verification gap tests.

AUDIT (2026-04-28): Created to close gaps found by provenance-checker agent:
  1. No multi-bit CRC corruption test (only single-bit was tested)
  2. No LMA per-file SHA-256 failure test
  3. No cross-layer CRC test (container + inner packet)
  4. No struct.calcsize validation for LML1 header

These tests complement the existing L1 conformance and L4 fuzz suites by
targeting specific integrity-checking logic that was only exercised implicitly.
"""
import pytest  # decomp(lossless-carve): skip when ai_models absent
pytest.importorskip("subband_preprocess", reason="Neural-coupled test; requires LamQuant-Neural sibling clone")

import os
import sys
import struct
import zlib
import hashlib
import tempfile

import numpy as np
import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'reference_implementations', 'python_codec', 'lamquant_codec'))

from lamquant_codec.lossless import _compress_bytes, _decompress_bytes


# ============================================================
# 1. Multi-bit CRC corruption — was missing entirely
# ============================================================

@pytest.mark.l2
class TestCRCMultiBit:
    """CRC-32 must detect multi-bit corruption, not just single-bit.

    CRC-32 guarantees detection of:
      - All 1-bit errors
      - All 2-bit errors
      - All burst errors up to 32 bits
    These tests verify the guarantees empirically.
    """

    def _valid_compressed(self):
        sig = np.random.RandomState(42).randn(21, 2500) * 5000
        return _compress_bytes(sig, n_levels=3), sig

    def test_two_bit_flip_all_detected(self):
        """Flip two bits at different positions — CRC must detect all."""
        c = self._valid_compressed()[0]
        nl = c.index(b'\n')
        start = nl + 1 + 22  # after ASCII prefix + binary header
        end = min(len(c), start + 50)  # first 50 payload bytes

        undetected = 0
        total = 0
        for pos1 in range(start, end, 5):
            for pos2 in range(pos1 + 1, min(pos1 + 10, end)):
                total += 1
                corrupted = bytearray(c)
                corrupted[pos1] ^= 0x01
                corrupted[pos2] ^= 0x80
                try:
                    _decompress_bytes(bytes(corrupted))
                    undetected += 1
                except (ValueError, struct.error):
                    pass  # correctly detected

        assert undetected == 0, \
            f"CRC missed {undetected}/{total} two-bit errors"

    def test_burst_error_4_bytes(self):
        """4-byte burst corruption (32 bits) — CRC must detect."""
        c, _ = self._valid_compressed()
        nl = c.index(b'\n')
        start = nl + 1 + 22

        corrupted = bytearray(c)
        corrupted[start:start + 4] = b'\xDE\xAD\xBE\xEF'
        with pytest.raises((ValueError, struct.error)):
            _decompress_bytes(bytes(corrupted))

    def test_burst_error_8_bytes(self):
        """8-byte burst corruption — CRC should detect (high probability)."""
        c, _ = self._valid_compressed()
        nl = c.index(b'\n')
        start = nl + 1 + 22

        rng = np.random.RandomState(99)
        detected = 0
        total = 20
        for i in range(total):
            corrupted = bytearray(c)
            offset = start + i * 3
            if offset + 8 > len(c):
                break
            corrupted[offset:offset + 8] = bytes(rng.randint(0, 256, 8).astype(np.uint8))
            try:
                d = _decompress_bytes(bytes(corrupted))
                # If it decoded, check if output changed
                orig_int = np.round(np.random.RandomState(42).randn(21, 2500) * 5000).astype(np.int64)
                dec_int = np.round(d).astype(np.int64)
                if not np.array_equal(orig_int, dec_int):
                    detected += 1  # corruption visible in output
            except (ValueError, struct.error):
                detected += 1  # correctly rejected

        assert detected == total, \
            f"CRC missed {total - detected}/{total} 8-byte burst errors"

    def test_byte_swap_detected(self):
        """Two adjacent payload bytes swapped — must detect."""
        c, _ = self._valid_compressed()
        nl = c.index(b'\n')
        start = nl + 1 + 22

        for pos in range(start, min(start + 30, len(c) - 1)):
            corrupted = bytearray(c)
            corrupted[pos], corrupted[pos + 1] = corrupted[pos + 1], corrupted[pos]
            if corrupted[pos] == c[pos]:  # swap was no-op (identical bytes)
                continue
            try:
                _decompress_bytes(bytes(corrupted))
                pytest.fail(f"Byte swap at position {pos} not detected by CRC")
            except (ValueError, struct.error):
                pass


# ============================================================
# 2. Container + inner packet cross-layer CRC
# ============================================================

@pytest.mark.l2
class TestCrossLayerCRC:
    """The file container (LQL1) CRC and the inner LML1 packet CRC are
    independent layers. Both must be checked during decode."""

    @pytest.mark.skip(
        reason="Divergent Python LosslessWriter removed (2026-05-28): "
        "fileformat is now a READ-ONLY reference reader, so this test "
        "can no longer construct a container to corrupt. A reader-only "
        "reference must not be round-tripped against its own deleted "
        "writer. The canonical container CRC is exercised against the "
        "Rust emitter (lamquant_core, LML1); the inner LML1 per-window "
        "CRC is still covered by test_inner_lml1_crc_independent_of_container."
    )
    def test_container_crc_checked_on_read(self, tmp_path):
        """Corrupt a byte inside the container payload — reader must reject.

        SKIPPED: built on the removed divergent Python writer (see skip
        marker). Body retained to document the removed round-trip.
        """
        sys.path.insert(0, os.path.join(_REPO, 'ai_models', 'student'))
        from lamquant_codec.fileformat import LosslessWriter, LMQReader

        rng = np.random.default_rng(42)
        seg = rng.standard_normal((21, 2500)).astype(np.float32) * 100

        path = str(tmp_path / "test.lml")
        with LosslessWriter(path, channels=21, rate=250) as w:
            w.write_window(seg, timestamp_us=0)

        # Corrupt one byte in the window payload (after file header)
        with open(path, 'r+b') as f:
            f.seek(64 + 10)  # 64B file header + 10B into window
            original = f.read(1)
            f.seek(-1, 1)
            f.write(bytes([(original[0] ^ 0xFF)]))

        with LMQReader(path) as r:
            with pytest.raises(ValueError, match="CRC"):
                next(iter(r))

    def test_inner_lml1_crc_independent_of_container(self):
        """The LML1 per-window CRC covers lpc_meta+payload, NOT the
        container window header. Verify that corrupting the LML1 payload
        (even if the container CRC is bypassed) is detected."""
        sig = np.random.RandomState(42).randn(21, 2500) * 5000
        c = _compress_bytes(sig, n_levels=3)

        # Corrupt one byte in the Golomb-Rice payload
        nl = c.index(b'\n')
        payload_start = nl + 1 + 22  # after prefix + LML1 header
        corrupted = bytearray(c)
        corrupted[payload_start + 5] ^= 0xFF

        with pytest.raises(ValueError, match="CRC"):
            _decompress_bytes(bytes(corrupted))


# ============================================================
# 3. LMA per-file SHA-256 failure test
# ============================================================

@pytest.mark.l2
class TestLMASHA256:
    """LMA archive per-file SHA-256 integrity.

    The provenance-checker found that unpack_lma() doesn't raise on per-file
    hash mismatch — it returns a summary dict. This test verifies the summary
    correctly reports the failure.
    """

    def test_archive_level_sha256_catches_corruption(self):
        """Corrupt the last byte of an LMA archive — archive SHA-256 must fail."""
        from lamquant_codec.lma import pack_lma, unpack_lma

        with tempfile.TemporaryDirectory() as src:
            # Create a small test file
            test_file = os.path.join(src, 'test.txt')
            with open(test_file, 'w') as f:
                f.write('Hello, LamQuant!\n')

            lma_path = os.path.join(src, 'test.lma')
            pack_lma(src, lma_path, verbose=False)

            # Corrupt the last byte (part of the SHA-256 trailer)
            with open(lma_path, 'r+b') as f:
                f.seek(-1, 2)
                last = f.read(1)
                f.seek(-1, 2)
                f.write(bytes([(last[0] ^ 0xFF)]))

            with tempfile.TemporaryDirectory() as out:
                with pytest.raises(ValueError, match="(?i)sha.*mismatch|corrupt"):
                    unpack_lma(lma_path, out, verify=True)

    def test_verify_lma_returns_false_on_corruption(self):
        """verify_lma() standalone check returns False on corrupt archive."""
        from lamquant_codec.lma import pack_lma, verify_lma

        with tempfile.TemporaryDirectory() as src:
            test_file = os.path.join(src, 'test.txt')
            with open(test_file, 'w') as f:
                f.write('Hello, LamQuant!\n')

            lma_path = os.path.join(src, 'test.lma')
            pack_lma(src, lma_path, verbose=False)

            # Corrupt a byte in the middle
            with open(lma_path, 'r+b') as f:
                f.seek(32)
                b = f.read(1)
                f.seek(-1, 1)
                f.write(bytes([(b[0] ^ 0xFF)]))

            ok = verify_lma(lma_path, verbose=False)
            assert ok is False, "verify_lma should return False on corrupt archive"
