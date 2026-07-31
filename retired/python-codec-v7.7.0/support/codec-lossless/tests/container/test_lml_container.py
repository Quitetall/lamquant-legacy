"""LML container format tests.

Tests write_lml_file / read_lml_file: header versioning, streaming
decompress, bounds checking, corruption detection.
"""
import os
import struct
import tempfile

import numpy as np
import pytest

from lamquant_codec.edf_to_lml import (
    LML_MAGIC,
    write_lml_file, read_lml_file, _read_header,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def tmp(tmp_path):
    return tmp_path


class TestContainerRoundtrip:
    """Write → read must be bit-exact."""

    @pytest.mark.parametrize("shape", [
        (21, 2500), (1, 100), (64, 5000), (256, 1250),
    ])
    def test_roundtrip(self, rng, tmp, shape):
        sig = rng.integers(-5000, 5000, shape, dtype=np.int64)
        p = str(tmp / "rt.lml")
        write_lml_file(p, sig, {"sample_rate": 250})
        rec, _ = read_lml_file(p)
        assert np.array_equal(sig, rec[:shape[0], :shape[1]])

    def test_multiwindow(self, rng, tmp):
        sig = rng.integers(-5000, 5000, (21, 25000), dtype=np.int64)
        p = str(tmp / "multi.lml")
        write_lml_file(p, sig, {"sample_rate": 250})
        rec, _ = read_lml_file(p)
        assert np.array_equal(sig, rec[:21, :25000])

    def test_metadata_preserved(self, rng, tmp):
        sig = rng.integers(-1000, 1000, (21, 2500), dtype=np.int64)
        meta = {"sample_rate": 256, "patient_id": "test123", "custom": [1, 2]}
        p = str(tmp / "meta.lml")
        write_lml_file(p, sig, meta)
        _, rmeta = read_lml_file(p)
        assert rmeta["sample_rate"] == 256
        assert rmeta["patient_id"] == "test123"
        assert rmeta["custom"] == [1, 2]


class TestEdgeCaseRejection:
    """Invalid inputs must raise ValueError, not crash."""

    def test_0_channels(self, tmp):
        with pytest.raises(ValueError, match="0-channel"):
            write_lml_file(str(tmp / "bad.lml"),
                           np.zeros((0, 100), dtype=np.int64), {})

    def test_0_samples(self, tmp):
        with pytest.raises(ValueError, match="0-sample"):
            write_lml_file(str(tmp / "bad.lml"),
                           np.zeros((21, 0), dtype=np.int64), {})


class TestCorruptionDetection:
    """Corrupt files must be caught with clear error messages."""

    def test_empty_file(self, tmp):
        p = str(tmp / "empty.lml")
        open(p, "w").close()
        with pytest.raises(ValueError, match="too small"):
            read_lml_file(p)

    def test_wrong_magic(self, tmp):
        p = str(tmp / "bad.lml")
        with open(p, "wb") as f:
            f.write(b"FAKE" + b"\x00" * 30)
        with pytest.raises(ValueError, match="Not an LML"):
            read_lml_file(p)

    def test_truncated_header(self, tmp):
        p = str(tmp / "trunc.lml")
        with open(p, "wb") as f:
            f.write(LML_MAGIC + b"\x01\x00")
        with pytest.raises(ValueError, match="[Tt]runcat"):
            read_lml_file(p)

    def test_truncated_payload(self, rng, tmp):
        sig = rng.integers(-5000, 5000, (21, 2500), dtype=np.int64)
        p = str(tmp / "full.lml")
        write_lml_file(p, sig, {"sample_rate": 250})
        data = open(p, "rb").read()
        p2 = str(tmp / "half.lml")
        with open(p2, "wb") as f:
            f.write(data[: len(data) // 2])
        with pytest.raises(ValueError, match="(?i)truncat|incomplete"):
            read_lml_file(p2)

    def test_crc_bit_flip(self, rng, tmp):
        sig = rng.integers(-5000, 5000, (21, 2500), dtype=np.int64)
        p = str(tmp / "ok.lml")
        write_lml_file(p, sig, {"sample_rate": 250})
        data = bytearray(open(p, "rb").read())
        data[len(data) // 2] ^= 0xFF
        p2 = str(tmp / "flipped.lml")
        with open(p2, "wb") as f:
            f.write(data)
        with pytest.raises(ValueError, match="(?i)crc|mismatch|corrupt"):
            read_lml_file(p2)

    def test_corrupt_metadata(self, rng, tmp):
        sig = rng.integers(-5000, 5000, (21, 2500), dtype=np.int64)
        p = str(tmp / "ok.lml")
        write_lml_file(p, sig, {"sample_rate": 250})
        data = bytearray(open(p, "rb").read())
        # Corrupt JSON metadata (starts at byte 32 after 32-byte LML header)
        data[32] = 0xFF
        data[33] = 0xFE
        p2 = str(tmp / "badmeta.lml")
        with open(p2, "wb") as f:
            f.write(data)
        with pytest.raises(ValueError, match="(?i)metadata|corrupt|json"):
            read_lml_file(p2)


class TestMagic:
    """LML magic byte handling."""

    def test_magic_written(self, rng, tmp):
        sig = rng.integers(-5000, 5000, (21, 2500), dtype=np.int64)
        p = str(tmp / "magic.lml")
        write_lml_file(p, sig, {"sample_rate": 250})
        with open(p, "rb") as f:
            assert f.read(4) == LML_MAGIC


class TestSHA256:
    """SHA-256 integrity at the container level."""

    def test_sha256_roundtrip(self, rng, tmp):
        import hashlib
        sig = rng.integers(-5000, 5000, (21, 2500), dtype=np.int64)
        p = str(tmp / "sha.lml")
        write_lml_file(p, sig, {"sample_rate": 250})
        h1 = hashlib.sha256(sig.tobytes()).hexdigest()
        rec, _ = read_lml_file(p)
        h2 = hashlib.sha256(rec[:21, :2500].tobytes()).hexdigest()
        assert h1 == h2
