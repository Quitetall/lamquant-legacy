"""EDF/EDF+/BDF reader tests.

Tests read_edf_digital, _parse_tal, _unpack_int24 for the complete
EDF family: EDF, EDF+C, EDF+D, BDF.
"""
import pytest  # decomp(lossless-carve): skip when ai_models absent
pytest.importorskip("subband_preprocess", reason="Neural-coupled test; requires LamQuant-Neural sibling clone")

import os
import struct
import tempfile

import numpy as np
import pytest

from lamquant_codec.edf_to_lml import (
    read_edf_digital, _parse_tal, _unpack_int24,
)


# ── Paths to real EDF files (skip if not present) ──
_DATA_ROOT = os.environ.get("LAMQUANT_DATA_ROOT", "/mnt/4tb/data")
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

_TUEG_EDF = os.path.join(_DATA_ROOT, "tueg_v2.0.1/edf/000/aaaaaaaa/s001_2015/01_tcp_ar/aaaaaaaa_s001_t000.edf")
_CHBMIT_EDF = os.path.join(_REPO_ROOT, "ai_models/dataset_sim/datasets/chbmit/chb01/chb01_03.edf")


class TestTALParser:
    """EDF+ Time-stamped Annotation List parsing."""

    def test_basic(self):
        tal = b"+0.0\x150.5\x14seizure onset\x14\x00+10.5\x14eyes closed\x14\x00"
        ann = _parse_tal(tal)
        assert len(ann) == 2
        assert ann[0]["onset"] == 0.0
        assert ann[0]["duration"] == 0.5
        assert ann[0]["description"] == "seizure onset"
        assert ann[1]["onset"] == 10.5
        assert ann[1]["description"] == "eyes closed"

    def test_no_duration(self):
        ann = _parse_tal(b"+5.0\x14spike\x14\x00")
        assert len(ann) == 1
        assert ann[0]["duration"] == 0.0

    def test_multi_description(self):
        tal = b"+30.5\x155.0\x14seizure\x14myoclonus\x14\x00"
        ann = _parse_tal(tal)
        assert len(ann) == 2
        assert ann[0]["description"] == "seizure"
        assert ann[1]["description"] == "myoclonus"

    def test_negative_onset(self):
        ann = _parse_tal(b"-0.5\x14pre-recording\x14\x00")
        assert len(ann) == 1
        assert ann[0]["onset"] == -0.5

    def test_empty(self):
        assert _parse_tal(b"\x00\x00") == []
        assert _parse_tal(b"") == []


class TestInt24Unpack:
    """BDF 24-bit signed integer unpacking."""

    def _pack24(self, v):
        if v < 0:
            v += 0x1000000
        return bytes([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF])

    @pytest.mark.parametrize("val", [0, 1, -1, 127, -128, 8388607, -8388608, 1000, -1000])
    def test_single_values(self, val):
        packed = self._pack24(val)
        unpacked = _unpack_int24(packed, 1)
        assert unpacked[0] == val

    def test_array(self):
        vals = [0, 1, -1, 8388607, -8388608, 100000, -100000]
        packed = b"".join(self._pack24(v) for v in vals)
        unpacked = _unpack_int24(packed, len(vals))
        for exp, got in zip(vals, unpacked):
            assert exp == got


class TestEDFReader:
    """read_edf_digital on real files."""

    @pytest.mark.skipif(not os.path.exists(_TUEG_EDF), reason="TUEG not available")
    def test_tueg(self):
        sig, meta = read_edf_digital(_TUEG_EDF)
        assert sig.shape[0] > 0
        assert sig.shape[1] > 0
        assert meta["format"] == "EDF"
        assert meta["bits_per_sample"] == 16
        assert meta["continuous"] is True
        assert len(meta["channels"]) == sig.shape[0]

    @pytest.mark.skipif(not os.path.exists(_CHBMIT_EDF), reason="CHB-MIT not available")
    def test_chbmit(self):
        sig, meta = read_edf_digital(_CHBMIT_EDF)
        assert sig.shape[0] > 0
        assert meta["sample_rate"] > 0

    def test_tiny_file_rejected(self, tmp_path):
        p = str(tmp_path / "tiny.edf")
        with open(p, "wb") as f:
            f.write(b"\x00" * 100)
        with pytest.raises(ValueError, match="too small"):
            read_edf_digital(p)

    def test_no_signals_rejected(self, tmp_path):
        # Valid 256-byte header but n_signals=0
        p = str(tmp_path / "nosig.edf")
        hdr = bytearray(256)
        hdr[0:8] = b"0       "
        hdr[236:244] = b"1       "  # 1 record
        hdr[244:252] = b"1.0     "  # 1s duration
        hdr[252:256] = b"0   "     # 0 signals
        with open(p, "wb") as f:
            f.write(hdr)
        with pytest.raises(ValueError, match="0 signals"):
            read_edf_digital(p)


class TestEDFToLMLRoundtrip:
    """Full EDF → LML → verify roundtrip."""

    @pytest.mark.skipif(not os.path.exists(_TUEG_EDF), reason="TUEG not available")
    def test_convert_verify(self, tmp_path):
        from lamquant_codec.edf_to_lml import convert_edf_to_lml
        p = str(tmp_path / "out.lml")
        stats = convert_edf_to_lml(_TUEG_EDF, p)
        assert stats.get("verified") is True
        assert stats.get("cr", 0) > 1.0
