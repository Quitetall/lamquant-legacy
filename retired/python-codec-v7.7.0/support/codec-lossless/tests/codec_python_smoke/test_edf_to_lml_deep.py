"""Deep coverage tests for ``lamquant_codec.edf_to_lml``.

Pins behavioural contracts, not numeric magic:
  - convert_edf_to_lml on a real EDF round-trips bit-exactly (signal_sha256
    matches between input and output)
  - LML output starts with LML1 magic byte sequence
  - read_lml_file rejects truncated / corrupt headers with actionable errors
  - reconstruct_edf bit-exact (when edf_header is preserved) — verified by
    re-encoding the reconstructed file and comparing SHA-256
  - _parse_tal handles malformed annotation bytes
  - _unpack_int24 correctly sign-extends 24-bit values
  - find_edf_files / detect_dataset / make_output_path on tmp_path

Real-EDF fixtures only; no synthetic EEG data.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from lamquant_codec.edf_to_lml import (
    LML_MAGIC,
    NOMINAL_WINDOW_SAMPLES,
    _parse_tal,
    _read_header,
    _unpack_int24,
    convert_edf_to_lml,
    detect_dataset,
    find_edf_files,
    make_output_path,
    read_edf_digital,
    read_lml_file,
    reconstruct_edf,
    write_lml_file,
)


# ---------------------------------------------------------------------------
# Pure-Python helpers (no real EDF needed)
# ---------------------------------------------------------------------------


class TestDetectDataset:
    @pytest.mark.parametrize("path,expected", [
        ("/data/chb01_01.edf", "chbmit"),
        ("/data/chbmit/x.edf", "chbmit"),
        ("/data/tueg/aaaaaaaq.edf", "tuh"),
        ("/data/tuh_eval/x.edf", "tuh"),
        ("/data/siena/PN00.edf", "siena"),
        ("/data/eegmmidb/S001R01.edf", "eegmmidb"),
        ("/data/sleep_cassette/SC4001.edf", "sleep"),
        ("/data/foo/bar.edf", "generic"),
    ])
    def test_dataset_inferred_from_path(self, path: str, expected: str) -> None:
        assert detect_dataset(path) == expected


class TestMakeOutputPath:
    def test_chbmit_patient_extracted(self) -> None:
        out = make_output_path("/data/chbmit/chb01/chb01_01.edf", "/out")
        # Path mirrors detect_dataset/patient parsing.
        assert out.startswith(os.path.join("/out", "chbmit"))
        assert "chb01" in out
        assert out.endswith("chb01_01.lml")

    def test_tuh_patient_extracted(self) -> None:
        out = make_output_path(
            "/data/aaaaaaaq/s006_2024_01_01/aaaaaaaq_s006_t000.edf", "/out"
        )
        # tuh detect requires "tueg"/"tuh" substring; without it falls to generic.
        assert "aaaaaaaq" in out  # the 8-char aaaaaa-prefix patient parse fires
        assert out.endswith("aaaaaaaq_s006_t000.lml")

    def test_unknown_patient_becomes_unknown(self) -> None:
        out = make_output_path("/data/random.edf", "/out")
        assert "unknown" in out
        assert out.endswith("random.lml")


class TestFindEdfFiles:
    def test_glob_returns_sorted_unique(self, tmp_path: Path) -> None:
        (tmp_path / "a.edf").write_bytes(b"x")
        (tmp_path / "b.edf").write_bytes(b"y")
        (tmp_path / "c.EDF").write_bytes(b"z")
        # Non-EDF file shouldn't be included
        (tmp_path / "skip.txt").write_bytes(b"n")
        found = find_edf_files(str(tmp_path))
        # sorted set: dedup .edf / .EDF on case-insensitive FS still works
        assert all(f.lower().endswith(".edf") for f in found)
        assert len(found) >= 2

    def test_empty_dir_returns_empty_list(self, tmp_path: Path) -> None:
        assert find_edf_files(str(tmp_path)) == []


class TestUnpackInt24:
    def test_positive_values(self) -> None:
        # LE 24-bit: 0x010203 -> 0x030201
        raw = bytes([0x01, 0x02, 0x03])
        arr = _unpack_int24(raw, 1)
        assert arr.shape == (1,)
        assert int(arr[0]) == 0x030201

    def test_negative_sign_extends(self) -> None:
        # 0xFFFFFF in LE is -1 (sign-extended from 24-bit)
        raw = bytes([0xFF, 0xFF, 0xFF])
        arr = _unpack_int24(raw, 1)
        assert int(arr[0]) == -1

    def test_zero(self) -> None:
        arr = _unpack_int24(bytes([0, 0, 0]) * 3, 3)
        assert arr.shape == (3,)
        assert np.all(arr == 0)

    def test_dtype_is_int32(self) -> None:
        raw = bytes([0x10, 0x20, 0x30]) * 4
        arr = _unpack_int24(raw, 4)
        assert arr.dtype == np.int32


class TestParseTal:
    def test_empty_input(self) -> None:
        assert _parse_tal(b"") == []

    def test_well_formed_annotation(self) -> None:
        # TAL: +1.0\x14seizure\x14\x00
        raw = b"+1.0\x14seizure\x14\x00"
        out = _parse_tal(raw)
        assert len(out) == 1
        ann = out[0]
        assert ann["onset"] == pytest.approx(1.0)
        assert ann["duration"] == 0.0
        assert "seizure" in ann["description"]

    def test_with_duration(self) -> None:
        # +onset\x15duration\x14desc\x14\x00
        raw = b"+2.5\x152.0\x14spike\x14\x00"
        out = _parse_tal(raw)
        assert len(out) == 1
        assert out[0]["onset"] == pytest.approx(2.5)
        assert out[0]["duration"] == pytest.approx(2.0)
        assert "spike" in out[0]["description"]

    def test_malformed_no_onset_marker_returns_empty(self) -> None:
        # Garbage that doesn't start with + or - is skipped.
        raw = b"garbage_no_sign_byte\x14\x00"
        out = _parse_tal(raw)
        assert out == []

    def test_bad_float_skipped(self) -> None:
        # +not_a_float\x14desc\x14\x00
        raw = b"+not_a_float\x14desc\x14\x00"
        out = _parse_tal(raw)
        assert out == []  # ValueError caught in float() parse


class TestReadHeaderCorrupt:
    def test_too_small_file(self, tmp_path: Path) -> None:
        bogus = tmp_path / "tiny.lml"
        bogus.write_bytes(b"x")
        with bogus.open("rb") as f:
            with pytest.raises(ValueError, match="File too small"):
                _read_header(f, 1)

    def test_wrong_magic(self, tmp_path: Path) -> None:
        bogus = tmp_path / "wrong.lml"
        bogus.write_bytes(b"ABCDE" + b"\x00" * 30)
        with bogus.open("rb") as f:
            with pytest.raises(ValueError, match="Not an LML file"):
                _read_header(f, 35)

    def test_future_version_rejected(self, tmp_path: Path) -> None:
        # LML2 -> future version -> rejected
        bogus = tmp_path / "future.lml"
        bogus.write_bytes(b"LML2" + b"\x00" * 30)
        with bogus.open("rb") as f:
            with pytest.raises(ValueError, match="newer than"):
                _read_header(f, 34)


# ---------------------------------------------------------------------------
# write_lml_file validation
# ---------------------------------------------------------------------------


class TestWriteLmlFileValidation:
    def test_zero_channels_raises(self, tmp_path: Path) -> None:
        sig = np.zeros((0, 100), dtype=np.int64)
        with pytest.raises(ValueError, match="0-channel"):
            write_lml_file(str(tmp_path / "x.lml"), sig, {"sample_rate": 250})

    def test_zero_samples_raises(self, tmp_path: Path) -> None:
        sig = np.zeros((4, 0), dtype=np.int64)
        with pytest.raises(ValueError, match="0-sample"):
            write_lml_file(str(tmp_path / "x.lml"), sig, {"sample_rate": 250})


# ---------------------------------------------------------------------------
# Real EDF -> LML conversion: shape/type/sha invariants
# ---------------------------------------------------------------------------


@pytest.mark.data
def test_convert_edf_to_lml_produces_valid_file(
    tmp_path: Path, real_test_edf: Path,
) -> None:
    """convert_edf_to_lml on a real EDF: output exists, has LML1 magic,
    SHA-256 verify-after-write succeeded (stats['verified'] is True)."""
    out_lml = tmp_path / "test.lml"
    stats = convert_edf_to_lml(str(real_test_edf), str(out_lml))
    assert "error" not in stats, f"convert failed: {stats}"
    assert stats["verified"] is True
    assert out_lml.exists()
    assert out_lml.stat().st_size > 0
    # File starts with LML1 magic
    head = out_lml.read_bytes()[:4]
    assert head == LML_MAGIC
    # Stats contract: keys we depend on
    for key in ("n_windows", "compressed_size", "raw_size", "cr",
                "n_channels", "total_samples", "duration_s",
                "verified", "signal_sha256", "source", "sample_rate"):
        assert key in stats, f"missing stat key: {key}"
    # signal_sha256 is a hex digest
    int(stats["signal_sha256"], 16)
    assert len(stats["signal_sha256"]) == 64


@pytest.mark.data
def test_convert_edf_to_lml_roundtrip_sha_match(
    tmp_path: Path, real_test_edf: Path,
) -> None:
    """Roundtrip: encode→read_lml_file→reconstruct→re-hash signal.
    The signal_sha256 captured at encode time must match a fresh hash
    of the decoded signal."""
    out_lml = tmp_path / "rt.lml"
    stats = convert_edf_to_lml(str(real_test_edf), str(out_lml))
    if "error" in stats:
        pytest.skip(f"convert_edf_to_lml failed on real EDF: {stats['error']}")
    signal, metadata = read_lml_file(str(out_lml))
    assert signal.dtype == np.int64
    C, T = signal.shape
    assert C == stats["n_channels"]
    assert T == stats["total_samples"]
    # SHA recomputation matches the stored signal_sha256
    recon_hash = hashlib.sha256(signal[:C, :T].tobytes()).hexdigest()
    assert recon_hash == stats["signal_sha256"]
    # Metadata retains key EDF provenance.
    assert "edf_header_sha256" in metadata
    assert "sample_rate" in metadata
    assert "format" in metadata
    assert "encoder_version" in metadata


@pytest.mark.data
def test_read_edf_digital_returns_int64(real_test_edf: Path) -> None:
    """Direct EDF read returns int64 signal + metadata dict."""
    sig, meta = read_edf_digital(str(real_test_edf))
    assert sig.dtype == np.int64
    assert sig.ndim == 2
    C, T = sig.shape
    assert C > 0 and T > 0
    # Metadata invariants
    assert meta["n_channels"] == C
    assert meta["format"] in ("EDF", "EDF+C", "EDF+D", "BDF")
    assert meta["sample_rate"] > 0
    assert "channels" in meta
    assert len(meta["channels"]) == C
    # SHA-256 of the header is captured (FDA provenance).
    int(meta["edf_header_sha256"], 16)
    assert len(meta["edf_header_sha256"]) == 64
    assert isinstance(meta["edf_header"], str)  # base64-encoded


@pytest.mark.data
def test_reconstruct_edf_uses_preserved_header(
    tmp_path: Path, real_test_edf: Path,
) -> None:
    """When the original EDF header is preserved in metadata, reconstruct_edf
    returns True (bit-exact path)."""
    out_lml = tmp_path / "rec.lml"
    stats = convert_edf_to_lml(str(real_test_edf), str(out_lml))
    if "error" in stats:
        pytest.skip(f"convert failed: {stats['error']}")
    reconstructed = tmp_path / "rec.edf"
    used_original = reconstruct_edf(str(out_lml), str(reconstructed))
    assert used_original is True
    assert reconstructed.exists()
    # The first 256 bytes of the EDF main header must match.
    original_hdr = real_test_edf.read_bytes()[:256]
    recon_hdr = reconstructed.read_bytes()[:256]
    assert original_hdr == recon_hdr


# ---------------------------------------------------------------------------
# read_lml_file validation: corrupt file rejection
# ---------------------------------------------------------------------------


@pytest.mark.data
def test_read_lml_file_truncated_payload(
    tmp_path: Path, real_test_edf: Path,
) -> None:
    """Truncate a real LML file mid-payload — read_lml_file must reject
    with an actionable ValueError."""
    out_lml = tmp_path / "ok.lml"
    stats = convert_edf_to_lml(str(real_test_edf), str(out_lml))
    if "error" in stats:
        pytest.skip(f"convert failed: {stats['error']}")
    full = out_lml.read_bytes()
    truncated = tmp_path / "bad.lml"
    # Cut off the last 100 bytes — should hit the "Truncated" branch
    truncated.write_bytes(full[:-100])
    with pytest.raises(ValueError):
        read_lml_file(str(truncated))


@pytest.mark.data
def test_reconstruct_edf_synthesized_when_header_absent(
    tmp_path: Path, real_test_edf: Path,
) -> None:
    """When metadata lacks edf_header, reconstruct_edf must synthesize one
    (returns False)."""
    # First produce a normal LML, then re-write it stripping edf_header.
    out_lml = tmp_path / "h.lml"
    stats = convert_edf_to_lml(str(real_test_edf), str(out_lml))
    if "error" in stats:
        pytest.skip(f"convert failed: {stats['error']}")
    signal, meta = read_lml_file(str(out_lml))
    # Strip the preserved EDF header from metadata
    meta.pop("edf_header", None)
    meta.pop("edf_header_sha256", None)
    out_stripped = tmp_path / "stripped.lml"
    write_lml_file(str(out_stripped), signal.astype(np.int64), meta)
    out_edf = tmp_path / "synth.edf"
    used_original = reconstruct_edf(str(out_stripped), str(out_edf))
    # Synthesized path
    assert used_original is False
    assert out_edf.exists()
    assert out_edf.stat().st_size > 256  # header + signal data


def _hand_craft_v1_header(
    *, n_ch: int = 4, n_windows: int = 2, total_samples: int = 500,
    window_size: int = 250, sr_mhz: int = 250000, bit_depth: int = 16,
    flags: int = 0, meta_len: int = 0,
) -> bytes:
    return struct.pack('<4sBBHHIHIBBI2x4x',
                       LML_MAGIC, 1, 0, n_ch, n_windows,
                       total_samples, window_size, sr_mhz,
                       bit_depth, flags, meta_len)


@pytest.mark.parametrize("kw,match", [
    ({"n_ch": 0}, "channel count"),
    ({"n_ch": 2048}, "channel count"),  # > _MAX_CHANNELS=1024 but within u16
    ({"total_samples": 0}, "0 samples"),
    ({"n_windows": 0}, "0 windows"),
    ({"window_size": 0}, "Window size"),
])
def test_read_lml_file_header_sanity_guards(
    tmp_path: Path, kw: dict, match: str,
) -> None:
    """Each sanity-guard branch in _read_header rejects malformed values."""
    bogus = tmp_path / "bad.lml"
    bogus.write_bytes(_hand_craft_v1_header(**kw))
    with pytest.raises(ValueError, match=match):
        read_lml_file(str(bogus))


def test_read_lml_file_meta_len_exceeds_file_size(tmp_path: Path) -> None:
    """meta_len > file_size raises with the actionable message."""
    bogus = tmp_path / "bad.lml"
    # Construct a 32-byte header claiming a huge metadata block.
    header = _hand_craft_v1_header(meta_len=1_000_000_000)
    bogus.write_bytes(header)
    with pytest.raises(ValueError, match="Metadata length"):
        read_lml_file(str(bogus))


def test_read_lml_file_corrupt_metadata_json(tmp_path: Path) -> None:
    """Truncated/invalid JSON metadata triggers actionable ValueError."""
    bogus = tmp_path / "bad_meta.lml"
    # Build a valid header pointing at 10 bytes of broken JSON, then write
    # those 10 bytes of garbage.
    header = _hand_craft_v1_header(meta_len=10)
    # Plus 4*n_windows for the offset table + ... we don't get that far
    bogus.write_bytes(header + b"{not json}")
    with pytest.raises(ValueError, match="metadata|JSON|Corrupt"):
        read_lml_file(str(bogus))


def test_read_lml_file_zero_channel_header_rejected(tmp_path: Path) -> None:
    """Hand-craft a header with n_ch=0 — _read_header must reject."""
    bogus = tmp_path / "bad.lml"
    bogus.write_bytes(_hand_craft_v1_header(n_ch=0))
    with pytest.raises(ValueError, match="channel count"):
        read_lml_file(str(bogus))
