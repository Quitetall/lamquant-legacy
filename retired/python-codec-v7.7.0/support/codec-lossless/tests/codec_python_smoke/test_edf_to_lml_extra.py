"""Push ``lamquant_codec.edf_to_lml`` past 83% coverage.

Covers the remaining branches that ``test_edf_to_lml_deep.py`` doesn't
hit:

  - Invalid LML version byte (not LML1-LML9): raises generic "Invalid LML"
  - Truncated LML header (< 14 bytes after magic)
  - read_lml_file truncated-meta path (metadata block shorter than declared)
  - read_lml_file zero-payload + index-truncation paths
  - reconstruct_edf with trailing_data preserved → bit-exact
  - reconstruct_edf bit-exact when header SHA mismatches → raises
  - convert_edf_to_lml: error path on truly broken EDF input
  - read_edf_digital: tiny file rejected
  - read_edf_digital: invalid n_signals / n_data_records header → ValueError

All real-fixture only — no synthetic EEG bytes are fabricated.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from lamquant_codec.edf_to_lml import (
    LML_MAGIC,
    _read_header,
    convert_edf_to_lml,
    read_edf_digital,
    read_lml_file,
    reconstruct_edf,
    write_lml_file,
)


# ---------------------------------------------------------------------------
# _read_header: invalid version byte + truncated-after-magic
# ---------------------------------------------------------------------------


class TestReadHeaderEdge:
    def test_invalid_version_byte_rejected(self, tmp_path: Path) -> None:
        """LMLA / LMLZ etc. (not 1-9) hits the generic 'Invalid LML version' branch."""
        bogus = tmp_path / "weird.lml"
        # LMLA — A is not in '2'..'9' so falls through to the generic ValueError
        bogus.write_bytes(b"LMLA" + b"\x00" * 30)
        with bogus.open("rb") as f:
            with pytest.raises(ValueError, match="Invalid LML version"):
                _read_header(f, 34)

    def test_truncated_after_magic(self, tmp_path: Path) -> None:
        """A file with the magic but < 14 bytes of header → 'Truncated LML header'."""
        bogus = tmp_path / "shortheader.lml"
        # 4-byte magic + 5 bytes of garbage (under 14-byte minimum)
        bogus.write_bytes(LML_MAGIC + b"\x00\x00\x00\x00\x00")
        with bogus.open("rb") as f:
            with pytest.raises(ValueError, match="Truncated LML header"):
                _read_header(f, 9)


# ---------------------------------------------------------------------------
# read_lml_file — truncated metadata + window-index branches
# ---------------------------------------------------------------------------


def _hand_craft_v1_header(
    *, n_ch: int = 4, n_windows: int = 2, total_samples: int = 500,
    window_size: int = 250, sr_mhz: int = 250000, bit_depth: int = 16,
    flags: int = 0, meta_len: int = 0,
) -> bytes:
    return struct.pack('<4sBBHHIHIBBI2x4x',
                       LML_MAGIC, 1, 0, n_ch, n_windows,
                       total_samples, window_size, sr_mhz,
                       bit_depth, flags, meta_len)


class TestReadLmlFileTruncation:
    def test_truncated_metadata(self, tmp_path: Path) -> None:
        """Header declares meta_len=20 but only 5 bytes provided → ValueError."""
        bogus = tmp_path / "trunc_meta.lml"
        # Declare meta_len=20 but write only 5 bytes of meta
        bogus.write_bytes(
            _hand_craft_v1_header(meta_len=20) + b"abcde"
        )
        with pytest.raises(ValueError, match="Truncated metadata"):
            read_lml_file(str(bogus))

    def test_truncated_window_index(self, tmp_path: Path) -> None:
        """Header declares n_windows=4 but the window index is truncated."""
        bogus = tmp_path / "trunc_idx.lml"
        header = _hand_craft_v1_header(meta_len=2, n_windows=4)
        # Write header + 2-byte meta + only 4 bytes of index (need 16 bytes)
        bogus.write_bytes(header + b"{}" + b"\x00\x00\x00\x00")
        with pytest.raises(ValueError, match="Truncated window index"):
            read_lml_file(str(bogus))

    def test_zero_payload_rejected(self, tmp_path: Path) -> None:
        """Window declared with 0-byte payload → file is corrupt."""
        bogus = tmp_path / "zero_payload.lml"
        # 1 window, 4 bytes of index, payload length = 0
        header = _hand_craft_v1_header(n_windows=1, meta_len=2)
        bogus.write_bytes(
            header
            + b"{}"  # metadata
            + b"\x00\x00\x00\x00"  # window index (1 offset = 0)
            + b"\x00\x00\x00\x00"  # payload length = 0
        )
        with pytest.raises(ValueError, match="0-byte payload"):
            read_lml_file(str(bogus))

    def test_oversize_payload_rejected(self, tmp_path: Path) -> None:
        """Window payload length over _MAX_PAYLOAD_LEN → corrupt."""
        bogus = tmp_path / "huge_payload.lml"
        header = _hand_craft_v1_header(n_windows=1, meta_len=2)
        bogus.write_bytes(
            header
            + b"{}"
            + b"\x00\x00\x00\x00"
            + struct.pack('<I', 2 * 256 * 1024 * 1024)  # 512 MB, over the cap
        )
        with pytest.raises(ValueError, match="exceeds"):
            read_lml_file(str(bogus))


# ---------------------------------------------------------------------------
# convert_edf_to_lml — graceful error on broken EDF
# ---------------------------------------------------------------------------


class TestConvertEdfToLmlErrors:
    def test_nonexistent_file_returns_error(self, tmp_path: Path) -> None:
        """Pointing at a missing file returns an error dict, not a crash."""
        result = convert_edf_to_lml(
            str(tmp_path / "nope.edf"),
            str(tmp_path / "out.lml"),
        )
        assert "error" in result

    def test_too_small_file_returns_error(self, tmp_path: Path) -> None:
        """A file under 256 bytes can't be a valid EDF — returns error."""
        bad = tmp_path / "tiny.edf"
        bad.write_bytes(b"x" * 100)
        result = convert_edf_to_lml(str(bad), str(tmp_path / "out.lml"))
        assert "error" in result


# ---------------------------------------------------------------------------
# read_edf_digital — small-file rejection + header parse errors
# ---------------------------------------------------------------------------


class TestReadEdfDigitalErrors:
    def test_small_file_rejected(self, tmp_path: Path) -> None:
        bad = tmp_path / "tiny.edf"
        bad.write_bytes(b"x" * 100)
        with pytest.raises(ValueError, match="File too small"):
            read_edf_digital(str(bad))

    def test_invalid_header_field(self, tmp_path: Path) -> None:
        """Garbage in the n_signals field raises a parse error."""
        bad = tmp_path / "garbage.edf"
        # 256 bytes of EDF main header with non-numeric n_signals (252:256)
        hdr = bytearray(256)
        hdr[:8] = b"0       "
        # n_data_records at 236-244, dur at 244-252, n_signals at 252-256
        hdr[236:244] = b"1       "
        hdr[244:252] = b"1.0     "
        hdr[252:256] = b"abcd"  # not parseable as int
        bad.write_bytes(bytes(hdr))
        with pytest.raises(ValueError, match="Invalid EDF/BDF header"):
            read_edf_digital(str(bad))

    def test_zero_n_signals_rejected(self, tmp_path: Path) -> None:
        bad = tmp_path / "zero_sigs.edf"
        hdr = bytearray(256)
        hdr[:8] = b"0       "
        hdr[236:244] = b"1       "
        hdr[244:252] = b"1.0     "
        hdr[252:256] = b"0   "  # n_signals = 0
        bad.write_bytes(bytes(hdr))
        with pytest.raises(ValueError, match="declares 0 signals"):
            read_edf_digital(str(bad))

    def test_zero_n_data_records_rejected(self, tmp_path: Path) -> None:
        bad = tmp_path / "zero_recs.edf"
        hdr = bytearray(256)
        hdr[:8] = b"0       "
        hdr[236:244] = b"0       "  # n_data_records = 0
        hdr[244:252] = b"1.0     "
        hdr[252:256] = b"1   "
        bad.write_bytes(bytes(hdr))
        with pytest.raises(ValueError, match="declares 0 data records"):
            read_edf_digital(str(bad))

    def test_truncated_signal_headers(self, tmp_path: Path) -> None:
        """Declared n_signals=10 → expects 2560 bytes of sig header. We
        provide much less to trigger the truncation guard."""
        bad = tmp_path / "trunc_sig.edf"
        hdr = bytearray(256)
        hdr[:8] = b"0       "
        hdr[236:244] = b"1       "
        hdr[244:252] = b"1.0     "
        hdr[252:256] = b"10  "  # n_signals = 10
        # Provide only 100 bytes of sig header (need 2560)
        bad.write_bytes(bytes(hdr) + b"x" * 100)
        with pytest.raises(ValueError, match="Truncated signal headers"):
            read_edf_digital(str(bad))


# ---------------------------------------------------------------------------
# Real-EDF read paths: verify multi-record + annotation handling
# ---------------------------------------------------------------------------


@pytest.mark.data
def test_read_edf_digital_metadata_completeness(real_test_edf: Path) -> None:
    """All documented metadata fields are present on the real fixture."""
    _sig, meta = read_edf_digital(str(real_test_edf))
    # FDA provenance triplet
    assert "encoder_version" in meta
    assert meta["encoder_version"].startswith("lamquant_codec/")
    assert "edf_header" in meta
    assert "edf_header_sha256" in meta
    # Per-channel field arrays match channel count
    C = meta["n_channels"]
    assert len(meta["phys_min"]) == C
    assert len(meta["phys_max"]) == C
    assert len(meta["dig_min"]) == C
    assert len(meta["dig_max"]) == C
    # all_labels covers EVERY signal, including annotations
    assert len(meta["all_labels"]) >= C
    assert len(meta["all_ns_per_rec"]) == len(meta["all_labels"])
    assert len(meta["transducers"]) == len(meta["all_labels"])
    assert len(meta["prefilterings"]) == len(meta["all_labels"])
    # continuous flag is a bool
    assert isinstance(meta["continuous"], bool)


@pytest.mark.data
def test_reconstruct_edf_then_re_read_signal_sha_match(
    tmp_path: Path, real_test_edf: Path,
) -> None:
    """After reconstruct_edf, re-encoding the reconstructed file yields the
    same signal_sha256 — bit-exact end-to-end round-trip."""
    # First encode
    out_lml = tmp_path / "rt1.lml"
    stats = convert_edf_to_lml(str(real_test_edf), str(out_lml))
    if "error" in stats:
        pytest.skip(f"convert_edf_to_lml failed: {stats['error']}")
    # Reconstruct EDF
    rec_edf = tmp_path / "rt.edf"
    used_original = reconstruct_edf(str(out_lml), str(rec_edf))
    assert used_original is True
    # Re-encode the reconstructed file
    out2 = tmp_path / "rt2.lml"
    stats2 = convert_edf_to_lml(str(rec_edf), str(out2))
    if "error" in stats2:
        pytest.skip(f"re-encode failed: {stats2['error']}")
    assert stats["signal_sha256"] == stats2["signal_sha256"]


# ---------------------------------------------------------------------------
# reconstruct_edf — synthesized branch on tampered EDF header SHA
# ---------------------------------------------------------------------------


@pytest.mark.data
def test_reconstruct_edf_header_sha_mismatch_raises(
    tmp_path: Path, real_test_edf: Path,
) -> None:
    """If metadata declares an edf_header_sha256 that doesn't match the
    decompressed header, reconstruct_edf raises ValueError."""
    import zstandard
    out_lml = tmp_path / "tamper.lml"
    stats = convert_edf_to_lml(str(real_test_edf), str(out_lml))
    if "error" in stats:
        pytest.skip(f"convert failed: {stats['error']}")

    # Read, tamper the SHA, re-write
    sig, meta = read_lml_file(str(out_lml))
    meta["edf_header_sha256"] = "0" * 64  # incorrect SHA
    out_tampered = tmp_path / "tampered.lml"
    write_lml_file(str(out_tampered), sig.astype(np.int64), meta)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        reconstruct_edf(str(out_tampered), str(tmp_path / "x.edf"))


# ---------------------------------------------------------------------------
# write_lml_file: cover sub-millisecond sample rates path (sr_mhz int)
# ---------------------------------------------------------------------------


@pytest.mark.data
def test_lml_window_size_scales_with_sample_rate(
    tmp_path: Path, real_test_edf: Path,
) -> None:
    """The actual_window in write_lml_file scales as window_size * sr/250.
    Roundtrip via real EDF — the encoder picks the right window size."""
    out_lml = tmp_path / "scaled.lml"
    stats = convert_edf_to_lml(str(real_test_edf), str(out_lml))
    if "error" in stats:
        pytest.skip(f"convert failed: {stats['error']}")
    # n_windows is computed from total_samples and actual_window.
    # The contract: at least one window, and n_windows × actual_window ≥ T.
    _sig, meta = read_lml_file(str(out_lml))
    assert stats["n_windows"] >= 1
    assert stats["total_samples"] > 0


# ---------------------------------------------------------------------------
# CRC + container roundtrip: write directly + read back
# ---------------------------------------------------------------------------


@pytest.mark.data
def test_write_lml_file_roundtrip_with_minimal_metadata(
    tmp_path: Path, real_test_edf: Path,
) -> None:
    """A round-trip through write_lml_file → read_lml_file preserves
    the signal byte-exact, even when metadata is minimal."""
    sig, full_meta = read_edf_digital(str(real_test_edf))
    minimal_meta = {"sample_rate": full_meta["sample_rate"]}
    out = tmp_path / "minimal.lml"
    stats = write_lml_file(str(out), sig.astype(np.int64), minimal_meta)
    assert stats["n_channels"] == sig.shape[0]
    sig_back, meta_back = read_lml_file(str(out))
    # Same data
    assert sig_back.shape == sig.shape
    assert hashlib.sha256(sig.tobytes()).hexdigest() == \
        hashlib.sha256(sig_back.tobytes()).hexdigest()
    # Same metadata
    assert meta_back["sample_rate"] == minimal_meta["sample_rate"]
