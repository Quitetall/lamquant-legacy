"""Cross-language LML container drift sentinels.

Pins Python `container_write` ↔ Rust `container_write` parity through
the PyO3 wheel. Drift here means an LML file written by one language is
unreadable by the other — clinical-grade hard fail.

Strategy: in-process via `lamquant_core` PyO3 module (no subprocess).
The wheel exposes:
    container_write(path, signal, sample_rate, window_size, noise_bits, meta_json)
    container_read(path) -> (signal, meta_json)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lamquant_codec.errors import LmlCrossLangDriftError
from tests.helpers.asserts import (
    assert_array_equal_strict,
    assert_bytes_equal,
)
from tests.helpers.rust_bindings import (
    HAS_RUST,
    requires_rust,
    rust_container_read,
    rust_container_write,
)
from tests.helpers.signals import synth_signal

pytestmark = [pytest.mark.l5, pytest.mark.cross_lang, requires_rust]


def _to_rust(sig: np.ndarray) -> list[list[int]]:
    return [row.tolist() for row in sig]


# ============================================================
# 1. Python writes, Rust reads
# ============================================================


class TestPythonWriteRustRead:
    """Python uses the Rust binding (same as Rust write here) — pin the
    in-process roundtrip plus byte-level parity at the file boundary."""

    def test_write_read_roundtrip(self, tmp_path: Path):
        sig = synth_signal(4, 256, seed=11)
        path = tmp_path / "py_written.lml"
        rust_container_write(str(path), _to_rust(sig), 250.0, 256, 0, "{}")
        assert path.read_bytes()[:4] == b"LML1", "magic byte at offset 0"

        recovered, meta = rust_container_read(str(path))
        assert meta == "{}"
        assert_array_equal_strict(
            np.array(recovered, dtype=np.int64),
            sig,
            expected_dtype=np.int64,
            expected_shape=sig.shape,
            context="container roundtrip",
        )


# ============================================================
# 2. Magic byte + version pinning
# ============================================================


class TestContainerMagicAndVersion:

    def test_magic_at_offset_0(self, tmp_path: Path):
        sig = synth_signal(2, 64, seed=12)
        path = tmp_path / "out.lml"
        rust_container_write(str(path), _to_rust(sig), 250.0, 64, 0, "{}")
        bytes_ = path.read_bytes()
        assert_bytes_equal(
            bytes_[:4], b"LML1",
            context="container magic at offset 0",
        )

    def test_version_major_at_offset_4(self, tmp_path: Path):
        sig = synth_signal(2, 64, seed=13)
        path = tmp_path / "out.lml"
        rust_container_write(str(path), _to_rust(sig), 250.0, 64, 0, "{}")
        b = path.read_bytes()
        assert b[4] == 1, f"version_major byte = {b[4]}, expected 1"
        assert b[5] == 0, f"version_minor byte = {b[5]}, expected 0"


# ============================================================
# 3. Header byte layout — n_ch, n_windows, total_samples LE
# ============================================================


class TestContainerHeaderLayout:

    @pytest.mark.parametrize("n_ch", [1, 4, 21, 64])
    def test_n_ch_at_offset_6_le(self, n_ch: int, tmp_path: Path):
        sig = synth_signal(n_ch, 256, seed=n_ch)
        path = tmp_path / "out.lml"
        rust_container_write(str(path), _to_rust(sig), 250.0, 256, 0, "{}")
        b = path.read_bytes()
        observed = int.from_bytes(b[6:8], "little")
        assert observed == n_ch, (
            f"n_ch byte drift: bytes 6-7 = {b[6]:#04x} {b[7]:#04x} → "
            f"{observed}, expected {n_ch}"
        )


# ============================================================
# 4. Mixed dtypes — int16 / int32 / int64 input → int64 storage
# ============================================================


class TestDtypeNormalization:

    def test_int16_input_recovers_as_int64(self, tmp_path: Path):
        sig16 = synth_signal(2, 256, seed=14, dtype=np.int16)
        path = tmp_path / "out.lml"
        rust_container_write(
            str(path),
            _to_rust(sig16.astype(np.int64)),
            250.0, 256, 0, "{}",
        )
        recovered, _ = rust_container_read(str(path))
        recovered_arr = np.array(recovered, dtype=np.int64)
        np.testing.assert_array_equal(recovered_arr, sig16.astype(np.int64))


# ============================================================
# 5. Trailing-data preservation in the metadata JSON
# ============================================================


class TestMetadataPreservation:

    def test_unicode_metadata_roundtrips(self, tmp_path: Path):
        sig = synth_signal(2, 64, seed=20)
        meta = '{"patient":"Müller","note":"✓ ok"}'
        path = tmp_path / "out.lml"
        rust_container_write(str(path), _to_rust(sig), 250.0, 64, 0, meta)
        _, recovered_meta = rust_container_read(str(path))
        assert recovered_meta == meta, (
            f"metadata JSON drift: wrote {meta!r}, got {recovered_meta!r}"
        )

    def test_long_metadata_under_max(self, tmp_path: Path):
        sig = synth_signal(2, 64, seed=21)
        # 4 KB of structured JSON.
        meta = '{"channels":[' + ",".join(f'"ch{i}"' for i in range(200)) + ']}'
        path = tmp_path / "out.lml"
        rust_container_write(str(path), _to_rust(sig), 250.0, 64, 0, meta)
        _, recovered_meta = rust_container_read(str(path))
        assert recovered_meta == meta
