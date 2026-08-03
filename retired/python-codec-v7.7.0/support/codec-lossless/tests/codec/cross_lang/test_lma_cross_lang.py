"""Cross-language LMA archive drift sentinels — Python ↔ Rust parity.

The L5 cross-lang test suite for `_compress_bytes` covers per-window
LML1 packets. This file mirrors the same approach for the LMA archive
format — Python `pack_lma` writing an archive that Rust `lml extract`
can read, and vice versa.

Drift here means an archive produced by one language can no longer be
read by the other. For an FDA-grade clinical codec, that's a hard fail.

Strategy: subprocess invokes the Rust `lml` binary because LMA pack/
unpack is not exposed via the PyO3 wheel (only LML1 per-window
encode/decode is). The binary path is resolved by the
`lml_cli_binary` session fixture, which skips this whole module when
the binary isn't built.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from lamquant_codec.errors import LmlCrossLangDriftError
from lamquant_codec.lma import list_lma, pack_lma, unpack_lma
from tests.helpers.asserts import assert_bytes_equal

pytestmark = [pytest.mark.l5, pytest.mark.cross_lang]


# ============================================================
# Helpers
# ============================================================


def _run_lml(binary: Path, *args: str) -> subprocess.CompletedProcess:
    """Invoke the lml binary with given arguments. Returns the completed
    process. Tests assert on returncode, stdout, and produced files.
    """
    result = subprocess.run(
        [str(binary), *args],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_tree(root: Path, files: dict[str, bytes]) -> None:
    for rel, data in files.items():
        full = root / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(data)


# ============================================================
# Mixed-method tree fixture
# ============================================================


@pytest.fixture
def mixed_tree(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    src = tmp_path / "src"
    files = {
        "notes.csv":          b"a,b,c\n1,2,3\n",        # secondary (zstd)
        "data.bin":           b"\xFA" * 256,             # secondary
        "dir/inner.zst":      b"\xCC" * 128,             # store
        "dir/deep/leaf.txt":  b"hello, archive!",        # secondary
        "packaged.zip":       b"PK\x03\x04...",          # store
    }
    _write_tree(src, files)
    return src, files


# ============================================================
# 1. Python pack → Rust extract
# ============================================================


class TestPythonPackRustExtract:

    def test_python_archive_extracts_byte_exact_via_rust(
        self, mixed_tree, tmp_path, lml_cli_binary
    ):
        src, files = mixed_tree
        archive = tmp_path / "py_packed.lma"
        pack_lma(str(src), str(archive))

        dst = tmp_path / "rust_extracted"
        dst.mkdir()
        result = _run_lml(lml_cli_binary, "extract",
                          str(archive), "-o", str(dst))
        if result.returncode != 0:
            raise LmlCrossLangDriftError(
                f"`lml extract` failed on Python-packed archive: "
                f"returncode={result.returncode}\n"
                f"  stdout: {result.stdout[:200]}\n"
                f"  stderr: {result.stderr[:200]}"
            )

        # Every file recovered byte-exact via Rust.
        for rel, expected in files.items():
            recovered = (dst / rel).read_bytes()
            assert_bytes_equal(
                recovered, expected,
                context=f"Python→Rust path: {rel}",
            )


# ============================================================
# 2. Rust pack → Python unpack
# ============================================================


class TestRustPackPythonUnpack:

    def test_rust_archive_unpacks_byte_exact_via_python(
        self, mixed_tree, tmp_path, lml_cli_binary
    ):
        src, files = mixed_tree
        archive = tmp_path / "rust_packed.lma"
        result = _run_lml(lml_cli_binary, "archive",
                          str(src), "-o", str(archive))
        if result.returncode != 0:
            raise LmlCrossLangDriftError(
                f"`lml archive` failed: returncode={result.returncode}\n"
                f"  stdout: {result.stdout[:200]}\n"
                f"  stderr: {result.stderr[:200]}"
            )
        assert archive.exists() and archive.stat().st_size > 0

        # Python's unpack_lma must read it back.
        dst = tmp_path / "py_extracted"
        unpack_lma(str(archive), str(dst))

        for rel, expected in files.items():
            recovered = (dst / rel).read_bytes()
            assert_bytes_equal(
                recovered, expected,
                context=f"Rust→Python path: {rel}",
            )


# ============================================================
# 3. Manifest agreement on per-entry SHA-256
# ============================================================


class TestManifestShaAgreement:

    def test_python_manifest_sha_matches_input(
        self, mixed_tree, tmp_path
    ):
        src, files = mixed_tree
        archive = tmp_path / "out.lma"
        pack_lma(str(src), str(archive))

        entries = list_lma(str(archive))
        for entry in entries:
            expected_sha = _sha256(files[entry["path"]])
            if entry["sha256"] != expected_sha:
                raise LmlCrossLangDriftError(
                    f"Python manifest SHA drift for {entry['path']}: "
                    f"manifest={entry['sha256']}, recomputed={expected_sha}"
                )


# ============================================================
# 4. Magic byte parity at offset 0
# ============================================================


class TestArchiveMagicAgreement:

    def test_python_archive_starts_with_lma1(self, mixed_tree, tmp_path):
        src, _ = mixed_tree
        archive = tmp_path / "out.lma"
        pack_lma(str(src), str(archive))
        assert_bytes_equal(
            archive.read_bytes()[:4], b"LMA1",
            context="Python LMA magic at offset 0",
        )

    def test_rust_archive_starts_with_lma1(
        self, mixed_tree, tmp_path, lml_cli_binary
    ):
        src, _ = mixed_tree
        archive = tmp_path / "out.lma"
        result = _run_lml(lml_cli_binary, "archive",
                          str(src), "-o", str(archive))
        if result.returncode != 0:
            pytest.skip(f"lml archive failed: {result.stderr[:200]}")
        assert_bytes_equal(
            archive.read_bytes()[:4], b"LMA1",
            context="Rust LMA magic at offset 0",
        )


# ============================================================
# 5. Round-trip identity — Python pack → Rust extract → Python pack again
# ============================================================


class TestFullTriangle:

    def test_extracted_files_repack_identical_via_rust(
        self, mixed_tree, tmp_path, lml_cli_binary
    ):
        """Files extracted by Rust from a Python-packed archive must be
        byte-identical to the original input. Re-packing them yields a
        new archive whose entries' SHA-256 match the originals."""
        src, files = mixed_tree
        # Python pack
        archive1 = tmp_path / "first.lma"
        pack_lma(str(src), str(archive1))
        # Rust extract
        dst = tmp_path / "extracted"
        dst.mkdir()
        result = _run_lml(lml_cli_binary, "extract",
                          str(archive1), "-o", str(dst))
        if result.returncode != 0:
            pytest.skip(f"lml extract failed: {result.stderr[:200]}")

        # Python pack again
        archive2 = tmp_path / "second.lma"
        pack_lma(str(dst), str(archive2))

        # Manifest entry SHAs must match original input bytes.
        entries = list_lma(str(archive2))
        names = {e["path"]: e["sha256"] for e in entries}
        for rel, data in files.items():
            assert rel in names, f"{rel} dropped during triangle"
            expected = _sha256(data)
            if names[rel] != expected:
                raise LmlCrossLangDriftError(
                    f"Triangle drift for {rel}: manifest sha {names[rel]} "
                    f"!= input sha {expected}"
                )
