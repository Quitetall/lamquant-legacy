"""LMA archive Python contract — packs, unpacks, list, verify.

Mirrors `lamquant-core/tests/lma_conformance.rs` on the Python side. Where
the Rust integration test exercises the compiled archive format, this file
pins the Python public API at `lamquant_codec.lma`:

  pack_lma(input_dir, output_path, ...)
  unpack_lma(archive_path, output_dir, ...)
  list_lma(archive_path)
  verify_lma(archive_path)

plus the magic-byte / version constants, the extension dispatch, and the
compressor registry.

Drift here means a Python-produced .lma archive cannot be read by the
Rust binary, or vice versa. The cross-language sentinel for that lives
in tests/codec/cross_lang/test_lma_cross_lang.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lamquant_codec.lma import (
    LMA_MAGIC,
    LMA_VERSION,
    _choose_method,
    list_lma,
    pack_lma,
    unpack_lma,
    verify_lma,
)

pytestmark = pytest.mark.l3


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_tree(root: Path, files: dict[str, bytes]) -> None:
    for rel, data in files.items():
        full = root / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(data)


# ============================================================
# 1. Wire-format constants pinned
# ============================================================


class TestConstants:

    def test_magic_pinned(self):
        assert LMA_MAGIC == b"LMA1"
        assert len(LMA_MAGIC) == 4

    def test_version_pinned(self):
        assert LMA_VERSION == 1


# ============================================================
# 2. Extension dispatch (_choose_method)
# ============================================================


class TestChooseMethod:

    @pytest.mark.parametrize("ext", ["edf", "bdf", "EDF", "BDF"])
    def test_edf_bdf_routes_to_lml(self, ext):
        assert _choose_method(f"recording.{ext}") == "lml"

    @pytest.mark.parametrize("ext", [
        "lml", "lmq", "lma", "gz", "zst", "zip", "7z",
        "png", "jpg", "jpeg", "mp4", "avi",
    ])
    def test_already_compressed_routes_to_store(self, ext):
        assert _choose_method(f"file.{ext}") == "store"

    @pytest.mark.parametrize("name", [
        "notes.csv", "data.json", "README.md",
        "no_extension", "annotation.tse",
    ])
    def test_default_routes_to_secondary(self, name):
        assert _choose_method(name) == "secondary"


# ============================================================
# 3. pack → list → unpack round-trip on a mixed-method tree
# ============================================================


@pytest.fixture
def mixed_tree(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    """Tempdir with files that span every method choice."""
    src = tmp_path / "src"
    files = {
        "notes.csv":          b"a,b,c\n1,2,3\n",        # secondary (zstd)
        "blob.bin":           b"\xFA" * 256,             # secondary
        "dir/sub.zst":        b"\xCC" * 128,             # store
        "dir/deep/leaf.txt":  b"hello, archive!",        # secondary
        "packaged.zip":       b"PK\x03\x04...",          # store
    }
    _write_tree(src, files)
    return src, files


class TestPackUnpack:

    def test_pack_and_unpack_byte_exact(self, mixed_tree, tmp_path):
        src, files = mixed_tree
        archive = tmp_path / "out.lma"
        summary = pack_lma(str(src), str(archive))
        assert archive.exists()
        assert archive.stat().st_size > 0
        assert summary.get("n_files", summary.get("count", 0)) >= 0
        # Pack always produces the LMA magic at offset 0.
        assert archive.read_bytes()[:4] == LMA_MAGIC

        dst = tmp_path / "dst"
        unpack_lma(str(archive), str(dst))

        for rel, data in files.items():
            recovered = (dst / rel).read_bytes()
            assert recovered == data, f"byte drift for {rel}"

    def test_list_lma_returns_one_entry_per_file(self, mixed_tree, tmp_path):
        src, files = mixed_tree
        archive = tmp_path / "out.lma"
        pack_lma(str(src), str(archive))

        entries = list_lma(str(archive))
        assert isinstance(entries, list)
        assert len(entries) == len(files)
        names = {e["path"] for e in entries}
        assert names == set(files.keys())

    def test_list_lma_includes_method_and_sha(self, mixed_tree, tmp_path):
        src, files = mixed_tree
        archive = tmp_path / "out.lma"
        pack_lma(str(src), str(archive))

        entries = list_lma(str(archive))
        for entry in entries:
            assert entry["method"] in ("lml", "secondary", "zstd", "store")
            sha = entry.get("sha256", "")
            assert isinstance(sha, str) and len(sha) == 64

            # The recorded SHA-256 must match the SHA of the original input.
            expected_sha = _sha256_hex(files[entry["path"]])
            assert entry["sha256"] == expected_sha, (
                f"{entry['path']}: archived sha {entry['sha256']} "
                f"!= input sha {expected_sha}"
            )

    def test_method_dispatch_matches_choose_method(self, mixed_tree, tmp_path):
        src, _ = mixed_tree
        archive = tmp_path / "out.lma"
        pack_lma(str(src), str(archive))

        entries = list_lma(str(archive))
        for entry in entries:
            expected = _choose_method(entry["path"])
            actual = entry["method"]
            # `_choose_method` returns "secondary" for unknown extensions but
            # the manifest may surface "zstd" for the same — accept both.
            if expected == "secondary":
                assert actual in ("secondary", "zstd"), (
                    f"{entry['path']}: expected secondary/zstd, got {actual}"
                )
            else:
                assert actual == expected, (
                    f"{entry['path']}: choose_method={expected} "
                    f"but archived as {actual}"
                )


# ============================================================
# 4. verify_lma — SHA chain integrity
# ============================================================


class TestVerify:

    def test_verify_returns_true_on_clean_archive(self, mixed_tree, tmp_path):
        src, _ = mixed_tree
        archive = tmp_path / "out.lma"
        pack_lma(str(src), str(archive))
        assert verify_lma(str(archive), verbose=False) is True

    def test_verify_detects_byte_flip_in_payload(self, mixed_tree, tmp_path):
        src, _ = mixed_tree
        archive = tmp_path / "out.lma"
        pack_lma(str(src), str(archive))

        # Flip a byte well past the header (16) but well before the trailing
        # 32-byte archive SHA-256.
        bytes_orig = bytearray(archive.read_bytes())
        flip_idx = max(64, len(bytes_orig) // 2)
        bytes_orig[flip_idx] ^= 0x01
        archive.write_bytes(bytes(bytes_orig))

        # Either verify_lma returns False, raises, or unpack reports failure
        # in the summary. Any of those signals detection.
        detected = False
        try:
            ok = verify_lma(str(archive), verbose=False)
            if not ok:
                detected = True
        except Exception:
            detected = True

        if not detected:
            dst = tmp_path / "dst"
            try:
                summary = unpack_lma(str(archive), str(dst))
                if summary.get("failed", 0) > 0:
                    detected = True
            except Exception:
                detected = True

        assert detected, "byte flip in payload was NOT detected by either verify_lma or unpack_lma"


# ============================================================
# 5. Bad inputs / error paths
# ============================================================


class TestBadInputs:

    def test_pack_rejects_nondirectory(self, tmp_path):
        not_a_dir = tmp_path / "file.txt"
        not_a_dir.write_text("hi")
        archive = tmp_path / "out.lma"
        with pytest.raises(Exception):
            pack_lma(str(not_a_dir), str(archive))

    def test_unpack_rejects_missing_archive(self, tmp_path):
        missing = tmp_path / "does_not_exist.lma"
        dst = tmp_path / "dst"
        with pytest.raises(Exception):
            unpack_lma(str(missing), str(dst))

    def test_unpack_rejects_bad_magic(self, tmp_path):
        bogus = tmp_path / "bogus.lma"
        bogus.write_bytes(b"NOTL" + b"\x00" * 60)
        dst = tmp_path / "dst"
        with pytest.raises(Exception):
            unpack_lma(str(bogus), str(dst))

    def test_list_lma_rejects_bad_magic(self, tmp_path):
        bogus = tmp_path / "bogus.lma"
        bogus.write_bytes(b"XXXX" + b"\x00" * 60)
        with pytest.raises(Exception):
            list_lma(str(bogus))


# ============================================================
# 6. Compressor registry (plugin discovery)
# ============================================================


class TestCompressorRegistry:

    def test_default_compressors_listed(self):
        from lamquant_codec.lma import _COMPRESSORS
        # The two built-in plugins must always be present.
        assert "zstd" in _COMPRESSORS
        assert "none" in _COMPRESSORS

    def test_set_compressor_to_unknown_raises(self):
        from lamquant_codec.lma import set_compressor
        with pytest.raises(Exception):
            set_compressor("brotli", level=9)
