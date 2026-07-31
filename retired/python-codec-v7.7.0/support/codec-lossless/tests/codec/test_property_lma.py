"""Property-based tests — LMA archive invariants under random trees.

Hypothesis generates random file trees (1-8 files, mixed extensions,
nested directories) and asserts:

  P1. pack→unpack roundtrip — every input file appears at the same path
      with byte-exact content after unpack_lma.

  P2. Manifest entry SHA-256 — every entry's recorded sha256 equals an
      independent SHA of the input bytes.

  P3. Method dispatch — every entry's method matches _choose_method
      (modulo the secondary/zstd alias).

  P4. verify_lma — passes on a clean archive for any valid tree.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from lamquant_codec.lma import _choose_method, list_lma, pack_lma, unpack_lma, verify_lma

pytestmark = [pytest.mark.l4]


# ============================================================
# Strategies
# ============================================================


# A modest pool of file extensions to exercise every dispatch branch.
_EXTENSIONS = ["csv", "json", "md", "txt", "bin", "tse", "lbl",
               "zip", "gz", "zst", "png", "jpg"]

_filename_chars = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
    min_size=1, max_size=8,
)


@st.composite
def _random_tree(draw, min_files: int = 1, max_files: int = 6):
    """A non-empty mapping of {relative_path: bytes} for one archive."""
    n = draw(st.integers(min_value=min_files, max_value=max_files))
    files: dict[str, bytes] = {}
    used: set[str] = set()
    for _ in range(n):
        # Maybe nest 1-2 levels deep.
        depth = draw(st.integers(min_value=0, max_value=2))
        parts = [draw(_filename_chars) for _ in range(depth)]
        stem = draw(_filename_chars)
        ext = draw(st.sampled_from(_EXTENSIONS))
        rel = "/".join(parts + [f"{stem}.{ext}"]) if parts else f"{stem}.{ext}"
        if rel in used:
            continue
        used.add(rel)
        size = draw(st.integers(min_value=0, max_value=512))
        # Use deterministic bytes derived from path so failure repros are stable.
        seed = hashlib.sha256(rel.encode()).digest()
        files[rel] = (seed * (size // len(seed) + 1))[:size]
    return files


_HYPO = settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example,
                           HealthCheck.function_scoped_fixture],
)


def _write_tree(root: Path, files: dict[str, bytes]) -> None:
    for rel, data in files.items():
        full = root / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(data)


# ============================================================
# P1. pack→unpack roundtrip
# ============================================================


class TestPackUnpackRoundtrip:

    @_HYPO
    @given(files=_random_tree())
    def test_every_file_recovered_byte_exact(self, files, tmp_path_factory):
        if not files:
            return  # hypothesis sometimes generates empty maps; skip
        src = tmp_path_factory.mktemp("src")
        _write_tree(src, files)
        archive = tmp_path_factory.mktemp("ar") / "out.lma"
        pack_lma(str(src), str(archive))

        dst = tmp_path_factory.mktemp("dst")
        unpack_lma(str(archive), str(dst))

        for rel, expected in files.items():
            recovered = (dst / rel).read_bytes()
            assert recovered == expected, f"byte drift for {rel}"


# ============================================================
# P2. Manifest SHA matches independent SHA of input
# ============================================================


class TestManifestShaContract:

    @_HYPO
    @given(files=_random_tree(min_files=1, max_files=4))
    def test_manifest_sha256_matches_input(self, files, tmp_path_factory):
        if not files:
            return
        src = tmp_path_factory.mktemp("src")
        _write_tree(src, files)
        archive = tmp_path_factory.mktemp("ar") / "out.lma"
        pack_lma(str(src), str(archive))

        entries = list_lma(str(archive))
        for entry in entries:
            expected_sha = hashlib.sha256(files[entry["path"]]).hexdigest()
            assert entry["sha256"] == expected_sha, (
                f"{entry['path']}: manifest sha {entry['sha256']} "
                f"!= input sha {expected_sha}"
            )


# ============================================================
# P3. Method dispatch matches _choose_method
# ============================================================


class TestMethodDispatch:

    @_HYPO
    @given(files=_random_tree(min_files=1, max_files=4))
    def test_archived_method_matches_choose_method(self, files, tmp_path_factory):
        if not files:
            return
        src = tmp_path_factory.mktemp("src")
        _write_tree(src, files)
        archive = tmp_path_factory.mktemp("ar") / "out.lma"
        pack_lma(str(src), str(archive))

        entries = list_lma(str(archive))
        for entry in entries:
            expected = _choose_method(entry["path"])
            actual = entry["method"]
            if expected == "secondary":
                assert actual in ("secondary", "zstd"), (
                    f"{entry['path']}: expected secondary/zstd, got {actual}"
                )
            else:
                assert actual == expected, (
                    f"{entry['path']}: choose_method={expected} but "
                    f"archived as {actual}"
                )


# ============================================================
# P4. verify_lma always passes on clean archive
# ============================================================


class TestVerifyAlwaysPassesClean:

    @_HYPO
    @given(files=_random_tree(min_files=1, max_files=4))
    def test_clean_archive_verifies(self, files, tmp_path_factory):
        if not files:
            return
        src = tmp_path_factory.mktemp("src")
        _write_tree(src, files)
        archive = tmp_path_factory.mktemp("ar") / "out.lma"
        pack_lma(str(src), str(archive))
        assert verify_lma(str(archive), verbose=False) is True
