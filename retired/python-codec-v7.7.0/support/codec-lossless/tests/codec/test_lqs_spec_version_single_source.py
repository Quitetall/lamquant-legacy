"""Drift guard: the LQS_SPEC_VERSION copies must stay identical (backlog #37).

The LQS specification version ``"1.0"`` is hardcoded independently in three
places because they live in three different repos (git submodules) with no
shared import path:

  1. ``lamquant_codec.lqs.LQS_SPEC_VERSION`` (this repo — codec-lossless,
     the deprecated Python LQS mirror).
  2. ``LQS_SPEC_VERSION`` in ``evaluation/eagle-lqs/tools/lqs_task_concordance.py``
     (the task-concordance tool, a sibling submodule).
  3. ``SPEC_VERSION`` in ``evaluation/openecs/src/lib.rs`` (the Rust
     canonical crate, a third sibling submodule).

This test makes the unavoidable duplication safe: it fails the moment one
copy is bumped without the others — exactly the silent drift that motivates
``test_eeg_bands_single_source.py`` for ``EEG_BANDS``. Since two of the three
copies live in sibling submodules that are not guaranteed to be checked out
next to this one (each repo can be cloned/tested standalone), the test
degrades to a skip when a sibling copy can't be found on disk rather than
failing the whole suite.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from lamquant_codec import lqs as _lqs_module


def _load_eagle_lqs_concordance_module(path: Path):
    """Import ``lqs_task_concordance.py`` by path (it's a standalone tool
    script in a sibling submodule, not an installed package)."""
    spec = importlib.util.spec_from_file_location("_lqs_task_concordance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _extract_rust_spec_version(path: Path) -> str:
    """Pull ``SPEC_VERSION`` out of the Rust source text (no compiled
    bindings expose it to Python, so this reads the literal like
    ``tools/scripts/audit_format_consistency.py`` does for other
    cross-language constants)."""
    text = path.read_text()
    match = re.search(r'pub const SPEC_VERSION:\s*&str\s*=\s*"([^"]+)"', text)
    assert match, f"{path}: could not find `pub const SPEC_VERSION: &str = \"...\"`"
    return match.group(1)


def test_lqs_spec_version_matches_across_repos():
    codec_version = _lqs_module.LQS_SPEC_VERSION

    # Sibling submodules sit at ../../../evaluation/* relative to this test
    # file (codec-lossless/tests/codec/ -> meta-repo root) when checked out
    # as part of the LamQuant meta-repo. Standalone codec-lossless checkouts
    # won't have them — skip rather than fail in that case.
    repo_root = Path(__file__).resolve().parents[3]
    concordance_path = repo_root / "evaluation" / "eagle-lqs" / "tools" / "lqs_task_concordance.py"
    openecs_lib_path = repo_root / "evaluation" / "openecs" / "src" / "lib.rs"

    if not concordance_path.exists() or not openecs_lib_path.exists():
        pytest.skip(
            "sibling evaluation/ submodules not checked out next to "
            "codec-lossless — can't cross-check LQS_SPEC_VERSION here "
            f"(looked for {concordance_path} and {openecs_lib_path})"
        )

    concordance_module = _load_eagle_lqs_concordance_module(concordance_path)
    concordance_version = concordance_module.LQS_SPEC_VERSION

    rust_version = _extract_rust_spec_version(openecs_lib_path)

    assert codec_version == concordance_version == rust_version, (
        "LQS_SPEC_VERSION drifted between "
        "lamquant_codec.lqs "
        f"({codec_version!r}), "
        "evaluation/eagle-lqs/tools/lqs_task_concordance.py "
        f"({concordance_version!r}), and "
        "evaluation/openecs/src/lib.rs SPEC_VERSION "
        f"({rust_version!r}) — update all three (single source of truth, "
        "backlog #37)."
    )
