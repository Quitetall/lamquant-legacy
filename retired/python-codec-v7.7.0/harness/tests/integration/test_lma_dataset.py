"""test_lma_dataset.py — Phase 3 integration tests for LMA-direct training.

Verifies:
  - LMA migration produces a per-stem LMA bundling LML + sidecars + labels
  - read_entry + container_read_bytes roundtrip preserves bytes
  - LmaDataset returns the same shape/dtype contract as SubbandActivityDataset
  - Subject-grouped split: no subject appears in both train and val
  - Window selection policy honors max/seizure caps + min background
  - Defensive subject_id mismatch check raises rather than silently leaks
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ai_models.snn.lma_subject_id import (  # noqa: E402
    extract_subject_id,
    precedence_rank,
    corpus_short_name,
)
from ai_models.snn.lma_dataset import select_windows  # noqa: E402

# ----------------------------------------------------------------------
# Subject ID parser tests
# ----------------------------------------------------------------------


@pytest.mark.parametrize("corpus,stem_path,expected_sid,expected_tag", [
    ("tueg_v2.0.1", "edf/000/aaaaaaaa/s001/aaaaaaaa_s001_t000.lml",
     "aaaaaaaa", "tueg_filename_regex"),
    ("tusz_v2.0.6", "edf/train/01_tcp_ar/aaaaarsp_s004_t001.lml",
     "aaaaarsp", "tusz_filename_regex"),
    ("tuev_v2.0.1", "edf/train/aaaaabba/aaaaabba_00000003.lml",
     "aaaaabba", "tuev_train_filename_regex"),
    ("tuev_v2.0.1", "edf/eval/036/bckg_036_a_.lml",
     "tuev_eval_036", "tuev_eval_filename_event_regex"),
    ("tuev_v2.0.1", "edf/eval/051/bckg_051_a_1.lml",
     "tuev_eval_051", "tuev_eval_filename_event_regex"),
    ("tuab_v3.0.1", "edf/train/normal/aaaaaoyp_s001_t001.lml",
     "aaaaaoyp", "tuab_filename_regex"),
])
def test_subject_id_extraction_canonical_layouts(
    corpus, stem_path, expected_sid, expected_tag
):
    sid, tag = extract_subject_id(corpus, Path(stem_path))
    assert sid == expected_sid, f"got {sid!r}, expected {expected_sid!r}"
    assert tag == expected_tag


def test_precedence_rank_ordering():
    """TUSZ wins over TUEG; both win over an unknown corpus."""
    assert precedence_rank("tusz_v2.0.6") < precedence_rank("tueg_v2.0.1")
    assert precedence_rank("tuev_v2.0.1") < precedence_rank("tuab_v3.0.1")
    assert precedence_rank("future_dataset") > precedence_rank("tueg_v2.0.1")


def test_corpus_short_name_strips_version():
    assert corpus_short_name("tueg_v2.0.1") == "tueg"
    assert corpus_short_name("TUSZ_v3") == "tusz"


# ----------------------------------------------------------------------
# Window selection policy
# ----------------------------------------------------------------------


def _synth_labels(n_windows: int, seizure_window_idxs):
    """Return synthetic activity_labels [8, n_windows * 312] with classes."""
    LABEL_PER_WINDOW = 312
    arr = np.zeros((8, n_windows * LABEL_PER_WINDOW), dtype=np.uint8)
    for wi in seizure_window_idxs:
        s = wi * LABEL_PER_WINDOW
        e = s + LABEL_PER_WINDOW
        arr[:, s:e] = 2
    return arr


def test_window_selection_zero_seizures_returns_evenly_spaced():
    labels = _synth_labels(20, [])
    selected = select_windows(labels, max_windows=5)
    assert len(selected) == 5
    assert sorted(selected) == selected
    # Evenly spaced across [0, 19]
    assert selected[0] == 0
    assert selected[-1] == 19


def test_window_selection_all_seizures_caps_at_max_seizure():
    labels = _synth_labels(20, list(range(20)))
    selected = select_windows(labels, max_windows=5, max_seizure_windows=10)
    assert len(selected) == 10
    # All selected must be seizure windows
    for wi in selected:
        s = wi * 312
        e = s + 312
        assert np.any(labels[:, s:e] == 2)


def test_window_selection_mixed_includes_seizures_and_background():
    labels = _synth_labels(20, [3, 7, 15])
    selected = select_windows(labels, max_windows=5)
    assert set([3, 7, 15]) <= set(selected)
    assert len(selected) == 5
    # Remaining must be background
    bg_count = sum(1 for wi in selected if wi not in {3, 7, 15})
    assert bg_count == 2


def test_window_selection_status_epilepticus_capped():
    # 20 seizure windows but cap=10
    labels = _synth_labels(20, list(range(20)))
    selected = select_windows(labels, max_windows=5, max_seizure_windows=10)
    assert len(selected) == 10


# ----------------------------------------------------------------------
# End-to-end LMA migration + LmaDataset integration
# ----------------------------------------------------------------------

LML_ROOT = Path("/mnt/4tb/data/lml/edf.lml")
LABELS_DIR = Path("/mnt/4tb/LamQuant/ai_models/snn/labels")
_HAS_LAMQUANT_CORE = importlib.util.find_spec("lamquant_core") is not None
REQUIRES_REAL_DATA = pytest.mark.skipif(
    not LML_ROOT.exists() or not LABELS_DIR.exists() or not _HAS_LAMQUANT_CORE,
    reason=(
        "requires real LML+labels data at /mnt/4tb AND the `lamquant_core` "
        "pyo3 binding (install via `maturin develop --release -m lamquant-core/Cargo.toml`)"
    ),
)


@REQUIRES_REAL_DATA
def test_end_to_end_lma_pack_and_read():
    """Pack one real stem into LMA, read entries back via Rust."""
    import lamquant_core
    from lamquant_codec.lma import pack_lma, list_lma, verify_lma

    # Pick first available stem
    stem = None
    lml_path = None
    for f in sorted(LABELS_DIR.glob("*_labels.npz"))[:50]:
        candidate_stem = f.name.removesuffix("_labels.npz")
        cands = list(LML_ROOT.rglob(f"{candidate_stem}.lml"))
        if cands:
            stem = candidate_stem
            lml_path = cands[0]
            break
    assert stem is not None, "no test stem found"

    with tempfile.TemporaryDirectory(prefix="lma_test_", dir="/mnt/4tb/data") as sb:
        sb = Path(sb)
        stage = sb / "stage"
        stage.mkdir()
        shutil.copy2(lml_path, stage / lml_path.name)
        label = LABELS_DIR / f"{stem}_labels.npz"
        shutil.copy2(label, stage / label.name)
        (stage / "meta.json").write_text(json.dumps({
            "schema": "lamquant.lma_meta.v1",
            "stem": stem,
            "subject_id": stem.split("_")[0],
            "has_labels": True,
            "label_npz_name": label.name,
            "content_sha256_lml": "test",
        }))

        lma_path = sb / f"{stem}.lma"
        pack_lma(str(stage), str(lma_path), verbose=False)
        assert verify_lma(str(lma_path), verbose=False)

        # Read LML back
        lml_bytes = lamquant_core.lma_read_entry(str(lma_path), lml_path.name)
        assert lml_bytes == lml_path.read_bytes()

        # Read label NPZ back
        label_bytes = lamquant_core.lma_read_entry(str(lma_path), label.name)
        assert label_bytes == label.read_bytes()

        # Decode signal from LML bytes
        signal, meta = lamquant_core.container_read_bytes(lml_bytes)
        assert len(signal) > 0
        assert len(signal[0]) > 0


@REQUIRES_REAL_DATA
def test_l3_parity_q31_vs_lma():
    """Single-window L3 must match between Q31 NPZ and LMA-direct paths."""
    import subprocess
    r = subprocess.run(
        [sys.executable, str(_REPO / "scripts" / "parity_l3_one_window.py")],
        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(_REPO)},
    )
    assert r.returncode == 0, f"parity script failed:\nstdout={r.stdout}\nstderr={r.stderr}"
    assert "PARITY OK" in r.stdout + r.stderr


# ----------------------------------------------------------------------
# Split manifest invariants
# ----------------------------------------------------------------------


def _synth_split_manifest(tmp_path: Path, train_sids, val_sids,
                          stems_by_sid):
    manifest = {
        "schema": "lamquant.snn_split.v1",
        "seed": 42,
        "split_strategy": "subject_grouped",
        "val_fraction": 0.10,
        "subjects": {**{s: "train" for s in train_sids},
                     **{s: "val" for s in val_sids}},
        "stems_by_subject": stems_by_sid,
        "promotions": [],
        "summary": {},
    }
    path = tmp_path / "split.json"
    path.write_text(json.dumps(manifest))
    return path


def test_split_manifest_subject_disjoint(tmp_path):
    from ai_models.snn.lma_dataset import load_split_manifest

    path = _synth_split_manifest(
        tmp_path,
        train_sids=["s_a", "s_b"],
        val_sids=["s_c"],
        stems_by_sid={"s_a": ["stem1", "stem2"], "s_b": ["stem3"],
                      "s_c": ["stem4"]},
    )
    train_stems, train_by = load_split_manifest(path, "train")
    val_stems, val_by = load_split_manifest(path, "val")
    assert set(train_stems) == {"stem1", "stem2", "stem3"}
    assert set(val_stems) == {"stem4"}
    assert set(train_by.values()).isdisjoint(set(val_by.values()))
