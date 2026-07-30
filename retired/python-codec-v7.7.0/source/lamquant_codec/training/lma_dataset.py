"""lma_dataset.py — canonical LMA-direct training datasets for the codec.

Reads per-recording ``.lma`` archives via the Rust PyO3 extension:
  - ``lamquant_core.lma_read_entry(lma_path, '<stem>.lml')`` -> container bytes
  - ``lamquant_core.container_read_bytes(bytes)`` -> (signal int64, metadata)

then runs the same preprocessing pipeline as ``convert_lml``
(``ai_models/dataset_sim/preprocess.py``) so per-window L3 is bit-exact
to the deprecated Q31 NPZ + precompute_l3_fast path:

    digital -> microvolts -> 21ch select -> resample 250 Hz
    -> highpass 0.5 Hz -> Q31 normalize -> de-quantize
    -> preprocess_subband_single

This file is the codec/teacher/decoder counterpart to
``ai_models/snn/lma_dataset.py`` (which carries seizure-aware label
selection on top of the same decode pipeline). The shared helpers
``_decode_and_preprocess`` and ``_cached_signal`` live here; the SNN
variant should eventually delegate to this module.

Public Datasets:
    LmaSignalDataset
        Random-access raw fullband windows ``[21, 2500] float32``. Drop-in
        replacement for ``StreamingQ31Dataset`` / ``HybridQ31Dataset`` /
        ``MemmapTeacherDataset`` / ``Q31Dataset``.

    LmaL3Dataset
        Random-access L3 windows ``[21, 313] float32`` plus the legacy
        ``(l3, l3, dummy_mask)`` 3-tuple contract that
        ``PrecomputedL3Dataset`` exposed. Drop-in replacement for the
        precompute path.

Caching: per-worker LRU on the fully-processed float32 signal (one
entry per LMA, default cap 2). Multiple windows from the same LMA reuse
the decode + preprocess work. L3 itself is not cached — fast enough
given the cached signal.

See ADR 0017.
"""
from __future__ import annotations

import collections
import io
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

LOG = logging.getLogger(__name__)

# Data-file extensions that carry an LML-encoded recording in an `lml archive`
# corpus LMA (sidecars like .csv/.tse/.npz are NOT recordings). Matched
# case-insensitively. All current corpora are EDF; extend here if a BDF/other
# EDF-family format is ever ingested.
_RECORDING_EXTS = (".edf", ".bdf")

# ----------------------------------------------------------------------
# Shape constants — must match the deprecated NPZ + L3 pipeline so swap
# is bit-exact (verified against 250-file sample, zero mismatches).
# ----------------------------------------------------------------------

WINDOW_SAMPLES = 2500       # 10 s @ 250 Hz
TARGET_SR = 250.0
TARGET_CHANNELS = 21
L3_T = 313                  # preprocess_subband_single time dim
Q31_HEADROOM = 0.72         # convert_lml default
HIGHPASS_HZ = 0.5
SIGNAL_CACHE_CAP = 2        # ~270 MB per 3.6 hr TUEG signal -> 540 MB / worker


# ----------------------------------------------------------------------
# Lazy import — keep `import lma_dataset` cheap. PyTorch DataLoader
# workers each pay the cost once on first __getitem__.
# ----------------------------------------------------------------------

_LAZY: dict = {}


def _lazy_imports():
    """Load heavy deps once per worker."""
    if _LAZY:
        return
    import lamquant_core  # PyO3 extension
    # CHANNEL_PRESETS moved to lamquant.dataset.preprocess in the BLUT/Neural
    # split (was ai_models.dataset_sim.preprocess). Repoint to the current
    # location; fall back to the inline 21-channel 10-20 firmware contract so a
    # decode can never crash on a missing import (R1 fix 2026-05-29).
    try:
        from lamquant.dataset.preprocess import CHANNEL_PRESETS as _CP
    except ImportError:
        _CP = {21: [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'A1', 'A2']}
    from lamquant_codec import channel_resolver as _cr
    from lamquant_codec.ops.pipeline import preprocess_subband_single as _psbs
    from scipy.signal import resample_poly as _rp
    _LAZY["lamquant_core"] = lamquant_core
    _LAZY["channel_resolver"] = _cr
    _LAZY["preprocess_subband_single"] = _psbs
    _LAZY["resample_poly"] = _rp
    _LAZY["CHANNEL_PRESETS"] = _CP


def _highpass_sos():
    if not hasattr(_highpass_sos, "_cache"):
        from scipy.signal import butter
        _highpass_sos._cache = butter(
            2, HIGHPASS_HZ, btype="high", fs=TARGET_SR, output="sos"
        )
    return _highpass_sos._cache


# ----------------------------------------------------------------------
# Decode helper — public so other modules (e.g. SNN dataset) can share
# this exact pipeline rather than maintaining a parallel copy.
# ----------------------------------------------------------------------

def list_lma_entries(lma_path: str) -> List[str]:
    """Return every entry path inside one ``.lma`` archive.

    Uses ``lml ls`` (subprocess). On a 114 K-entry TUEG archive this
    completes in ~290 ms; on the smaller per-dataset corpora it's
    sub-10 ms. Called once at ``LmaDataset.__init__`` per LMA to
    build the stem -> internal-path index.
    """
    import subprocess
    try:
        res = subprocess.run(
            ["lml", "ls", str(lma_path)],
            capture_output=True, text=True, timeout=60, check=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "lml binary not on PATH; cannot list LMA entries"
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"lml ls failed for {lma_path}: rc={e.returncode} {e.stderr[:300]}"
        ) from e
    return [ln for ln in res.stdout.splitlines() if ln]


def list_lma_entries_typed(lma_path: str) -> List[Tuple[str, str]]:
    """Return ``[(entry_path, method)]`` for one ``.lma`` archive.

    Parses ``lml list-archive`` (the table form) so callers can tell a
    losslessly-compressed RECORDING (``method == "lml"``) from a sibling
    annotation stored via zstd / store. Needed for the ``lml archive``
    (per-corpus source-of-truth) layout, where recordings keep their
    original data-file name (``chb01/chb01_01.edf``, method ``lml``)
    rather than a ``<stem>.lml`` rename.

    Table columns: ``PATH  ORIGINAL  COMPRESSED  METHOD  SHA256``. The
    size columns embed a space (``40.4 MB``), so PATH is the first field
    and METHOD/SHA256 are the last two. Header / separator / summary
    lines are skipped.
    """
    import subprocess
    try:
        res = subprocess.run(
            ["lml", "list-archive", str(lma_path)],
            capture_output=True, text=True, timeout=120, check=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "lml binary not on PATH; cannot list LMA entries"
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"lml list-archive failed for {lma_path}: rc={e.returncode} "
            f"{e.stderr[:300]}"
        ) from e
    out: List[Tuple[str, str]] = []
    for ln in res.stdout.splitlines():
        s = ln.strip()
        if not s or s.startswith("PATH") or set(s) <= set("-"):
            continue
        if "files," in s and "original" in s:  # summary line
            continue
        parts = ln.split()
        if len(parts) < 3:
            continue
        path, method = parts[0], parts[-2]
        out.append((path, method))
    return out


def build_lma_entry_index(lma_paths: Sequence[str]) -> dict:
    """Walk every LMA's entries; return ``{stem: {"lma": <path>,
    "lml": <internal_path>, "labels": <internal_path>|None,
    "meta": <internal_path>|None}}``.

    Per-dataset LMAs (Phase M, 2026-05-20) hold many recordings under
    their original directory tree. Entries look like:
        ``edf/<NNN>/<subj>/<sess>/<montage>/<stem>.lml`` (deep — TUEG)
        ``edf/<stem>.lml`` (flat — TUAR/TUAB/TUEV/PhysioNet)
        ``labels/<stem>_labels.npz``
        ``meta.json`` (per-dataset, single)

    Stem is the basename minus ``.lml`` or ``_labels.npz`` suffix.
    Returns a dict keyed by stem. A stem present in multiple LMAs is
    deterministically resolved to the FIRST LMA seen (caller controls
    ordering of ``lma_paths``); a warning is logged for collisions so
    audit can investigate cross-corpus duplicates.
    """
    import os
    import re
    out: dict = {}
    # Match ANY recording stem basename — TUH <subj>_sN_tN, TUEV <subj>_8digit,
    # CHB-MIT chbNN_NN, Siena PNxx-N (R2 fix 2026-05-29). The basename IS the stem.
    lml_re = re.compile(r"(?:^|/)([^/]+)\.lml$")
    lbl_re = re.compile(r"(?:^|/)([^/]+)_labels\.npz$")
    for lp in lma_paths:
        entries = list_lma_entries(lp)
        matched_any = False
        for e in entries:
            m_lml = lml_re.search(e)
            if m_lml:
                matched_any = True
                stem = m_lml.group(1)
                if stem in out:
                    LOG.debug("stem %s already mapped to %s; skipping %s",
                              stem, out[stem]["lma"], lp)
                    continue
                out[stem] = {"lma": str(lp), "lml": e, "labels": None,
                             "meta": None}
        # `lml archive` source-of-truth layout (2026-05-30): recordings keep
        # their original data-file name (``chb01/chb01_01.edf``, method
        # ``lml``) instead of a ``<stem>.lml`` rename, and there is no
        # ``meta.json`` or in-archive labels NPZ (labels live in the
        # disk-staged cache). Recognise lml-method data entries as recordings;
        # the basename minus its extension is the stem. `lma_read_entry` on an
        # lml-method entry returns the LML container bytes (magic LML1), so
        # decode_lma_signal works unchanged.
        #
        # Run this whenever the archive has any non-``.lml`` entry — NOT only
        # when zero ``<stem>.lml`` matched. A single stray ``<stem>.lml`` left
        # in an EDF source tree would otherwise flip ``matched_any`` and mask
        # thousands of ``.edf`` recordings (tuep_v3.1.0 regen: 1 stray `.lml`
        # hid 2808 `.edf`). Deduped by stem, so the ``<stem>.lml`` fast-path
        # entries above stay authoritative. Pure-``.lml`` archives skip the
        # heavier ``list-archive`` typed scan entirely (fast path preserved).
        if any(not e.endswith(".lml") for e in entries):
            # Identify recordings by data-file extension from the FULL `lml ls`
            # paths already in `entries`. Do NOT parse `lml list-archive` here:
            # its table left-truncates the PATH column to a fixed width, so
            # entry names longer than ~60 chars are corrupted (tuep's
            # `00_epilepsy/...edf` -> `epilepsy/...edf`), silently breaking the
            # lookup. `lma_read_entry` on these returns the LML container bytes
            # (magic LML1); a data file that wasn't lml-stored decodes to None
            # downstream and is skipped, so extension is a safe discriminator.
            for path in entries:
                if not path.lower().endswith(_RECORDING_EXTS):
                    continue
                stem = os.path.splitext(os.path.basename(path))[0]
                if not stem or stem in out:
                    continue
                out[stem] = {"lma": str(lp), "lml": path, "labels": None,
                             "meta": None}
        # Second pass for labels (LML entry must exist for stem before we
        # attach labels; orphan labels with no LML are ignored).
        for e in entries:
            m_lbl = lbl_re.search(e)
            if m_lbl:
                stem = m_lbl.group(1)
                if stem in out and out[stem]["lma"] == str(lp):
                    out[stem]["labels"] = e
        # Per-dataset meta (single per archive) — attach to every stem
        # from this LMA so callers can find dataset-level metadata.
        if "meta.json" in entries:
            for stem, info in out.items():
                if info["lma"] == str(lp) and info["meta"] is None:
                    info["meta"] = "meta.json"
    return out


def _validate_stem(stem: str) -> None:
    """Reject stems that could escape ``lma_root`` via path traversal.

    Stems come from the split manifest which is generated by
    ``build_manifest.py`` over an EDF file scan — not user-supplied at
    runtime. Defence-in-depth: if a manifest is ever crafted by an
    adversary (or accidentally contains a relative path), reject the
    construction rather than reading e.g. ``/etc/passwd.lma``.
    """
    if not stem or any(sep in stem for sep in ("/", "\\")) or ".." in stem.split("."):
        raise ValueError(f"invalid stem (path traversal guard): {stem!r}")
    if stem.startswith(("-", "~", ".")):
        raise ValueError(f"invalid stem leading char: {stem!r}")


def decode_lma_signal(lma_path: str, stem: str,
                      lml_entry_name: Optional[str] = None) -> Optional[np.ndarray]:
    """LMA -> fully-preprocessed float32 signal ``[21, T_resampled]``.

    Bit-exact with the deprecated ``precompute_l3_fast`` pipeline:
    digital -> microvolts -> 21ch select -> 250 Hz resample
    -> 0.5 Hz highpass -> Q31 round-trip.

    Memory (ADR 0075 A3): a memory-mapped, selected-channel decode. The
    container header is peeked via mmap (no full-entry copy), the 21
    target channels are resolved to source indices up front, and ONLY
    those channels are decoded straight from the map — skipping both the
    ~1.4 GB compressed-entry copy and the ``[n_ch, total]`` all-channel
    f32 array. Peak drops from ~4.7 GB to ~1.2 GB on an 8 hr 27-ch TUEG
    file. BIT-IDENTICAL to the prior ``container_read_phys_f32`` +
    ``extract_channel_data`` path (same resolver, calibration, and
    missing-channel zero-fill).

    Args:
        lma_path: path to the ``.lma`` archive containing the LML.
        stem: recording stem (e.g. ``aaaaaaaa_s001_t000``). Used for
            log messages and (when ``lml_entry_name`` is None) to derive
            the per-stem entry name ``<stem>.lml`` — the historic layout
            for per-recording archives.
        lml_entry_name: explicit internal entry path inside the LMA
            (e.g. ``edf/<NNN>/<subj>/<sess>/<montage>/<stem>.lml``).
            Required for the per-dataset LMA layout (Phase M, 2026-05-20)
            where one archive holds many recordings under their original
            directory tree. When omitted, falls back to ``<stem>.lml``.

    Returns ``None`` if the LML cannot be decoded, channels are missing,
    the file is flat (max_abs < 1e-12), or the metadata reports a
    non-positive sample rate. Callers cache the non-None results; ``None``
    is never cached to avoid cache poisoning on transient I/O errors
    (V4 Pro 2026-05-16 footgun).
    """
    _lazy_imports()

    lc = _LAZY["lamquant_core"]
    # ADR 0075 A3 requires the mmap+selected decode entry points. On a STALE extension a
    # bare attribute miss would surface as an AttributeError caught below → every recording
    # silently drops → training on empty data. Fail LOUD + actionable instead.
    if not hasattr(lc, "lma_mmap_read_phys_selected"):
        raise RuntimeError(
            "lamquant_core is stale: decode_lma_signal (ADR 0075 A3) needs "
            "lma_mmap_read_phys_selected / lma_mmap_entry_metadata — rebuild the extension "
            "with `maturin develop` (or `pip install -e`)."
        )
    entry = lml_entry_name if lml_entry_name is not None else f"{stem}.lml"

    # ADR 0075 A3 — mmap + selected-channel decode. Peek the container header/metadata via
    # a memory map (touches only the header pages, no full-entry copy), resolve the 21
    # TARGET_CHANNELS to source indices UP FRONT, then decode ONLY those channels straight
    # from the map. This skips BOTH the ~1.4 GB compressed-entry copy AND the [n_ch, total]
    # all-channel f32 array — the two allocations that drove the ~4.7 GB decode-fallback
    # peak (→ ~1.2 GB). BIT-IDENTICAL to the old container_read_phys_f32 +
    # extract_channel_data path: same resolver (channel_resolver.pick_channels), same
    # per-channel calibration, same zero-fill of missing optional electrodes.
    try:
        meta_json, _n_ch, _n_win, _total, _ws = lc.lma_mmap_entry_metadata(lma_path, entry)
    except Exception as e:
        LOG.warning("LMA mmap header peek failed for %s (%s): %s", stem, entry, e)
        return None
    metadata = json.loads(meta_json)
    all_ch_names = metadata.get("channels", [])
    n_ch = len(all_ch_names)
    if n_ch == 0:
        LOG.warning("LMA %s has zero channels in metadata", stem)
        return None

    original_sr = float(metadata.get("sample_rate", TARGET_SR))
    if not (original_sr > 0.0):
        LOG.warning("LMA %s reports non-positive sample_rate=%s; skipping.",
                    stem, original_sr)
        return None

    # Resolve the 21 targets to source indices (the same resolver the old
    # extract_channel_data used); None => too many required channels absent.
    src_indices = _LAZY["channel_resolver"].pick_channels(all_ch_names)
    if src_indices is None:
        LOG.warning("LMA %s missing channels (unresolved)", stem)
        return None
    # -1 (missing optional electrode) => u16::MAX sentinel; the Rust decode zero-fills it.
    _MISSING = 65535
    channel_mask = [i if i >= 0 else _MISSING for i in src_indices]

    # Per-channel calibration, selected + reordered to the 21 target rows. Defaults match
    # the legacy `lml_digital_to_float` fallbacks (full int16 range). Missing rows get a
    # harmless default — their output is zero-filled regardless.
    dig_min = metadata.get("dig_min", [-32768.0] * n_ch)
    dig_max = metadata.get("dig_max", [32767.0] * n_ch)
    phys_min = metadata.get("phys_min", [-32768.0] * n_ch)
    phys_max = metadata.get("phys_max", [32767.0] * n_ch)
    calib_sel = np.zeros((len(channel_mask), 4), dtype=np.float32)
    for row, src in enumerate(src_indices):
        if src >= 0:
            calib_sel[row] = (dig_min[src], dig_max[src], phys_min[src], phys_max[src])
    calib_sel = np.ascontiguousarray(calib_sel.reshape(-1), dtype=np.float32)

    # (Second mmap of the same archive — the header peek above already warmed the page
    # cache, so re-opening + re-parsing the index is cheap; channel resolution must sit
    # between the two calls, so they can't be fused without a stateful handle.)
    try:
        data, _meta_again, _n_win = lc.lma_mmap_read_phys_selected(
            lma_path, entry, calib_sel, channel_mask
        )
    except Exception as e:
        LOG.warning("LMA mmap selected decode failed for %s (%s): %s", stem, entry, e)
        return None
    if data.size == 0:
        return None

    target_chs = _LAZY["CHANNEL_PRESETS"][TARGET_CHANNELS]
    n_target = len(target_chs)
    if data.shape[0] != n_target:
        padded = np.zeros((n_target, data.shape[1]), dtype=data.dtype)
        padded[: min(data.shape[0], n_target)] = data[: min(data.shape[0], n_target)]
        data = padded

    # ADR 0075 — SINGLE SOURCE of the normalization DSP: Rust (`src/normalize.rs`):
    # resample→250 → 0.5 Hz zero-phase HP → Q31 → f32. Rust handles EVERY rate (the FFT
    # branch, #37) and is parity-gated bit-exact to the former scipy path on the common
    # corpus rates (within the ADC noise floor on exotic rates). The scipy RUNTIME path is
    # gone — scipy survives only as the OFFLINE golden generator (tools/dump_normalize_
    # golden.py) that regenerates the normalize_parity golden, so there is no second live
    # implementation that could silently drift. (A missing/stale extension fails loud via
    # the guard at the top of this function rather than falling back to a second DSP.)
    # Returns the [21, T'] f32 array, or None on an all-flat signal.
    return _LAZY["lamquant_core"].normalize_eeg_f32(
        np.ascontiguousarray(data, dtype=np.float32), original_sr
    )


# ----------------------------------------------------------------------
# Per-worker LRU on processed signals
# ----------------------------------------------------------------------

_SIGNAL_CACHE: "collections.OrderedDict[Tuple[str, str], np.ndarray]" = \
    collections.OrderedDict()


def _cached_signal(lma_path: str, stem: str) -> Optional[np.ndarray]:
    """Per-worker LRU wrapping ``decode_lma_signal``.

    Returns the same ``Optional[np.ndarray]`` as the underlying decode.
    Only non-None results are cached so a transient I/O failure can be
    re-attempted on the next call.
    """
    key = (lma_path, stem)
    cached = _SIGNAL_CACHE.get(key)
    if cached is not None:
        _SIGNAL_CACHE.move_to_end(key)
        return cached
    result = decode_lma_signal(lma_path, stem)
    if result is None:
        return None
    _SIGNAL_CACHE[key] = result
    if len(_SIGNAL_CACHE) > SIGNAL_CACHE_CAP:
        _SIGNAL_CACHE.popitem(last=False)
    return result


# ----------------------------------------------------------------------
# Window-selection helpers
# ----------------------------------------------------------------------

def select_random_windows(
    n_total_windows: int,
    windows_per_epoch: int,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """Random-with-replacement window indices when the corpus is larger
    than ``windows_per_epoch``; full enumeration otherwise.

    Replaces ``StreamingQ31Dataset``'s per-epoch sampling without the
    NPZ-specific bookkeeping.
    """
    rng = rng or random.Random()
    if windows_per_epoch >= n_total_windows:
        return list(range(n_total_windows))
    return [rng.randrange(n_total_windows) for _ in range(windows_per_epoch)]


def _stem_from_lma(p: Path) -> str:
    """``.lma`` filename stem (drop the ``.lma`` suffix only)."""
    name = p.name
    if name.endswith(".lma"):
        return name[: -len(".lma")]
    return p.stem


def load_split_stems(
    manifest_path: Path,
    split: str,
) -> Tuple[List[str], dict]:
    """Read a BLUT/build_manifest split manifest and return ``(stems, subject_by_stem)``.

    Manifest schema (matches ``ai_models/snn/lma_dataset.py``):
        {
            "subjects": {subject_id: "train" | "val", ...},
            "stems_by_subject": {subject_id: [stem, ...], ...}
        }

    Also enforces the defence-in-depth subject-bleed check: loading
    ``train`` validates that no stem belongs to ``val`` and vice
    versa. Returns hostile-caller-safe data.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"split manifest not found: {manifest_path}")
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    meta = json.loads(manifest_path.read_text())
    subjects = meta["subjects"]
    stems_by_subject = meta.get("stems_by_subject", {})

    def _stems_for(target_split: str) -> Tuple[List[str], dict]:
        out, subj_map = [], {}
        for sid, assigned in subjects.items():
            if assigned != target_split:
                continue
            for stem in stems_by_subject.get(sid, []):
                out.append(stem)
                subj_map[stem] = sid
        return out, subj_map

    stems, subject_by_stem = _stems_for(split)
    other = "val" if split == "train" else "train"
    other_stems, _ = _stems_for(other)
    overlap = set(stems) & set(other_stems)
    if overlap:
        raise RuntimeError(
            f"split manifest corrupt: {len(overlap)} stems in both train and val "
            f"(first 5: {sorted(overlap)[:5]})"
        )
    return stems, subject_by_stem


def _enumerate_lma_files(lma_root: Path) -> List[Path]:
    return sorted(p for p in Path(lma_root).glob("*.lma") if p.is_file())


def _read_meta(lma_path: Path) -> Optional[dict]:
    _lazy_imports()
    try:
        meta_bytes = _LAZY["lamquant_core"].lma_read_entry(str(lma_path), "meta.json")
        return json.loads(meta_bytes)
    except Exception as e:
        LOG.warning("meta.json unreadable for %s: %s", lma_path, e)
        return None


# ----------------------------------------------------------------------
# Generic signal-window dataset
# ----------------------------------------------------------------------

class LmaSignalDataset(Dataset):
    """Random-access raw fullband windows from an LMA corpus.

    Replaces ``StreamingQ31Dataset`` / ``HybridQ31Dataset`` /
    ``Q31Dataset`` / ``MemmapTeacherDataset``. ``__getitem__`` returns a
    single ``[21, 2500] float32`` tensor.

    Args:
        lma_root: directory containing ``.lma`` archives (one per recording).
        file_stems: optional list of stems to restrict to (used by manifest-
            driven train/val splits). When ``None`` every ``.lma`` under
            ``lma_root`` is enumerated.
        windows_per_epoch: epoch length; sampled with replacement when the
            corpus is larger than this.
        seed: per-instance random seed for window sampling.

    Raises ``RuntimeError`` when zero LMAs match or when meta.json is
    unreadable across the entire corpus.
    """

    def __init__(
        self,
        lma_root: Path,
        file_stems: Optional[Sequence[str]] = None,
        windows_per_epoch: int = 50_000,
        seed: int = 0,
    ):
        lma_root = Path(lma_root)
        if not lma_root.exists():
            raise FileNotFoundError(f"lma_root not found: {lma_root}")

        if file_stems is not None:
            for stem in file_stems:
                _validate_stem(stem)
            lma_paths = [lma_root / f"{stem}.lma" for stem in file_stems]
            lma_paths = [p for p in lma_paths if p.exists()]
        else:
            lma_paths = _enumerate_lma_files(lma_root)

        if not lma_paths:
            raise RuntimeError(f"no .lma archives found under {lma_root}")

        # Build a window index using each LMA's metadata. One pass of
        # meta.json reads; window counts derived from signal_len / 2500.
        self.windows: List[Tuple[Path, str, int]] = []
        n_missing_meta = 0
        for lp in lma_paths:
            meta = _read_meta(lp)
            if meta is None:
                n_missing_meta += 1
                continue
            n_samples = int(meta.get("n_samples_resampled") or 0)
            if n_samples <= 0:
                # Fall back to decoding the signal head — rare, costs ~50 ms.
                stem = _stem_from_lma(lp)
                sig = decode_lma_signal(str(lp), stem)
                if sig is None:
                    continue
                n_samples = sig.shape[1]
            n_windows = max(1, n_samples // WINDOW_SAMPLES)
            stem = _stem_from_lma(lp)
            for wi in range(n_windows):
                self.windows.append((lp, stem, wi))

        if not self.windows:
            raise RuntimeError(
                f"LmaSignalDataset produced 0 windows under {lma_root}"
            )

        self.windows_per_epoch = windows_per_epoch
        self._rng = random.Random(seed)
        self._epoch_indices = select_random_windows(
            len(self.windows), self.windows_per_epoch, self._rng
        )
        LOG.info(
            "[LmaSignalDS] %d LMAs, %d total windows, %d sampled per epoch "
            "(%d meta-missing)",
            len(lma_paths) - n_missing_meta,
            len(self.windows),
            len(self._epoch_indices),
            n_missing_meta,
        )

    def reshuffle(self):
        """Re-sample epoch_indices; call between epochs for SGD coverage."""
        self._epoch_indices = select_random_windows(
            len(self.windows), self.windows_per_epoch, self._rng
        )

    def __len__(self) -> int:
        return len(self._epoch_indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        lp, stem, wi = self.windows[self._epoch_indices[idx]]
        signal = _cached_signal(str(lp), stem)
        if signal is None:
            return torch.zeros(TARGET_CHANNELS, WINDOW_SAMPLES, dtype=torch.float32)
        start = wi * WINDOW_SAMPLES
        end = start + WINDOW_SAMPLES
        if end > signal.shape[1]:
            window = np.zeros((TARGET_CHANNELS, WINDOW_SAMPLES), dtype=np.float32)
            window[:, : signal.shape[1] - start] = signal[:, start:]
        else:
            window = signal[:, start:end]
        return torch.from_numpy(np.ascontiguousarray(window))


# ----------------------------------------------------------------------
# Generic L3 dataset — drop-in PrecomputedL3Dataset replacement
# ----------------------------------------------------------------------

class LmaL3Dataset(Dataset):
    """Random-access L3 windows from an LMA corpus.

    Drop-in replacement for ``ai_models.oracle.streaming_dataset.PrecomputedL3Dataset``.
    Same 3-tuple contract ``(l3, l3, dummy_mask)`` so the trainer loop
    (which only consumes the first element) needs zero edits beyond the
    constructor swap.

    Args mirror PrecomputedL3Dataset where they have a 1:1 equivalent;
    the legacy ``file_paths`` argument is reinterpreted as a list of
    ``.lma`` paths (or stems under ``lma_root`` when provided), and the
    NPZ-only fullband memmap path is dropped (LMA-direct training reads
    fullband from the same archive, see ``LmaSignalDataset``).
    """

    def __init__(
        self,
        lma_root: Optional[Path] = None,
        *,
        file_stems: Optional[Sequence[str]] = None,
        file_paths: Optional[Sequence[Path]] = None,
        windows_per_epoch: int = 50_000,
        max_windows: Optional[int] = None,
        seed: int = 0,
        return_fullband: bool = False,
    ):
        if lma_root is None and file_paths is None:
            raise ValueError("Pass lma_root (+ optional file_stems) OR file_paths.")
        if file_paths is not None:
            lma_paths = [Path(p) for p in file_paths]
        else:
            lma_root = Path(lma_root)
            if file_stems is not None:
                for stem in file_stems:
                    _validate_stem(stem)
                lma_paths = [lma_root / f"{stem}.lma" for stem in file_stems]
            else:
                lma_paths = _enumerate_lma_files(lma_root)
        lma_paths = [p for p in lma_paths if p.exists()]
        if not lma_paths:
            raise RuntimeError("LmaL3Dataset: no .lma archives matched.")

        self.windows: List[Tuple[Path, str, int]] = []
        for lp in lma_paths:
            meta = _read_meta(lp)
            if meta is None:
                continue
            n_samples = int(meta.get("n_samples_resampled") or 0)
            if n_samples <= 0:
                stem = _stem_from_lma(lp)
                sig = decode_lma_signal(str(lp), stem)
                if sig is None:
                    continue
                n_samples = sig.shape[1]
            n_windows = max(1, n_samples // WINDOW_SAMPLES)
            if max_windows is not None and len(self.windows) + n_windows > max_windows:
                n_windows = max_windows - len(self.windows)
                if n_windows <= 0:
                    break
            stem = _stem_from_lma(lp)
            for wi in range(n_windows):
                self.windows.append((lp, stem, wi))

        if not self.windows:
            raise RuntimeError("LmaL3Dataset: 0 windows after meta scan.")

        self.windows_per_epoch = windows_per_epoch
        self._rng = random.Random(seed)
        self._epoch_indices = select_random_windows(
            len(self.windows), self.windows_per_epoch, self._rng
        )
        # 3-tuple legacy contract — last element is a dummy mask of L3_T zeros
        # in the default mode, or the fullband window when return_fullband=True.
        # Joint encoder + Vocos training (train_joint.py with tier >= 3) needs
        # the raw fullband target to compute the product-metric loss; in the
        # deprecated path that was wired via PrecomputedL3Dataset(with_fullband=...)
        # and a separate memmap. LMA-direct delivers it from the same archive
        # we already decoded for L3.
        self._dummy_mask = torch.zeros(L3_T)
        self._return_fullband = return_fullband
        LOG.info(
            "[LmaL3DS] %d LMAs, %d windows total, %d sampled per epoch",
            len(lma_paths),
            len(self.windows),
            len(self._epoch_indices),
        )

    def reshuffle(self):
        self._epoch_indices = select_random_windows(
            len(self.windows), self.windows_per_epoch, self._rng
        )

    def __len__(self) -> int:
        return len(self._epoch_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lp, stem, wi = self.windows[self._epoch_indices[idx]]
        signal = _cached_signal(str(lp), stem)
        if signal is None:
            l3 = torch.zeros(TARGET_CHANNELS, L3_T, dtype=torch.float32)
            tail = (
                torch.zeros(TARGET_CHANNELS, WINDOW_SAMPLES, dtype=torch.float32)
                if self._return_fullband
                else self._dummy_mask
            )
            return l3, l3, tail
        start = wi * WINDOW_SAMPLES
        end = start + WINDOW_SAMPLES
        if end > signal.shape[1]:
            window = np.zeros((TARGET_CHANNELS, WINDOW_SAMPLES), dtype=np.float32)
            window[:, : signal.shape[1] - start] = signal[:, start:]
        else:
            window = signal[:, start:end]

        _lazy_imports()
        l3_arr, _, _ = _LAZY["preprocess_subband_single"](
            window.astype(np.float32), order=8, autocorr_len=256
        )
        l3 = torch.from_numpy(np.ascontiguousarray(np.asarray(l3_arr, dtype=np.float32)))
        if self._return_fullband:
            fullband = torch.from_numpy(np.ascontiguousarray(window))
            return l3, l3, fullband
        return l3, l3, self._dummy_mask
