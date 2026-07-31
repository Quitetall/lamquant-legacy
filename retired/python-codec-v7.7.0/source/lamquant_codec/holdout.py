"""Canonical benchmark holdout set and standardized evaluation.

Defines a fixed set of EEG windows for reproducible quality measurement.
No cherry-picking — the same 100 windows every time, seeded from the
sorted file list with a fixed random state.

Usage:
    from lamquant_codec.holdout import load_holdout, run_benchmark

    windows = load_holdout()                    # 100 fixed windows
    report = run_benchmark(codec_fn, windows)   # standardized report

The holdout set is split across sources and content types:
    - 40 windows from CHB-MIT (pediatric epilepsy)
    - 40 windows from TUH (adult clinical)
    - 10 windows with seizure activity (high energy)
    - 10 windows with minimal activity (near-baseline)
"""

import os
import glob
import hashlib
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional

from lamquant_codec.codec_types import EEGPacket
from lamquant_codec.benchmark import Benchmark

from lamquant_codec._paths import REPO_ROOT
_REPO = str(REPO_ROOT)


def _default_holdout_dir():
    """Resolve the holdout-dataset directory without training-tree leak.

    Resolution (ADR 0018):
      1. ``$LAMQUANT_HOLDOUT_DIR`` env var (caller override).
      2. ``<repo>/weights/holdout_q31/`` (companion of the
         weights tree the codec is allowed to know about).

    Both candidates are validated against ``os.path.isdir``. If the
    env var points at a non-directory, falls through to the default
    rather than silently glob-zero-filing. If neither resolves the
    caller (``load_holdout``) raises a clear "dir not found" error
    instead of the misleading "< HOLDOUT_SIZE" RuntimeError that the
    pre-validation behaviour produced.

    The pre-ADR-0018 default (`ai_models/dataset_sim/q31_events`) is
    removed — codec runtime no longer references the training tree.
    Producer-side companion (write q31_events → weights/holdout_q31)
    is queued as a follow-up commit ("T1.4b" in the active plan).
    """
    env_dir = os.environ.get("LAMQUANT_HOLDOUT_DIR")
    if env_dir and os.path.isdir(env_dir):
        return env_dir
    return os.path.join(_REPO, "weights", "holdout_q31")

# Fixed seed for reproducible holdout selection
HOLDOUT_SEED = 20260415
HOLDOUT_SIZE = 100

# Holdout composition
N_CHBMIT = 40
N_TUH = 40
N_HIGH_ENERGY = 10
N_LOW_ENERGY = 10

RAW_BYTES_PER_WINDOW = 21 * 2500 * 2  # int16


@dataclass
class HoldoutWindow:
    """One window from the canonical holdout set."""
    filename: str
    signal: np.ndarray       # [21, 2500] float32 (uV scale)
    l3: np.ndarray           # [21, 313] L3 approximation
    source: str              # 'chbmit' or 'tuh'
    energy: float            # RMS energy of the window
    index: int               # position in holdout set


def _load_window(path: str) -> tuple:
    """Load one Q31 file and extract the first 10s window."""
    d = np.load(path)
    raw = d['data']
    l3 = d['l3']
    spw = raw.shape[1] // l3.shape[0]
    seg = (raw[:, :spw].astype(np.float32) / 2147483647.0) * 1000.0
    l3_seg = l3[:, :313].astype(np.float32) if l3.shape[1] >= 313 else l3.astype(np.float32)
    energy = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
    source = 'chbmit' if 'chbmit' in os.path.basename(path) else 'tuh'
    return seg, l3_seg, source, energy


def load_holdout(data_dir: Optional[str] = None) -> list:
    """Load the canonical holdout set of 100 EEG windows.

    Deterministic selection: sorted file list, fixed seed, stratified sampling.
    Returns the same 100 windows every time regardless of platform.

    Args:
        data_dir: override Q31 data directory (default: repo q31_events)
    Returns:
        list of HoldoutWindow objects
    """
    data_dir = data_dir or _default_holdout_dir()
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"holdout dataset directory not found: {data_dir!r}. "
            f"Set $LAMQUANT_HOLDOUT_DIR or pass `data_dir=`. "
            f"Default is weights/holdout_q31/ (see ADR 0018)."
        )
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if len(files) < HOLDOUT_SIZE:
        raise RuntimeError(
            f"Need at least {HOLDOUT_SIZE} Q31 files for holdout, found {len(files)}. "
            f"Run dataset conversion first.")

    # Split by source
    chb_files = [f for f in files if 'chbmit' in os.path.basename(f)]
    tuh_files = [f for f in files if 'tuh' in os.path.basename(f)]

    rng = np.random.RandomState(HOLDOUT_SEED)

    # Select from each source
    chb_idx = rng.choice(len(chb_files), size=min(N_CHBMIT, len(chb_files)), replace=False)
    tuh_idx = rng.choice(len(tuh_files), size=min(N_TUH, len(tuh_files)), replace=False)

    selected_files = ([chb_files[i] for i in sorted(chb_idx)] +
                      [tuh_files[i] for i in sorted(tuh_idx)])

    # Load all selected windows
    windows = []
    for i, path in enumerate(selected_files):
        try:
            seg, l3, source, energy = _load_window(path)
            windows.append(HoldoutWindow(
                filename=os.path.basename(path),
                signal=seg, l3=l3, source=source,
                energy=energy, index=i,
            ))
        except Exception as exc:
            import warnings
            warnings.warn(f"holdout window {path}: {exc}")
            continue

    # Sort by energy and select high/low energy extremes
    by_energy = sorted(windows, key=lambda w: w.energy)
    low_energy = by_energy[:N_LOW_ENERGY]
    high_energy = by_energy[-N_HIGH_ENERGY:]

    # Remaining slots filled from the middle
    middle = by_energy[N_LOW_ENERGY:-N_HIGH_ENERGY]
    n_middle = HOLDOUT_SIZE - N_HIGH_ENERGY - N_LOW_ENERGY
    if len(middle) > n_middle:
        step = len(middle) / n_middle
        middle = [middle[int(i * step)] for i in range(n_middle)]

    holdout = low_energy + middle + high_energy

    # Re-index
    for i, w in enumerate(holdout):
        w.index = i

    return holdout


def holdout_fingerprint(windows: list) -> str:
    """SHA256 fingerprint of the holdout set for reproducibility verification."""
    h = hashlib.sha256()
    for w in windows:
        h.update(w.filename.encode())
        h.update(w.signal[:, :10].tobytes())  # first 10 samples per channel
    return h.hexdigest()[:16]


@dataclass
class BenchmarkReport:
    """Standardized benchmark results."""
    n_windows: int
    mean_prd: float
    mean_r: float
    mean_cr: float
    mean_snr_db: float
    per_band_prd: dict          # {delta, theta, alpha, beta, gamma}
    per_source: dict            # {chbmit: {prd, r}, tuh: {prd, r}}
    high_energy_prd: float      # PRD on seizure-like windows
    low_energy_prd: float       # PRD on baseline windows
    holdout_fingerprint: str
    lossless_count: int         # how many were lossless

    def summary(self) -> str:
        lines = [
            f"LamQuant Benchmark Report ({self.n_windows} windows)",
            f"  Holdout fingerprint: {self.holdout_fingerprint}",
            f"  Mean PRD:  {self.mean_prd:.2f}%",
            f"  Mean R:    {self.mean_r:.4f}",
            f"  Mean CR:   {self.mean_cr:.1f}:1",
            f"  Mean SNR:  {self.mean_snr_db:.1f} dB",
            f"  Lossless:  {self.lossless_count}/{self.n_windows}",
            f"  Band PRD:  " + ", ".join(f"{k}={v:.1f}%" for k, v in self.per_band_prd.items()),
            f"  CHB-MIT:   PRD={self.per_source.get('chbmit', {}).get('prd', 0):.2f}%  R={self.per_source.get('chbmit', {}).get('r', 0):.4f}",
            f"  TUH:       PRD={self.per_source.get('tuh', {}).get('prd', 0):.2f}%  R={self.per_source.get('tuh', {}).get('r', 0):.4f}",
            f"  High E:    PRD={self.high_energy_prd:.2f}%",
            f"  Low E:     PRD={self.low_energy_prd:.2f}%",
        ]
        return "\n".join(lines)


def run_benchmark(codec_fn: Callable, windows: Optional[list] = None,
                  use_l3: bool = True) -> BenchmarkReport:
    """Run standardized benchmark on the holdout set.

    Args:
        codec_fn: function(signal) -> EEGPacket. Takes [C, T] numpy array,
                  returns an EEGPacket with the reconstruction.
        windows: holdout windows (default: load canonical set)
        use_l3: if True, benchmark on L3 [21, 313]; if False, on full [21, 2500]

    Returns:
        BenchmarkReport with all standardized metrics
    """
    if windows is None:
        windows = load_holdout()

    fp = holdout_fingerprint(windows)
    prds, rs, crs, snrs = [], [], [], []
    band_prds = {b: [] for b in ['delta', 'theta', 'alpha', 'beta', 'gamma']}
    source_metrics = {}
    high_e_prds, low_e_prds = [], []
    lossless_count = 0

    for w in windows:
        original = w.l3 if use_l3 else w.signal
        try:
            packet = codec_fn(original)
        except Exception as e:
            print(f"[!] Window {w.index} ({w.filename}): {e}")
            continue

        prd = Benchmark.prd(original, packet)
        r = Benchmark.pearson_r(original, packet)
        cr = Benchmark.compression_ratio(original, packet)
        snr = Benchmark.snr_db(original, packet)
        bp = Benchmark.per_band_prd(original, packet, sample_rate=250)

        prds.append(prd)
        rs.append(r)
        crs.append(cr)
        snrs.append(snr)
        for band_name, val in bp.items():
            band_prds[band_name].append(val)

        if Benchmark.is_lossless(original, packet):
            lossless_count += 1

        # Per-source tracking
        if w.source not in source_metrics:
            source_metrics[w.source] = {'prds': [], 'rs': []}
        source_metrics[w.source]['prds'].append(prd)
        source_metrics[w.source]['rs'].append(r)

        # Energy-stratified
        if w.index >= HOLDOUT_SIZE - N_HIGH_ENERGY:
            high_e_prds.append(prd)
        elif w.index < N_LOW_ENERGY:
            low_e_prds.append(prd)

    per_source = {}
    for src, m in source_metrics.items():
        per_source[src] = {
            'prd': float(np.mean(m['prds'])) if m['prds'] else 0,
            'r': float(np.mean(m['rs'])) if m['rs'] else 0,
        }

    return BenchmarkReport(
        n_windows=len(prds),
        mean_prd=float(np.mean(prds)) if prds else 0,
        mean_r=float(np.mean(rs)) if rs else 0,
        mean_cr=float(np.mean(crs)) if crs else 0,
        mean_snr_db=float(np.mean(snrs)) if snrs else 0,
        per_band_prd={k: float(np.mean(v)) for k, v in band_prds.items()},
        per_source=per_source,
        high_energy_prd=float(np.mean(high_e_prds)) if high_e_prds else 0,
        low_energy_prd=float(np.mean(low_e_prds)) if low_e_prds else 0,
        holdout_fingerprint=fp,
        lossless_count=lossless_count,
    )
