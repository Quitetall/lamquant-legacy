"""
LamQuant Quality Standard (LQS) v1.0  [DEPRECATED — Rust is canonical]
=====================================
.. deprecated::
    The canonical LQS implementation is now the Rust ``lqs`` crate in
    Eagle (https://github.com/Quitetall/Eagle, ``lqs/``). The Rust core
    is faster (CI/pre-commit) and vendor-neutral. Rust↔Python numeric
    parity is frozen + enforced by Eagle's ``lqs/tests/metrics_parity.rs``
    against goldens this module generates, so the two agree to 1e-9.

    This Python module is RETAINED only for: (1) backward compatibility
    with existing callers, and (2) generating the frozen parity goldens
    (``Eagle/tools/parity/gen_golden.py`` imports from here). Do NOT add
    new features here — extend the Rust crate and re-freeze the goldens.

    Note: the Rust L tier intentionally diverges (LQS-L = PRD exactly 0.0
    on the integer sample domain + min_cr 0.8). C/M/A tiers match
    field-for-field.

An open standard for evaluating EEG compression quality.

Any codec — neural, classical, hybrid — can declare compliance
with an LQS quality level by passing the standard test suite
against the standard holdout dataset.

LQS levels:
  LQS-L    Lossless (bit-exact reconstruction)
  LQS-C    Clinical (neurologist cannot distinguish)
  LQS-M    Monitoring (automated analysis preserved)
  LQS-A    Alerting (event detection preserved)

A codec declares: "I am LQS-C compliant at CR=42:1"
The test suite verifies or rejects the claim.

No self-reported numbers. No cherry-picked patients.
Standardized holdout, standardized metrics, pass/fail.

Usage:
    from lamquant_codec.lqs import run_compliance, LQS_LEVELS

    result = run_compliance(
        codec_encode=my_encoder.encode,
        codec_decode=my_decoder.decode,
        level='C',
    )
    print(result.badge())
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# Canonical LQS is now the Rust crate in Eagle. This Python copy is
# kept for back-compat + frozen-golden parity generation only.
# Suppressed during golden generation (LQS_SILENCE_DEPRECATION=1).
import os as _os
if _os.environ.get("LQS_SILENCE_DEPRECATION") != "1":
    warnings.warn(
        "lamquant_codec.lqs is deprecated; the canonical LQS implementation "
        "is the Rust `lqs` crate in Eagle (github.com/Quitetall/Eagle). "
        "This module is retained for back-compat + parity-golden generation.",
        DeprecationWarning,
        stacklevel=2,
    )


# The LQS specification version this mirror tracks (Eagle ``lqs/SPEC/
# LQS-v1.0.md``). Stamped onto emitted results so a Python-side compliance
# record is self-identifying, matching the Rust ``lqs::SPEC_VERSION``. Per
# the spec §11 a change to any tier threshold / metric / the L-tier rule is a
# major bump; the canonical thresholds live in ``evaluation/openecs/src/levels.rs``.
# NOTE: ``TaskRequirement`` (downstream task preservation) is INFORMATIVE
# only in v1.0 — the task-concordance axis is reported separately and never
# gates an L/C/M/A tier (spec §10).
LQS_SPEC_VERSION: str = "1.0"


# ────────────────────────────────────────────────────────────────────
# Standard types
# ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BandRequirement:
    """Per-frequency-band quality requirement."""
    freq_range: Tuple[float, float]
    max_prd: float
    min_r: float


@dataclass(frozen=True)
class TaskRequirement:
    """Downstream task preservation requirement."""
    task: str
    metric: str
    max_degradation: float


@dataclass(frozen=True)
class LQSLevel:
    """One level of the LamQuant Quality Standard."""
    name: str
    level: str
    version: str = "1.0"
    max_prd: float = 0.0
    min_r: float = 1.0
    max_snr_loss: float = 0.0
    min_cr: float = 1.0
    band_fidelity: Dict[str, BandRequirement] = field(default_factory=dict)
    max_psd_divergence: float = 0.0
    max_coherence_loss: float = 0.0
    downstream: Dict[str, TaskRequirement] = field(default_factory=dict)
    max_encode_latency_ms: float = 1000.0
    bit_exact: bool = False


# ────────────────────────────────────────────────────────────────────
# THE STANDARD LEVELS
# ────────────────────────────────────────────────────────────────────

LQS_LEVELS: Dict[str, LQSLevel] = {

    'L': LQSLevel(
        name='Lossless', level='L',
        max_prd=0.0, min_r=1.0, max_snr_loss=0.0, min_cr=2.0,
        max_psd_divergence=0.0, max_coherence_loss=0.0,
        max_encode_latency_ms=1000.0, bit_exact=True,
    ),

    'C': LQSLevel(
        name='Clinical', level='C',
        max_prd=9.0, min_r=0.95, max_snr_loss=3.0, min_cr=20.0,
        band_fidelity={
            'delta':  BandRequirement((0.5, 4.0),   5.0,  0.98),
            'theta':  BandRequirement((4.0, 8.0),   7.0,  0.97),
            'alpha':  BandRequirement((8.0, 13.0),  8.0,  0.96),
            'beta':   BandRequirement((13.0, 30.0), 12.0, 0.93),
            'gamma':  BandRequirement((30.0, 50.0), 20.0, 0.85),
        },
        max_psd_divergence=0.05, max_coherence_loss=0.10,
        downstream={
            'seizure_detection': TaskRequirement('seizure_detection', 'sensitivity', 0.02),
            'sleep_staging':     TaskRequirement('sleep_staging', 'kappa', 0.05),
            'pathology':         TaskRequirement('pathology_detection', 'auc', 0.03),
        },
        max_encode_latency_ms=500.0, bit_exact=False,
    ),

    'M': LQSLevel(
        name='Monitoring', level='M',
        max_prd=20.0, min_r=0.85, max_snr_loss=6.0, min_cr=100.0,
        band_fidelity={
            'delta':  BandRequirement((0.5, 4.0),   10.0, 0.95),
            'theta':  BandRequirement((4.0, 8.0),   12.0, 0.93),
            'alpha':  BandRequirement((8.0, 13.0),  15.0, 0.90),
            'beta':   BandRequirement((13.0, 30.0), 25.0, 0.80),
            'gamma':  BandRequirement((30.0, 50.0), 40.0, 0.60),
        },
        max_psd_divergence=0.15, max_coherence_loss=0.20,
        downstream={
            'seizure_detection': TaskRequirement('seizure_detection', 'sensitivity', 0.05),
            'sleep_staging':     TaskRequirement('sleep_staging', 'kappa', 0.10),
        },
        max_encode_latency_ms=1000.0, bit_exact=False,
    ),

    'A': LQSLevel(
        name='Alerting', level='A',
        max_prd=40.0, min_r=0.70, max_snr_loss=10.0, min_cr=200.0,
        band_fidelity={
            'delta':  BandRequirement((0.5, 4.0),   20.0, 0.85),
            'theta':  BandRequirement((4.0, 8.0),   25.0, 0.80),
            'alpha':  BandRequirement((8.0, 13.0),  30.0, 0.75),
            'beta':   BandRequirement((13.0, 30.0), 40.0, 0.65),
            'gamma':  BandRequirement((30.0, 50.0), 60.0, 0.40),
        },
        max_psd_divergence=0.30, max_coherence_loss=0.35,
        downstream={
            'seizure_detection': TaskRequirement('seizure_detection', 'sensitivity', 0.10),
        },
        max_encode_latency_ms=5000.0, bit_exact=False,
    ),
}


# ────────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────────

def prd(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Percent Root-mean-square Difference."""
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    denom = np.sum(original.astype(np.float64) ** 2)
    if denom < 1e-30:
        return 0.0 if np.allclose(diff, 0) else 100.0
    return float(100.0 * np.sqrt(np.sum(diff ** 2) / denom))


def pearson_r(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Pearson correlation (flattened)."""
    x = original.flatten().astype(np.float64)
    y = reconstructed.flatten().astype(np.float64)
    if x.std() < 1e-10 or y.std() < 1e-10:
        return 1.0 if np.allclose(x, y) else 0.0
    return float(np.corrcoef(x, y)[0, 1])


def snr_db(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Signal-to-noise ratio in dB."""
    sig = np.mean(original.astype(np.float64) ** 2)
    noise = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    if noise < 1e-30:
        return 120.0
    return float(10 * np.log10(sig / noise))


def band_metrics(original: np.ndarray, reconstructed: np.ndarray,
                 freq_range: Tuple[float, float], sr: float = 250.0) -> Tuple[float, float]:
    """PRD and R on a bandpass-filtered version."""
    from scipy.signal import butter, sosfiltfilt
    lo, hi = freq_range
    if hi >= sr / 2:
        hi = sr / 2 - 1
    sos = butter(4, (lo, hi), btype='bandpass', fs=sr, output='sos')
    of = sosfiltfilt(sos, original.astype(np.float64), axis=-1)
    rf = sosfiltfilt(sos, reconstructed.astype(np.float64), axis=-1)
    return prd(of, rf), pearson_r(of, rf)


def psd_divergence(original: np.ndarray, reconstructed: np.ndarray,
                   sr: float = 250.0) -> float:
    """KL divergence of power spectral density."""
    from scipy.signal import welch
    nperseg = min(256, max(16, original.size // 4))
    _, po = welch(original.flatten().astype(np.float64), fs=sr, nperseg=nperseg)
    _, pr = welch(reconstructed.flatten().astype(np.float64), fs=sr, nperseg=nperseg)
    po = po / (po.sum() + 1e-30)
    pr = pr / (pr.sum() + 1e-30)
    return float(max(0, np.sum(po * np.log((po + 1e-10) / (pr + 1e-10)))))


def compute_all_metrics(original: np.ndarray, reconstructed: np.ndarray,
                        packet: bytes, sr: float = 250.0) -> dict:
    """All LQS metrics for one window."""
    return {
        'prd': prd(original, reconstructed),
        'r': pearson_r(original, reconstructed),
        'snr_db': snr_db(original, reconstructed),
        'cr': original.size * 2 / max(len(packet), 1),
        'psd_divergence': psd_divergence(original, reconstructed, sr),
        'bit_exact': bool(np.array_equal(original, reconstructed)),
    }


# ────────────────────────────────────────────────────────────────────
# Contract checking
# ────────────────────────────────────────────────────────────────────

def check_contract(metrics: dict, level: LQSLevel,
                   original: np.ndarray = None,
                   reconstructed: np.ndarray = None,
                   sr: float = 250.0) -> List[str]:
    """Check one window against a level. Returns violations (empty = pass)."""
    v = []

    if level.bit_exact and not metrics.get('bit_exact'):
        v.append("bit_exact: required")
    if metrics['prd'] > level.max_prd:
        v.append(f"prd: {metrics['prd']:.2f}% > {level.max_prd}%")
    # For lossless (min_r=1.0), bit_exact covers it. Otherwise check R with epsilon.
    if level.bit_exact:
        pass  # bit_exact check above is sufficient
    elif metrics['r'] < level.min_r - 1e-6:
        v.append(f"r: {metrics['r']:.4f} < {level.min_r}")
    if metrics['cr'] < level.min_cr:
        v.append(f"cr: {metrics['cr']:.1f} < {level.min_cr}")
    if metrics.get('psd_divergence', 0) > level.max_psd_divergence > 0:
        v.append(f"psd_div: {metrics['psd_divergence']:.4f} > {level.max_psd_divergence}")
    if metrics.get('encode_latency_ms', 0) > level.max_encode_latency_ms:
        v.append(f"latency: {metrics['encode_latency_ms']:.0f}ms > {level.max_encode_latency_ms}ms")

    if original is not None and reconstructed is not None:
        for band_name, req in level.band_fidelity.items():
            try:
                bp, br = band_metrics(original, reconstructed, req.freq_range, sr)
                if bp > req.max_prd:
                    v.append(f"band_{band_name}_prd: {bp:.1f}% > {req.max_prd}%")
                if br < req.min_r:
                    v.append(f"band_{band_name}_r: {br:.4f} < {req.min_r}")
            except Exception as exc:
                v.append(f"band_{band_name}_error: measurement failed ({exc})")
    return v


# ────────────────────────────────────────────────────────────────────
# Result
# ────────────────────────────────────────────────────────────────────

@dataclass
class ComplianceResult:
    """LQS compliance test result."""
    passed: bool
    level: str
    version: str = "1.0"
    mean_cr: float = 0.0
    mean_prd: float = 0.0
    mean_r: float = 0.0
    mean_snr_db: float = 0.0
    n_windows: int = 0
    n_violations: int = 0
    violations: List[str] = field(default_factory=list)
    band_results: Dict[str, dict] = field(default_factory=dict)
    total_encode_ms: float = 0.0
    total_decode_ms: float = 0.0
    wall_time_s: float = 0.0
    codec_name: str = ""
    dataset: str = ""
    n_subjects: int = 0
    n_files: int = 0
    timestamp: str = ""

    def badge(self) -> str:
        from lamquant_codec.cli.box import Box
        status = "COMPLIANT" if self.passed else "FAILED"
        box = Box(title=f"LQS-{self.level} {status}", width=50)
        box.line(self.codec_name)
        box.line(f"CR: {self.mean_cr:.0f}:1  |  PRD: {self.mean_prd:.1f}%  |  R: {self.mean_r:.3f}")
        box.line(f"Dataset: {self.dataset}  ({self.n_files} files)")
        box.line(f"Date: {self.timestamp[:10]}  |  LQS v{self.version}")
        return box.render()

    def to_dict(self) -> dict:
        return asdict(self)


# ────────────────────────────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────────────────────────────

def run_compliance(
    codec_encode: Callable,
    codec_decode: Callable,
    level: str = 'L',
    signals: Optional[List[np.ndarray]] = None,
    dataset_name: str = "synthetic",
    codec_name: str = "unknown",
    sr: float = 250.0,
) -> ComplianceResult:
    """Run the full LQS compliance test suite.

    Args:
        codec_encode: callable(signal[C, T]) -> bytes
        codec_decode: callable(bytes) -> signal[C, T]
        level: 'L', 'C', 'M', or 'A'
        signals: test signals. If None, generates 50 synthetic windows.
        dataset_name: for the badge
        codec_name: for the badge
        sr: sample rate

    Returns:
        ComplianceResult — passed/failed with full metrics.
    """
    contract = LQS_LEVELS[level]
    if signals is None:
        signals = _synthetic_signals()

    # JIT warmup — don't count compilation in latency
    try:
        warmup = signals[0]
        codec_decode(codec_encode(warmup))
    except Exception:
        pass

    all_violations = []
    all_metrics = []
    t_enc = t_dec = 0.0
    t0 = time.time()

    for sig in signals:
        sig = np.asarray(sig)

        te = time.perf_counter()
        packet = codec_encode(sig)
        enc_ms = (time.perf_counter() - te) * 1000
        t_enc += enc_ms

        td = time.perf_counter()
        recon = np.asarray(codec_decode(packet))
        dec_ms = (time.perf_counter() - td) * 1000
        t_dec += dec_ms

        m = compute_all_metrics(sig, recon, packet, sr)
        m['encode_latency_ms'] = enc_ms

        v = check_contract(m, contract, sig, recon, sr)
        all_violations.extend(v)
        all_metrics.append(m)

    unique_v = list(dict.fromkeys(all_violations))

    return ComplianceResult(
        passed=len(unique_v) == 0,
        level=level,
        mean_cr=float(np.mean([m['cr'] for m in all_metrics])) if all_metrics else 0,
        mean_prd=float(np.mean([m['prd'] for m in all_metrics])) if all_metrics else 0,
        mean_r=float(np.mean([m['r'] for m in all_metrics])) if all_metrics else 0,
        mean_snr_db=float(np.mean([m['snr_db'] for m in all_metrics])) if all_metrics else 0,
        n_windows=len(signals),
        n_violations=len(unique_v),
        violations=unique_v,
        total_encode_ms=t_enc,
        total_decode_ms=t_dec,
        wall_time_s=time.time() - t0,
        codec_name=codec_name,
        dataset=dataset_name,
        n_files=len(signals),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _synthetic_signals(n=50, ch=21, t=2500):
    """Generate EEG-like test signals."""
    rng = np.random.RandomState(42)
    signals = []
    for _ in range(n):
        ts = np.linspace(0, 10, t)
        sig = np.zeros((ch, t))
        for c in range(ch):
            sig[c] += rng.randn() * 50 * np.sin(2 * np.pi * rng.uniform(0.5, 4) * ts)
            sig[c] += rng.randn() * 30 * np.sin(2 * np.pi * rng.uniform(4, 8) * ts)
            sig[c] += rng.randn() * 20 * np.sin(2 * np.pi * rng.uniform(8, 13) * ts)
            sig[c] += rng.randn() * 10 * np.sin(2 * np.pi * rng.uniform(13, 30) * ts)
            sig[c] += rng.randn(t) * 5
        signals.append(sig.astype(np.int16))
    return signals
