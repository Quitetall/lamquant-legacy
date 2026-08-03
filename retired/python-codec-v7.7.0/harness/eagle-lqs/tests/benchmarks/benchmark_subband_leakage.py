#!/usr/bin/env python3
"""
LamQuant Gen 7.5 — Subband Boundary Leakage Diagnostic
=======================================================
The single most important diagnostic for validating the subband architecture.

Tests whether TNN reconstruction error concentrates near the Le Gall 5/3
subband boundary (~15.6 Hz at 250 Hz sampling) or distributes uniformly
across the 0-15.6 Hz L3 approximation band.

Interpretation:
  - Error UNIFORM across frequency: undertraining. More epochs / hardening
    will close the gap. Architecture is sound.
  - Error CONCENTRATED at 12-16 Hz: lifting leakage. Le Gall's 10-12 dB
    stopband attenuation is splitting clinically relevant features (alpha-beta
    transition, sleep spindles) across subbands. No amount of training fixes
    this — need sharper filter (CDF 9/7) or lower boundary (4th lifting level).

Frequency bands (L3 subband, 0-15.6 Hz effective):
  sub-delta:  0.0 -  1.0 Hz
  delta:      1.0 -  4.0 Hz
  theta:      4.0 -  8.0 Hz
  alpha:      8.0 - 12.0 Hz
  boundary:  12.0 - 15.6 Hz  <-- the critical region

Usage:
  python benchmark_subband_leakage.py
  python benchmark_subband_leakage.py --checkpoint path/to/model.ckpt
"""

import torch
import os
import sys
import numpy as np
from scipy.stats import pearsonr
from scipy.signal import welch
from pathlib import Path
import pytest

# Internal LamQuant-vendor neural introspection bench — gated out of the
# external LQS suite. Run with `pytest -m internal`. See tests/internal/README.md.
pytestmark = pytest.mark.internal


def find_project_root(marker='.git'):
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / marker).exists():
            return str(parent)
    raise RuntimeError("Project root not found")


ROOT_DIR = find_project_root()
sys.path.append(os.path.join(ROOT_DIR, 'ai_models', 'student'))
from lamquant_codec.models.encoder import TernaryMobileNetV5_Subband


# Le Gall 5/3 at 3 lifting levels: fs=250, L3 effective band = 0 to fs/2^4 = 15.625 Hz
# L3 sample rate = 250/8 = 31.25 Hz (313 samples covers 10.016 seconds)
L3_FS = 31.25
SUBBAND_BOUNDARY_HZ = 15.625

# Frequency bands within L3 (all below 15.625 Hz)
BANDS = {
    'sub-delta': (0.0,  1.0),
    'delta':     (1.0,  4.0),
    'theta':     (4.0,  8.0),
    'alpha':     (8.0, 12.0),
    'boundary':  (12.0, 15.6),   # the critical region
}

# Held-out patients (not used in training)
PATIENTS = {
    'chb15': 'chbmit_chb15_01_q31.npz',
    'chb16': 'chbmit_chb16_01_q31.npz',
    'chb17': 'chbmit_chb17a_03_q31.npz',
    'chb18': 'chbmit_chb18_01_q31.npz',
    'chb19': 'chbmit_chb19_01_q31.npz',
    'chb20': 'chbmit_chb20_01_q31.npz',
}


def load_l3_windows(path, max_windows=20):
    """Load multiple L3 windows from a q31 npz file."""
    with np.load(path) as data:
        l3 = data['l3']  # [N, 21, 313]
    n = min(l3.shape[0], max_windows)
    idx = np.linspace(0, l3.shape[0] - 1, n, dtype=int)
    return l3[idx]  # [n, 21, 313]


def per_band_error(original, reconstructed, fs=L3_FS, nperseg=64):
    """Compute reconstruction error power spectrum binned by clinical frequency band.

    Returns dict of {band_name: (error_power_dB, signal_power_dB, SNR_dB)}.
    Averages across channels and windows.
    """
    error = original - reconstructed  # [n_windows, 21, 313]

    # Pool across windows and channels for robust spectral estimate
    n_win, n_ch, T = original.shape
    all_sig_psd = []
    all_err_psd = []

    for w in range(n_win):
        for ch in range(n_ch):
            f, psd_sig = welch(original[w, ch], fs=fs, nperseg=min(nperseg, T),
                               noverlap=nperseg // 2, scaling='density')
            _, psd_err = welch(error[w, ch], fs=fs, nperseg=min(nperseg, T),
                               noverlap=nperseg // 2, scaling='density')
            all_sig_psd.append(psd_sig)
            all_err_psd.append(psd_err)

    mean_sig_psd = np.mean(all_sig_psd, axis=0)
    mean_err_psd = np.mean(all_err_psd, axis=0)

    results = {}
    for band_name, (f_lo, f_hi) in BANDS.items():
        mask = (f >= f_lo) & (f < f_hi)
        if not np.any(mask):
            continue
        sig_power = np.mean(mean_sig_psd[mask])
        err_power = np.mean(mean_err_psd[mask])
        snr = 10 * np.log10(sig_power / (err_power + 1e-30))
        results[band_name] = {
            'sig_dB': 10 * np.log10(sig_power + 1e-30),
            'err_dB': 10 * np.log10(err_power + 1e-30),
            'snr_dB': snr,
            'err_frac': err_power / (np.mean(mean_err_psd) + 1e-30),
        }

    return results, f, mean_sig_psd, mean_err_psd


def per_band_pearson(original, reconstructed, fs=L3_FS, nperseg=64):
    """Compute per-band Pearson R by bandpass filtering then correlating.

    Uses FFT-based filtering for clean band isolation.
    """
    n_win, n_ch, T = original.shape
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)

    results = {}
    for band_name, (f_lo, f_hi) in BANDS.items():
        mask = (freqs >= f_lo) & (freqs < f_hi)
        if not np.any(mask):
            continue

        rs = []
        for w in range(n_win):
            for ch in range(n_ch):
                orig_fft = np.fft.rfft(original[w, ch])
                recon_fft = np.fft.rfft(reconstructed[w, ch])

                # Zero out everything outside the band
                orig_band = np.zeros_like(orig_fft)
                recon_band = np.zeros_like(recon_fft)
                orig_band[mask] = orig_fft[mask]
                recon_band[mask] = recon_fft[mask]

                # Back to time domain
                orig_t = np.fft.irfft(orig_band, n=T)
                recon_t = np.fft.irfft(recon_band, n=T)

                if np.std(orig_t) > 1e-8 and np.std(recon_t) > 1e-8:
                    r, _ = pearsonr(orig_t, recon_t)
                    rs.append(r)

        results[band_name] = np.mean(rs) if rs else 0.0

    return results


def run(checkpoint_path=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Find checkpoint
    if checkpoint_path is None:
        for candidate in [
            os.path.join(ROOT_DIR, 'ai_models/student/student_hardened.ckpt'),
            os.path.join(ROOT_DIR, 'weights/student_subband.ckpt'),
            os.path.join(ROOT_DIR, 'weights/student_subband_fast.ckpt'),
        ]:
            if os.path.exists(candidate):
                checkpoint_path = candidate
                break

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print("[SKIP] No student checkpoint found.")
        return None

    # Find patient data
    q31_dir = os.path.join(ROOT_DIR, 'ai_models/dataset_sim/q31_events')
    available = {}
    for name, fname in PATIENTS.items():
        p = os.path.join(q31_dir, fname)
        if os.path.exists(p):
            available[name] = p

    if len(available) < 3:
        print(f"[SKIP] Need at least 3 held-out patients, found {len(available)}")
        return None

    # Load model
    model = TernaryMobileNetV5_Subband.from_checkpoint(checkpoint_path, device=device)
    model.eval()
    print(f"[*] Subband Leakage Diagnostic on {device}")
    print(f"    Checkpoint: {checkpoint_path}")
    print(f"    Patients: {', '.join(sorted(available.keys()))}")
    print(f"    Le Gall 5/3 boundary: {SUBBAND_BOUNDARY_HZ} Hz")
    print()

    # Collect per-band results across all patients
    all_band_snr = {b: [] for b in BANDS}
    all_band_r = {b: [] for b in BANDS}
    all_band_err_frac = {b: [] for b in BANDS}

    for patient, path in sorted(available.items()):
        l3_windows = load_l3_windows(path, max_windows=20)
        x = torch.from_numpy(l3_windows).float().to(device)

        with torch.no_grad():
            recon = model(x, quantize=True).cpu().numpy()

        original = l3_windows

        # Per-band spectral error
        band_results, _, _, _ = per_band_error(original, recon)

        # Per-band Pearson R
        band_r = per_band_pearson(original, recon)

        for band in BANDS:
            if band in band_results:
                all_band_snr[band].append(band_results[band]['snr_dB'])
                all_band_err_frac[band].append(band_results[band]['err_frac'])
            if band in band_r:
                all_band_r[band].append(band_r[band])

    # === Report ===
    print("=" * 78)
    print(" SUBBAND BOUNDARY LEAKAGE DIAGNOSTIC")
    print(" Le Gall 5/3, 3 levels, boundary = 15.6 Hz")
    print("=" * 78)
    print()
    print(f"{'Band':<14} {'Freq (Hz)':<14} {'SNR (dB)':<12} {'R':<10} {'Err Frac':<12} {'Verdict'}")
    print("-" * 78)

    boundary_snr = None
    other_snrs = []
    boundary_r = None
    other_rs = []

    for band, (f_lo, f_hi) in BANDS.items():
        snr = np.mean(all_band_snr[band]) if all_band_snr[band] else float('nan')
        r = np.mean(all_band_r[band]) if all_band_r[band] else float('nan')
        ef = np.mean(all_band_err_frac[band]) if all_band_err_frac[band] else float('nan')

        if band == 'boundary':
            boundary_snr = snr
            boundary_r = r
            verdict = '<-- CRITICAL'
        else:
            other_snrs.append(snr)
            other_rs.append(r)
            verdict = ''

        print(f"{band:<14} {f_lo:4.1f}-{f_hi:4.1f} Hz   {snr:>7.2f} dB  {r:>7.4f}   {ef:>7.2f}x       {verdict}")

    print("-" * 78)

    # Diagnosis
    mean_other_snr = np.mean(other_snrs) if other_snrs else float('nan')
    mean_other_r = np.mean(other_rs) if other_rs else float('nan')

    print()
    print(f"  Mean non-boundary SNR:  {mean_other_snr:.2f} dB")
    print(f"  Boundary (12-16 Hz) SNR: {boundary_snr:.2f} dB")
    print(f"  SNR deficit at boundary: {mean_other_snr - boundary_snr:.2f} dB")
    print()
    print(f"  Mean non-boundary R:    {mean_other_r:.4f}")
    print(f"  Boundary (12-16 Hz) R:  {boundary_r:.4f}")
    print(f"  R deficit at boundary:  {mean_other_r - boundary_r:.4f}")
    print()

    # Interpretation
    snr_deficit = mean_other_snr - boundary_snr
    r_deficit = mean_other_r - boundary_r

    if snr_deficit > 6.0:
        diagnosis = "LEAKAGE"
        detail = (
            f"Boundary SNR is {snr_deficit:.1f} dB below average — strong evidence of "
            f"Le Gall 5/3 stopband leakage. Beta-band energy straddling 15.6 Hz is "
            f"split across subbands. Consider CDF 9/7 or 4th lifting level."
        )
        passed = False
    elif snr_deficit > 3.0:
        diagnosis = "BORDERLINE"
        detail = (
            f"Boundary SNR is {snr_deficit:.1f} dB below average — moderate boundary "
            f"concentration. Could be undertraining or mild leakage. Run production "
            f"training to distinguish."
        )
        passed = True  # not conclusive — need production run
    else:
        diagnosis = "UNIFORM"
        detail = (
            f"Boundary SNR deficit is only {snr_deficit:.1f} dB — error is uniformly "
            f"distributed. Architecture is sound. Quality will improve with more "
            f"training epochs and hardening."
        )
        passed = True

    print(f"  DIAGNOSIS: {diagnosis}")
    print(f"  {detail}")
    print()
    print("=" * 78)

    return {
        'diagnosis': diagnosis,
        'passed': passed,
        'snr_deficit_dB': snr_deficit,
        'r_deficit': r_deficit,
        'boundary_snr': boundary_snr,
        'boundary_r': boundary_r,
        'mean_other_snr': mean_other_snr,
        'mean_other_r': mean_other_r,
        'per_band_snr': {b: np.mean(v) for b, v in all_band_snr.items() if v},
        'per_band_r': {b: np.mean(v) for b, v in all_band_r.items() if v},
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()
    result = run(args.checkpoint)
    if result:
        sys.exit(0 if result['passed'] else 1)
