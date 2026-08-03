#!/usr/bin/env python3
"""
LamQuant Gen 7.5 — D1: Fullband Error Heatmap
==============================================
Diagnostic D1: Does TNN reconstruction error spike at Le Gall lifting boundaries
when measured in the FULLBAND (raw EEG) domain via inverse lifting?

Round-trip: raw EEG → HP filter → 3-level lifting → [TNN encode/decode L3] →
            inverse lifting → reconstructed raw EEG

The subband boundaries are at:
  L1 boundary: 62.5  Hz  (fs/4)
  L2 boundary: 31.25 Hz  (fs/8)
  L3 boundary: 15.625 Hz (fs/16)

Clinical frequency bands:
  sub-delta:  0-1  Hz
  delta:      1-4  Hz
  theta:      4-8  Hz
  alpha:      8-12 Hz
  low-beta:  12-16 Hz  <-- straddles the L3 lifting boundary at 15.625 Hz
  high-beta: 16-25 Hz
  gamma:     25-50 Hz

PASS: SNR deficit at boundary zone (12-20 Hz) < 6 dB vs mean of other bands.

Usage:
  python benchmark_fullband_error_heatmap.py
  python benchmark_fullband_error_heatmap.py --checkpoint path/to/model.ckpt
"""

import torch
import os
import sys
import numpy as np
from scipy.signal import welch, butter, sosfilt
from pathlib import Path


def find_project_root(marker='.git'):
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / marker).exists():
            return str(parent)
    raise RuntimeError("Project root not found")


ROOT_DIR = find_project_root()
sys.path.append(os.path.join(ROOT_DIR, 'ai_models', 'student'))
from lamquant_codec.models.encoder import TernaryMobileNetV5_Subband
from subband_preprocess import lifting_3level_forward, lifting_3level_inverse


# Le Gall 5/3 subband boundaries (Hz at fs=250)
LIFTING_BOUNDARIES_HZ = [15.625, 31.25, 62.5]

# Clinical frequency bands (fullband, 0-125 Hz Nyquist)
BANDS = {
    'sub-delta': (0.0,   1.0),
    'delta':     (1.0,   4.0),
    'theta':     (4.0,   8.0),
    'alpha':     (8.0,  12.0),
    'low-beta':  (12.0, 16.0),   # straddles L3 boundary at 15.625 Hz
    'high-beta': (16.0, 25.0),
    'gamma':     (25.0, 50.0),
}

# Boundary zone for pass/fail evaluation
BOUNDARY_BANDS = {'low-beta'}   # 12-16 Hz

PATIENTS = {
    'chb15': 'chbmit_chb15_01_q31.npz',
    'chb16': 'chbmit_chb16_01_q31.npz',
    'chb17': 'chbmit_chb17a_03_q31.npz',
    'chb18': 'chbmit_chb18_01_q31.npz',
    'chb19': 'chbmit_chb19_01_q31.npz',
    'chb20': 'chbmit_chb20_01_q31.npz',
}

FS = 250.0
WIN_LEN = 2500       # 10 seconds at 250 Hz
N_CHANNELS = 21
MAX_WINDOWS = 20
Q31_SCALE = 2147483647.0
MV_SCALE = 1000.0


def hp_filter_batch(signal_ch21, fs=FS, fc=0.5):
    """HP filter all 21 channels. signal_ch21: [21, T]. Returns [21, T] float64."""
    sos = butter(2, fc, btype='high', fs=fs, output='sos')
    out = np.empty_like(signal_ch21)
    for ch in range(signal_ch21.shape[0]):
        out[ch] = sosfilt(sos, signal_ch21[ch])
    return out


def reconstruct_windows(model, npz_path, device, max_windows=MAX_WINDOWS):
    """Load raw windows from npz, run full round-trip, return (originals, reconstructeds).

    Returns:
        originals:      list of [21, 2500] float64 — HP-filtered raw EEG
        reconstructeds: list of [21, 2500] float64 — after TNN + inverse lifting
    """
    with np.load(npz_path) as npz:
        raw_data = npz['data']        # [21, N_samples] int32
        l3_shape = npz['l3'].shape    # [N_win, 21, 313]

    n_l3 = l3_shape[0]
    total_samples = raw_data.shape[1]

    # Evenly-spaced window indices over available L3 windows
    n_win = min(n_l3, max_windows)
    win_indices = np.linspace(0, n_l3 - 1, n_win, dtype=int)

    originals = []
    reconstructeds = []

    for wi in win_indices:
        start = int(wi) * WIN_LEN
        end = start + WIN_LEN
        if end > total_samples:
            continue

        # Q31 → float mV
        raw_slice = raw_data[:, start:end].astype(np.float64) / Q31_SCALE * MV_SCALE  # [21, 2500]

        # HP filter
        hp_signal = hp_filter_batch(raw_slice)  # [21, 2500]

        # Forward lifting: collect l3_approx for all channels → [21, 313]
        subs_per_ch = []
        l3_all = np.zeros((N_CHANNELS, 313), dtype=np.float64)
        for ch in range(N_CHANNELS):
            subs = lifting_3level_forward(hp_signal[ch])
            subs_per_ch.append(subs)
            l3_all[ch] = subs['l3_approx']

        # TNN forward: [1, 21, 313] → [1, 21, 313]
        l3_tensor = torch.from_numpy(l3_all[np.newaxis].astype(np.float32)).to(device)
        with torch.no_grad():
            l3_recon = model(l3_tensor, quantize=True)  # [1, 21, 313]
        l3_recon_np = l3_recon[0].cpu().numpy().astype(np.float64)  # [21, 313]

        # Inverse lifting per channel: replace l3_approx with reconstructed
        recon_raw = np.zeros((N_CHANNELS, WIN_LEN), dtype=np.float64)
        for ch in range(N_CHANNELS):
            subs_recon = dict(subs_per_ch[ch])
            subs_recon['l3_approx'] = l3_recon_np[ch]
            recon_raw[ch] = lifting_3level_inverse(subs_recon)[:WIN_LEN]

        originals.append(hp_signal)
        reconstructeds.append(recon_raw)

    return originals, reconstructeds


def compute_band_snr(originals, reconstructeds, fs=FS, nperseg=512, noverlap=256):
    """Compute per-band SNR across all windows and channels.

    Returns:
        band_snr: dict {band_name: mean_SNR_dB}
        band_psd_orig: dict {band_name: mean_psd_orig}
        band_psd_error: dict {band_name: mean_psd_error}
    """
    all_sig_psd = []
    all_err_psd = []
    freq_bins = None

    for orig, recon in zip(originals, reconstructeds):
        error = orig - recon  # [21, 2500]
        for ch in range(orig.shape[0]):
            f, psd_sig = welch(orig[ch], fs=fs, nperseg=nperseg, noverlap=noverlap)
            _, psd_err = welch(error[ch], fs=fs, nperseg=nperseg, noverlap=noverlap)
            all_sig_psd.append(psd_sig)
            all_err_psd.append(psd_err)
            if freq_bins is None:
                freq_bins = f

    mean_sig_psd = np.mean(all_sig_psd, axis=0)
    mean_err_psd = np.mean(all_err_psd, axis=0)

    band_snr = {}
    band_psd_orig = {}
    band_psd_error = {}

    for band_name, (f_lo, f_hi) in BANDS.items():
        mask = (freq_bins >= f_lo) & (freq_bins < f_hi)
        if not np.any(mask):
            continue
        sig_power = float(np.mean(mean_sig_psd[mask]))
        err_power = float(np.mean(mean_err_psd[mask]))
        snr = 10.0 * np.log10(sig_power / (err_power + 1e-30))
        band_snr[band_name] = snr
        band_psd_orig[band_name] = sig_power
        band_psd_error[band_name] = err_power

    return band_snr, band_psd_orig, band_psd_error, freq_bins, mean_sig_psd, mean_err_psd


def run(checkpoint_path=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Checkpoint discovery
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
        print("[SKIP] D1 Fullband Error Heatmap requires a trained student checkpoint.")
        return None

    # Patient data discovery
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

    print(f"[*] D1: Fullband Error Heatmap on {device}")
    print(f"    Checkpoint: {checkpoint_path}")
    print(f"    Patients  : {', '.join(sorted(available.keys()))}")
    print(f"    Boundaries: {LIFTING_BOUNDARIES_HZ} Hz (Le Gall 5/3, 3 levels)")
    print(f"    Windows   : {MAX_WINDOWS} per patient, stride=2500 samples")
    print()

    # Accumulate band SNR across patients
    all_band_snr = {b: [] for b in BANDS}

    for patient, path in sorted(available.items()):
        originals, reconstructeds = reconstruct_windows(model, path, device)
        if not originals:
            print(f"  {patient}: no valid windows, skipping")
            continue

        band_snr, _, _, _, _, _ = compute_band_snr(originals, reconstructeds)

        pat_parts = []
        for band in BANDS:
            snr = band_snr.get(band, float('nan'))
            all_band_snr[band].append(snr)
            pat_parts.append(f"{snr:+.1f}")
        print(f"  {patient}: " + "  ".join(
            f"{b}={snr:+.1f}dB" for b, snr in band_snr.items()
        ))

    # === Aggregate across patients ===
    mean_band_snr = {
        b: float(np.mean(v)) if v else float('nan')
        for b, v in all_band_snr.items()
    }

    boundary_snrs = [mean_band_snr[b] for b in BOUNDARY_BANDS if not np.isnan(mean_band_snr.get(b, float('nan')))]
    other_snrs = [mean_band_snr[b] for b in BANDS if b not in BOUNDARY_BANDS and not np.isnan(mean_band_snr.get(b, float('nan')))]

    mean_boundary_snr = float(np.mean(boundary_snrs)) if boundary_snrs else float('nan')
    mean_other_snr = float(np.mean(other_snrs)) if other_snrs else float('nan')
    snr_deficit = mean_other_snr - mean_boundary_snr

    # === Report ===
    print()
    print("=" * 78)
    print(" D1: FULLBAND ERROR HEATMAP")
    print(f" Lifting boundaries: {', '.join(f'{b} Hz' for b in LIFTING_BOUNDARIES_HZ)}")
    print("=" * 78)
    print()
    print(f"  {'Band':<12} {'Freq (Hz)':<14} {'Mean SNR (dB)':<15} {'Boundary?'}")
    print(f"  {'-'*60}")

    for band, (f_lo, f_hi) in BANDS.items():
        snr = mean_band_snr.get(band, float('nan'))
        is_boundary = '*' if band in BOUNDARY_BANDS else ''
        print(f"  {band:<12} {f_lo:4.1f} - {f_hi:4.1f} Hz   {snr:>+8.2f} dB      {is_boundary}")

    print(f"  {'-'*60}")
    print()
    print(f"  Mean SNR (non-boundary bands): {mean_other_snr:+.2f} dB")
    print(f"  Mean SNR (boundary zone 12-16 Hz): {mean_boundary_snr:+.2f} dB")
    print(f"  SNR deficit at boundary: {snr_deficit:.2f} dB  (threshold: < 6 dB)")
    print()
    print(f"  Lifting boundaries marked: L3={LIFTING_BOUNDARIES_HZ[0]} Hz, "
          f"L2={LIFTING_BOUNDARIES_HZ[1]} Hz, L1={LIFTING_BOUNDARIES_HZ[2]} Hz")
    print()

    # Diagnosis
    if np.isnan(snr_deficit):
        print("  [SKIP] Insufficient data for evaluation.")
        return None

    if snr_deficit < 6.0:
        print(f"  [PASS] SNR deficit at boundary ({snr_deficit:.2f} dB) < 6 dB — "
              f"no significant lifting boundary artifact in fullband domain.")
        passed = True
    else:
        print(f"  [FAIL] SNR deficit at boundary ({snr_deficit:.2f} dB) >= 6 dB — "
              f"lifting boundary artifact visible in fullband reconstruction error.")
        passed = False

    print()
    print("=" * 78)

    return {
        'passed': passed,
        'snr_deficit_dB': snr_deficit,
        'mean_boundary_snr': mean_boundary_snr,
        'mean_other_snr': mean_other_snr,
        'per_band_snr': mean_band_snr,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()
    result = run(args.checkpoint)
    if result and result.get('passed') is not None:
        sys.exit(0 if result['passed'] else 1)
