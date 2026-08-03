#!/usr/bin/env python3
"""
LamQuant Gen 7.5 — D8: Spectral Fidelity (PSD)
===============================================
Diagnostic D8: What is the effective bandwidth of the codec?

Measures the dB error per frequency bin between the original HP-filtered EEG
and its reconstruction via the full inverse-lifting round-trip:
  raw EEG → HP filter → 3-level lifting → [TNN encode/decode] → inverse lifting → raw'

Key question: up to what frequency does the codec faithfully reconstruct the
signal within 3 dB? Below that threshold (f_3dB), clinical algorithms using
spectral features (delta, theta, alpha, beta) are unaffected by the codec.

PASS: f_3dB > 30 Hz (codec covers the clinically-important 0-30 Hz range).

Note: Because the TNN only reconstructs the L3 approximation band (0-15.625 Hz),
the detail subbands (L1, L2, L3 detail) pass through unchanged. In a lossless
encode of the details, the reconstruction above 15.625 Hz is exact. In the
TNN-only mode tested here, only the L3 approx is processed by the model;
details are passed through analytically, so high-frequency fidelity reflects
reconstruction of full lifting subbands.

Usage:
  python benchmark_spectral_fidelity.py
  python benchmark_spectral_fidelity.py --checkpoint path/to/model.ckpt
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


PATIENTS = {
    'chb15': 'chbmit_chb15_01_q31.npz',
    'chb16': 'chbmit_chb16_01_q31.npz',
    'chb17': 'chbmit_chb17a_03_q31.npz',
    'chb18': 'chbmit_chb18_01_q31.npz',
    'chb19': 'chbmit_chb19_01_q31.npz',
    'chb20': 'chbmit_chb20_01_q31.npz',
}

# Clinical frequency bands for reporting
BANDS = {
    'sub-delta': (0.0,   1.0),
    'delta':     (1.0,   4.0),
    'theta':     (4.0,   8.0),
    'alpha':     (8.0,  12.0),
    'low-beta':  (12.0, 16.0),
    'high-beta': (16.0, 25.0),
    'gamma':     (25.0, 50.0),
}

FS = 250.0
WIN_LEN = 2500
N_CHANNELS = 21
MAX_WINDOWS = 20
Q31_SCALE = 2147483647.0
MV_SCALE = 1000.0
NPERSEG = 512
NOVERLAP = 256
SMOOTH_BINS = 3    # moving-average smoothing window for f_3dB detection
F_3DB_THRESHOLD = 3.0   # dB
PASS_F3DB_HZ = 30.0


def hp_filter_batch(signal_ch21, fs=FS, fc=0.5):
    """HP filter all 21 channels. signal_ch21: [21, T]. Returns [21, T] float64."""
    sos = butter(2, fc, btype='high', fs=fs, output='sos')
    out = np.empty_like(signal_ch21)
    for ch in range(signal_ch21.shape[0]):
        out[ch] = sosfilt(sos, signal_ch21[ch])
    return out


def reconstruct_windows(model, npz_path, device, max_windows=MAX_WINDOWS):
    """Full round-trip for each window.

    Returns:
        originals:      list of [21, 2500] — HP-filtered EEG
        reconstructeds: list of [21, 2500] — after TNN + inverse lifting
    """
    with np.load(npz_path) as npz:
        raw_data = npz['data']
        n_l3 = npz['l3'].shape[0]

    total_samples = raw_data.shape[1]
    n_win = min(n_l3, max_windows)
    win_indices = np.linspace(0, n_l3 - 1, n_win, dtype=int)

    originals = []
    reconstructeds = []

    for wi in win_indices:
        start = int(wi) * WIN_LEN
        end = start + WIN_LEN
        if end > total_samples:
            continue

        raw_slice = raw_data[:, start:end].astype(np.float64) / Q31_SCALE * MV_SCALE
        hp_signal = hp_filter_batch(raw_slice)

        subs_per_ch = []
        l3_all = np.zeros((N_CHANNELS, 313), dtype=np.float64)
        for ch in range(N_CHANNELS):
            subs = lifting_3level_forward(hp_signal[ch])
            subs_per_ch.append(subs)
            l3_all[ch] = subs['l3_approx']

        l3_tensor = torch.from_numpy(l3_all[np.newaxis].astype(np.float32)).to(device)
        with torch.no_grad():
            l3_recon = model(l3_tensor, quantize=True)
        l3_recon_np = l3_recon[0].cpu().numpy().astype(np.float64)

        recon_raw = np.zeros((N_CHANNELS, WIN_LEN), dtype=np.float64)
        for ch in range(N_CHANNELS):
            subs_recon = dict(subs_per_ch[ch])
            subs_recon['l3_approx'] = l3_recon_np[ch]
            recon_raw[ch] = lifting_3level_inverse(subs_recon)[:WIN_LEN]

        originals.append(hp_signal)
        reconstructeds.append(recon_raw)

    return originals, reconstructeds


def compute_spectral_db_error(originals, reconstructeds, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP):
    """Compute mean dB error per frequency bin across all windows and channels.

    dB_error[f] = 10 * log10((psd_recon[f] + eps) / (psd_orig[f] + eps))

    Returns:
        freqs: [F] frequency bins
        mean_db_error: [F] mean dB error across patients/channels/windows
        mean_psd_orig: [F] mean original PSD
        mean_psd_recon: [F] mean reconstructed PSD
    """
    all_psd_orig = []
    all_psd_recon = []
    freq_bins = None

    for orig, recon in zip(originals, reconstructeds):
        for ch in range(orig.shape[0]):
            f, psd_orig = welch(orig[ch], fs=fs, nperseg=nperseg, noverlap=noverlap)
            _, psd_recon = welch(recon[ch], fs=fs, nperseg=nperseg, noverlap=noverlap)
            all_psd_orig.append(psd_orig)
            all_psd_recon.append(psd_recon)
            if freq_bins is None:
                freq_bins = f

    mean_psd_orig = np.mean(all_psd_orig, axis=0)
    mean_psd_recon = np.mean(all_psd_recon, axis=0)
    db_error = 10.0 * np.log10((mean_psd_recon + 1e-30) / (mean_psd_orig + 1e-30))

    return freq_bins, db_error, mean_psd_orig, mean_psd_recon


def find_f3db(freqs, db_error, smooth_bins=SMOOTH_BINS, threshold=F_3DB_THRESHOLD):
    """Find lowest frequency where |smoothed dB error| > threshold.

    Returns:
        f_3db: float (Hz), or None if always within threshold
    """
    # Simple moving average smoothing
    kernel = np.ones(smooth_bins) / smooth_bins
    smoothed = np.convolve(np.abs(db_error), kernel, mode='same')

    for i, (f, err) in enumerate(zip(freqs, smoothed)):
        if f < 0.3:   # skip DC / sub-HP-cutoff noise
            continue
        if err > threshold:
            return float(f)

    return None   # never exceeds threshold → full bandwidth codec


def band_mean_db_error(freqs, db_error):
    """Compute mean |dB error| per clinical band."""
    result = {}
    for band, (f_lo, f_hi) in BANDS.items():
        mask = (freqs >= f_lo) & (freqs < f_hi)
        if np.any(mask):
            result[band] = float(np.mean(db_error[mask]))
    return result


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
        print("[SKIP] D8 Spectral Fidelity requires a trained student checkpoint.")
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

    print(f"[*] D8: Spectral Fidelity (PSD) on {device}")
    print(f"    Checkpoint: {checkpoint_path}")
    print(f"    Patients  : {', '.join(sorted(available.keys()))}")
    print(f"    nperseg={NPERSEG}, noverlap={NOVERLAP}, fs={FS} Hz")
    print(f"    f_3dB threshold: |dB error| > {F_3DB_THRESHOLD} dB (smoothed with {SMOOTH_BINS}-bin MA)")
    print(f"    PASS criterion: f_3dB > {PASS_F3DB_HZ} Hz")
    print()

    # Accumulate dB error curves across all patients
    all_psd_orig_acc = []
    all_psd_recon_acc = []
    freq_bins = None

    for patient, path in sorted(available.items()):
        originals, reconstructeds = reconstruct_windows(model, path, device)
        if not originals:
            print(f"  {patient}: no valid windows, skipping")
            continue

        for orig, recon in zip(originals, reconstructeds):
            for ch in range(orig.shape[0]):
                f, psd_orig = welch(orig[ch], fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
                _, psd_recon = welch(recon[ch], fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
                all_psd_orig_acc.append(psd_orig)
                all_psd_recon_acc.append(psd_recon)
                if freq_bins is None:
                    freq_bins = f

        print(f"  {patient}: {len(originals)} windows processed")

    if not all_psd_orig_acc:
        print("[SKIP] No data to process.")
        return None

    mean_psd_orig = np.mean(all_psd_orig_acc, axis=0)
    mean_psd_recon = np.mean(all_psd_recon_acc, axis=0)
    db_error = 10.0 * np.log10((mean_psd_recon + 1e-30) / (mean_psd_orig + 1e-30))

    band_db = band_mean_db_error(freq_bins, db_error)
    f_3db = find_f3db(freq_bins, db_error)

    # === Report ===
    print()
    print("=" * 78)
    print(" D8: SPECTRAL FIDELITY (PSD)")
    print(f" dB error = 10*log10(PSD_recon / PSD_orig) averaged across patients & channels")
    print("=" * 78)
    print()
    print(f"  {'Band':<12} {'Freq (Hz)':<14} {'Mean dB Error':<16} {'Interpretation'}")
    print(f"  {'-'*70}")

    for band, (f_lo, f_hi) in BANDS.items():
        db = band_db.get(band, float('nan'))
        if abs(db) < 1.0:
            interp = "excellent (<1 dB)"
        elif abs(db) < 3.0:
            interp = "good (<3 dB)"
        elif abs(db) < 6.0:
            interp = "fair (<6 dB)"
        else:
            interp = "degraded (>=6 dB)"
        print(f"  {band:<12} {f_lo:4.1f} - {f_hi:4.1f} Hz   {db:>+8.2f} dB       {interp}")

    print(f"  {'-'*70}")
    print()

    if f_3db is not None:
        print(f"  f_3dB = {f_3db:.1f} Hz  (lowest frequency where |dB error| > {F_3DB_THRESHOLD} dB, "
              f"{SMOOTH_BINS}-bin smoothed)")
    else:
        print(f"  f_3dB = > {freq_bins[-1]:.1f} Hz  (never exceeded {F_3DB_THRESHOLD} dB threshold)")
        f_3db = float(freq_bins[-1])

    print()

    effective_bw = f_3db
    if effective_bw > PASS_F3DB_HZ:
        print(f"  [PASS] f_3dB={effective_bw:.1f} Hz > {PASS_F3DB_HZ:.0f} Hz — codec has adequate "
              f"spectral fidelity for clinical EEG analysis.")
        passed = True
    else:
        print(f"  [FAIL] f_3dB={effective_bw:.1f} Hz <= {PASS_F3DB_HZ:.0f} Hz — codec loses "
              f"fidelity too early in the spectrum. Check L3 reconstruction quality.")
        passed = False

    print()
    print("=" * 78)

    return {
        'passed': passed,
        'f_3dB_hz': effective_bw,
        'per_band_db_error': band_db,
        'freqs': freq_bins,
        'db_error': db_error,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()
    result = run(args.checkpoint)
    if result and result.get('passed') is not None:
        sys.exit(0 if result['passed'] else 1)
