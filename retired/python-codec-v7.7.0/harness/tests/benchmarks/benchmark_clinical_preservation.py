#!/usr/bin/env python3
"""
LamQuant Gen 7.5 — D7: Clinical Feature Preservation at Mode 0
==============================================================
Diagnostic D7: Do clinical seizure detection algorithms agree on original
vs compressed-and-reconstructed EEG at maximum compression (Mode 0, FSQ L=8)?

Round-trip:
  raw EEG → HP filter → 3-level lifting → TNN encode → FSQ(L=8) → FSQ decode
  → TNN decode → inverse lifting → reconstructed raw EEG

Seizure detector:
  Simple band-power ratio: power(8-30 Hz) / power(0.5-4 Hz) per channel.
  Max ratio across channels per window is the detector score.

Evaluation:
  1. Find optimal threshold that maximizes F1 on ORIGINAL signal.
  2. Apply same threshold to both original and reconstructed signal.
  3. Compute sensitivity_ratio = sensitivity(reconstructed) / sensitivity(original).

PASS: sensitivity_ratio >= 0.80 (codec preserves >= 80% of detector sensitivity).
SKIP: If no seizure windows available in holdout data.

Usage:
  python benchmark_clinical_preservation.py
  python benchmark_clinical_preservation.py --checkpoint path/to/model.ckpt
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
sys.path.insert(0, os.path.dirname(__file__))
from lamquant_codec.models.encoder import TernaryMobileNetV5_Subband
from subband_preprocess import lifting_3level_forward, lifting_3level_inverse
from benchmark_compression_ratio import fsq_encode, fsq_decode


PATIENTS = {
    'chb15': 'chbmit_chb15_01_q31.npz',
    'chb16': 'chbmit_chb16_01_q31.npz',
    'chb17': 'chbmit_chb17a_03_q31.npz',
    'chb18': 'chbmit_chb18_01_q31.npz',
    'chb19': 'chbmit_chb19_01_q31.npz',
    'chb20': 'chbmit_chb20_01_q31.npz',
}

FS = 250.0
WIN_LEN = 2500
N_CHANNELS = 21
Q31_SCALE = 2147483647.0
MV_SCALE = 1000.0
FSQ_L = 8          # Mode 0: maximum compression

# Seizure detector band parameters
SEIZURE_BAND = (8.0, 30.0)     # high-frequency band (Hz)
BASELINE_BAND = (0.5,  4.0)    # low-frequency baseline (Hz)
NPERSEG_DET = 512
SENSITIVITY_RATIO_PASS = 0.80
MIN_SEIZURE_WINDOWS = 5


def hp_filter_batch(signal_ch21, fs=FS, fc=0.5):
    """HP filter all 21 channels."""
    sos = butter(2, fc, btype='high', fs=fs, output='sos')
    out = np.empty_like(signal_ch21)
    for ch in range(signal_ch21.shape[0]):
        out[ch] = sosfilt(sos, signal_ch21[ch])
    return out


def band_power(signal_1d, fs, f_lo, f_hi, nperseg=NPERSEG_DET):
    """Compute power in [f_lo, f_hi] Hz for a 1D signal."""
    f, psd = welch(signal_1d, fs=fs, nperseg=min(nperseg, len(signal_1d)))
    mask = (f >= f_lo) & (f <= f_hi)
    if not np.any(mask):
        return 1e-30
    return float(np.mean(psd[mask]))


def seizure_score(hp_signal, fs=FS):
    """Compute seizure detector score for a [21, 2500] window.

    Returns max over channels of (power 8-30 Hz) / (power 0.5-4 Hz).
    """
    ratios = []
    for ch in range(hp_signal.shape[0]):
        sig = hp_signal[ch]
        p_high = band_power(sig, fs, SEIZURE_BAND[0], SEIZURE_BAND[1])
        p_low = band_power(sig, fs, BASELINE_BAND[0], BASELINE_BAND[1])
        ratios.append(p_high / (p_low + 1e-30))
    return float(np.max(ratios))


def round_trip_mode0(model, hp_signal, device):
    """Full Mode 0 round-trip: HP signal → TNN(FSQ L=8) → reconstructed raw.

    hp_signal: [21, 2500] float64
    Returns: [21, 2500] float64 reconstructed
    """
    subs_per_ch = []
    l3_all = np.zeros((N_CHANNELS, 313), dtype=np.float64)
    for ch in range(N_CHANNELS):
        subs = lifting_3level_forward(hp_signal[ch])
        subs_per_ch.append(subs)
        l3_all[ch] = subs['l3_approx']

    l3_tensor = torch.from_numpy(l3_all[np.newaxis].astype(np.float32)).to(device)

    with torch.no_grad():
        # Encode to latent [1, 32, 79]
        latent = model.encode(l3_tensor, quantize=True)
        lat_np = latent[0].cpu().numpy()  # [32, 79]

        # FSQ quantize to L=8 symbols
        syms, vmin, vmax = fsq_encode(lat_np, FSQ_L)

        # FSQ dequantize
        lat_recon = fsq_decode(syms, FSQ_L, vmin, vmax)  # [32, 79] float32
        lat_t = torch.from_numpy(lat_recon).unsqueeze(0).float().to(device)  # [1, 32, 79]

        # Decode: [1, 32, 79] → [1, 21, 313]
        recon_l3 = model.decode(lat_t, target_len=313, quantize=True)
    recon_l3_np = recon_l3[0].cpu().numpy().astype(np.float64)  # [21, 313]

    # Inverse lifting per channel
    recon_raw = np.zeros((N_CHANNELS, WIN_LEN), dtype=np.float64)
    for ch in range(N_CHANNELS):
        subs_recon = dict(subs_per_ch[ch])
        subs_recon['l3_approx'] = recon_l3_np[ch]
        recon_raw[ch] = lifting_3level_inverse(subs_recon)[:WIN_LEN]

    return recon_raw


def process_patient(model, npz_path, device):
    """Process all windows for one patient.

    Returns:
        scores_orig: list of (score, is_seizure) for original windows
        scores_recon: list of (score, is_seizure) for reconstructed windows
    """
    with np.load(npz_path) as npz:
        raw_data = npz['data']         # [21, N]
        seizure_mask = npz['seizure_mask']  # [N]
        n_l3 = npz['l3'].shape[0]

    total_samples = raw_data.shape[1]
    n_samples_mask = len(seizure_mask)

    scores_orig = []
    scores_recon = []

    for wi in range(n_l3):
        start = wi * WIN_LEN
        end = start + WIN_LEN
        if end > total_samples:
            break

        # Seizure label from mask
        mask_start = min(start, n_samples_mask - 1)
        mask_end = min(end, n_samples_mask)
        is_seizure = bool(np.any(seizure_mask[mask_start:mask_end] > 0))

        # Q31 → float mV
        raw_slice = raw_data[:, start:end].astype(np.float64) / Q31_SCALE * MV_SCALE
        hp_signal = hp_filter_batch(raw_slice)

        # Detector on original
        score_orig = seizure_score(hp_signal)

        # Full Mode 0 round-trip
        recon_raw = round_trip_mode0(model, hp_signal, device)
        score_recon = seizure_score(recon_raw)

        scores_orig.append((score_orig, is_seizure))
        scores_recon.append((score_recon, is_seizure))

    return scores_orig, scores_recon


def find_optimal_threshold(scores_with_labels):
    """Sweep detection threshold to find optimal F1 on the given set.

    scores_with_labels: list of (score, is_seizure)
    Returns: optimal threshold (float)
    """
    scores = np.array([s for s, _ in scores_with_labels])
    labels = np.array([int(lbl) for _, lbl in scores_with_labels])

    if not np.any(labels):
        return float('inf')   # no positives → any threshold misses all

    best_f1 = -1.0
    best_thr = 0.0

    for thr in sorted(set(scores)):
        preds = (scores >= thr).astype(int)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    return best_thr


def apply_threshold(scores_with_labels, thr):
    """Apply a fixed threshold, return (sensitivity, specificity, TP, FP, TN, FN)."""
    scores = np.array([s for s, _ in scores_with_labels])
    labels = np.array([int(lbl) for _, lbl in scores_with_labels])

    preds = (scores >= thr).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    n_pos = tp + fn
    sensitivity = tp / max(n_pos, 1)
    specificity = tn / max(tn + fp, 1)

    return sensitivity, specificity, tp, fp, tn, fn


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
        print("[SKIP] D7 Clinical Preservation requires a trained student checkpoint.")
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

    print(f"[*] D7: Clinical Feature Preservation at Mode 0 (FSQ L={FSQ_L}) on {device}")
    print(f"    Checkpoint: {checkpoint_path}")
    print(f"    Patients  : {', '.join(sorted(available.keys()))}")
    print(f"    Detector  : band_power_ratio = power({SEIZURE_BAND[0]}-{SEIZURE_BAND[1]} Hz) "
          f"/ power({BASELINE_BAND[0]}-{BASELINE_BAND[1]} Hz), max over channels")
    print(f"    PASS: sensitivity_ratio >= {SENSITIVITY_RATIO_PASS}")
    print()

    all_scores_orig = []
    all_scores_recon = []
    pat_summary = []

    for patient, path in sorted(available.items()):
        print(f"  Processing {patient}...", end='', flush=True)
        scores_orig, scores_recon = process_patient(model, path, device)
        n_seiz = sum(1 for _, lbl in scores_orig if lbl)
        n_quie = sum(1 for _, lbl in scores_orig if not lbl)
        print(f" {len(scores_orig)} windows ({n_seiz} seizure, {n_quie} quiescent)")
        all_scores_orig.extend(scores_orig)
        all_scores_recon.extend(scores_recon)
        pat_summary.append((patient, len(scores_orig), n_seiz, n_quie))

    total_seizure = sum(1 for _, lbl in all_scores_orig if lbl)
    total_windows = len(all_scores_orig)

    print()
    print("=" * 78)
    print(" D7: CLINICAL FEATURE PRESERVATION AT MODE 0")
    print(f" Seizure detector: max-channel band-power ratio ({SEIZURE_BAND[0]}-{SEIZURE_BAND[1]} Hz) "
          f"/ ({BASELINE_BAND[0]}-{BASELINE_BAND[1]} Hz)")
    print(f" FSQ Level: L={FSQ_L} (Mode 0, maximum compression)")
    print("=" * 78)
    print()
    print(f"  {'Patient':<10} {'Windows':>8} {'Seizure':>8} {'Quiescent':>10}")
    print(f"  {'-'*42}")
    for (pat, n_w, n_s, n_q) in pat_summary:
        print(f"  {pat:<10} {n_w:>8} {n_s:>8} {n_q:>10}")
    print(f"  {'-'*42}")
    print(f"  {'Total':<10} {total_windows:>8} {total_seizure:>8} {total_windows-total_seizure:>10}")
    print()

    if total_seizure < MIN_SEIZURE_WINDOWS:
        msg = (f"Only {total_seizure} seizure windows found across all patients "
               f"(need >= {MIN_SEIZURE_WINDOWS}).")
        print(f"  [SKIP] {msg}")
        print("         Cannot evaluate clinical preservation. "
              "Check if holdout patients have seizure annotations.")
        print()
        print("=" * 78)
        return {
            'passed': None,
            'skipped': True,
            'reason': msg,
            'total_seizure_windows': total_seizure,
            'total_windows': total_windows,
        }

    # Find optimal threshold on ORIGINAL data
    opt_thr = find_optimal_threshold(all_scores_orig)
    print(f"  Optimal threshold (from original signal): {opt_thr:.4f}")
    print()

    # Apply same threshold to both
    sens_orig, spec_orig, tp_o, fp_o, tn_o, fn_o = apply_threshold(all_scores_orig, opt_thr)
    sens_recon, spec_recon, tp_r, fp_r, tn_r, fn_r = apply_threshold(all_scores_recon, opt_thr)

    sensitivity_ratio = sens_recon / max(sens_orig, 1e-8)

    print(f"  {'Metric':<30} {'Original':>12} {'Reconstructed':>14}")
    print(f"  {'-'*58}")
    print(f"  {'Sensitivity (TP / P)':<30} {sens_orig:>11.3f}  {sens_recon:>13.3f}")
    print(f"  {'Specificity (TN / N)':<30} {spec_orig:>11.3f}  {spec_recon:>13.3f}")
    print(f"  {'True Positives':<30} {tp_o:>12}  {tp_r:>13}")
    print(f"  {'False Negatives':<30} {fn_o:>12}  {fn_r:>13}")
    print(f"  {'False Positives':<30} {fp_o:>12}  {fp_r:>13}")
    print(f"  {'True Negatives':<30} {tn_o:>12}  {tn_r:>13}")
    print()
    print(f"  Sensitivity ratio (recon / orig): {sensitivity_ratio:.3f}  "
          f"(threshold: >= {SENSITIVITY_RATIO_PASS})")
    print()

    passed = sensitivity_ratio >= SENSITIVITY_RATIO_PASS

    if passed:
        print(f"  [PASS] Sensitivity ratio={sensitivity_ratio:.3f} >= {SENSITIVITY_RATIO_PASS} — "
              f"clinical detector agreement preserved at Mode 0 compression.")
    else:
        print(f"  [FAIL] Sensitivity ratio={sensitivity_ratio:.3f} < {SENSITIVITY_RATIO_PASS} — "
              f"Mode 0 compression degrades seizure detector sensitivity too much.")

    print()
    print("=" * 78)

    return {
        'passed': passed,
        'sensitivity_ratio': sensitivity_ratio,
        'sensitivity_orig': sens_orig,
        'sensitivity_recon': sens_recon,
        'specificity_orig': spec_orig,
        'specificity_recon': spec_recon,
        'optimal_threshold': opt_thr,
        'total_seizure_windows': total_seizure,
        'total_windows': total_windows,
        'fsq_L': FSQ_L,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()
    result = run(args.checkpoint)
    if result and result.get('passed') is not None:
        sys.exit(0 if result['passed'] else 1)
