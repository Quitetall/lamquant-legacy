#!/usr/bin/env python3
"""
LamQuant Gen 7.5 — D3: FSQ Token Entropy vs Activity Level
===========================================================
Diagnostic D3: Does SNN-driven adaptive FSQ help compression?

Maps each L3 window to one of three activity buckets:
  - 'seizure'  : seizure_mask[start:end] contains any non-zero value
  - 'active'   : RMS > median RMS across all windows (but not seizure)
  - 'quiet'    : RMS <= median RMS (interictal baseline)

For each window, encodes via the TNN encoder, FSQ-quantizes at L=16,
then computes the empirical entropy of the resulting symbol histogram.

PASS criterion: H(seizure) > H(quiet) — seizure windows carry more information
and should have higher entropy (harder to compress).

Usage:
  python benchmark_fsq_entropy_activity.py
  python benchmark_fsq_entropy_activity.py --checkpoint path/to/model.ckpt
"""

import torch
import os
import sys
import numpy as np
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

sys.path.insert(0, os.path.dirname(__file__))
from benchmark_compression_ratio import fsq_encode, fsq_decode, RANSEncoder

PATIENTS = {
    'chb15': 'chbmit_chb15_01_q31.npz',
    'chb16': 'chbmit_chb16_01_q31.npz',
    'chb17': 'chbmit_chb17a_03_q31.npz',
    'chb18': 'chbmit_chb18_01_q31.npz',
    'chb19': 'chbmit_chb19_01_q31.npz',
    'chb20': 'chbmit_chb20_01_q31.npz',
}

FSQ_L = 16
N_L3_WINDOWS = 20


def load_patient_data(path, max_windows=N_L3_WINDOWS):
    """Load up to max_windows L3 windows and the seizure_mask."""
    with np.load(path) as data:
        l3 = data['l3']              # [N, 21, 313]
        seizure_mask = data['seizure_mask']
    n = min(l3.shape[0], max_windows)
    idx = np.linspace(0, l3.shape[0] - 1, n, dtype=int)
    return l3[idx], seizure_mask


def window_entropy(symbols_flat, L):
    """Compute empirical entropy (bits) of a flat symbol array over L bins."""
    counts = np.bincount(symbols_flat.astype(np.int32), minlength=L)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / float(total)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


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

    model = TernaryMobileNetV5_Subband.from_checkpoint(checkpoint_path, device=device).eval()
    print(f"[*] D3: FSQ Token Entropy vs Activity Level on {device}")
    print(f"    Checkpoint: {checkpoint_path}")
    print(f"    Patients  : {', '.join(sorted(available.keys()))}")
    print(f"    FSQ levels: L={FSQ_L}  Windows per patient: {N_L3_WINDOWS}")
    print()

    # Collect entropy per activity bucket across all patients
    bucket_entropies = {'quiet': [], 'active': [], 'seizure': []}
    patient_rows = []

    for patient, path in sorted(available.items()):
        l3, seizure_mask = load_patient_data(path, max_windows=N_L3_WINDOWS)
        n_l3_windows = l3.shape[0]
        stride = max(1, len(seizure_mask) // n_l3_windows)

        # Compute RMS per window for active/quiet split
        rms_per_window = np.array([np.sqrt(np.mean(l3[wi] ** 2)) for wi in range(n_l3_windows)])
        median_rms = float(np.median(rms_per_window))

        pat_buckets = {'quiet': [], 'active': [], 'seizure': []}

        x_batch = torch.from_numpy(l3).float().to(device)
        with torch.no_grad():
            latents = model.encode(x_batch, quantize=True)  # [n, 32, 79]
        lat_np = latents.cpu().numpy()

        for wi in range(n_l3_windows):
            start = wi * stride
            end = min(start + stride, len(seizure_mask))
            is_seizure = bool(np.any(seizure_mask[start:end] > 0))

            syms, _, _ = fsq_encode(lat_np[wi], FSQ_L)   # [32, 79] uint16
            H = window_entropy(syms.flatten(), FSQ_L)

            if is_seizure:
                bucket = 'seizure'
            elif rms_per_window[wi] > median_rms:
                bucket = 'active'
            else:
                bucket = 'quiet'

            pat_buckets[bucket].append(H)
            bucket_entropies[bucket].append(H)

        patient_rows.append((patient, pat_buckets))

    # Per-patient table
    print("=" * 82)
    print(" D3: FSQ TOKEN ENTROPY VS ACTIVITY LEVEL")
    print(" Per-patient breakdown")
    print("=" * 82)
    print(f"  {'Patient':<10} {'Bucket':<10} {'N Win':>6} {'Mean H':>9} {'Std H':>9}")
    print(f"  {'-'*52}")

    for patient, pat_buckets in patient_rows:
        for bucket in ('quiet', 'active', 'seizure'):
            hs = pat_buckets[bucket]
            if hs:
                print(f"  {patient:<10} {bucket:<10} {len(hs):>6} {np.mean(hs):>8.4f} {np.std(hs):>8.4f}")
            else:
                print(f"  {patient:<10} {bucket:<10} {0:>6} {'   N/A':>9} {'   N/A':>9}")
        print()

    # Global summary table
    print("=" * 82)
    print(" GLOBAL SUMMARY (pooled across all patients)")
    print("=" * 82)
    print(f"  {'Activity':<12} {'N Windows':>10} {'Mean H (bits)':>14} {'Std H':>9}")
    print(f"  {'-'*52}")
    summary = {}
    for bucket in ('quiet', 'active', 'seizure'):
        hs = bucket_entropies[bucket]
        n = len(hs)
        mean_h = float(np.mean(hs)) if hs else float('nan')
        std_h = float(np.std(hs)) if hs else float('nan')
        summary[bucket] = (n, mean_h, std_h)
        mean_str = f"{mean_h:.4f}" if not np.isnan(mean_h) else "    N/A"
        std_str  = f"{std_h:.4f}"  if not np.isnan(std_h)  else "    N/A"
        print(f"  {bucket:<12} {n:>10} {mean_str:>14} {std_str:>9}")

    print()

    # Pass/fail
    n_seiz, h_seiz, _ = summary['seizure']
    n_quiet, h_quiet, _ = summary['quiet']

    if n_seiz == 0:
        print(f"  [SKIP] No seizure windows found — cannot evaluate H(seizure) > H(quiet).")
        print("=" * 82)
        return {
            'passed': None,
            'skipped': True,
            'reason': 'No seizure windows found',
            'summary': summary,
        }

    passed = h_seiz > h_quiet

    if passed:
        print(f"  [PASS] H(seizure)={h_seiz:.4f} > H(quiet)={h_quiet:.4f} "
              f"— seizure windows carry more information")
    else:
        print(f"  [FAIL] H(seizure)={h_seiz:.4f} <= H(quiet)={h_quiet:.4f} "
              f"— unexpected: seizure should have higher entropy")

    print()
    print(f"  Interpretation: {'Higher entropy at seizure onset confirms SNN-adaptive FSQ will' if passed else 'Flat entropy suggests model is not differentiating'}")
    if passed:
        print(f"  yield better compression for quiescent baseline windows.")
    print()
    print("=" * 82)

    return {
        'passed': passed,
        'h_seizure': h_seiz,
        'h_active': summary['active'][1],
        'h_quiet': h_quiet,
        'n_seizure': n_seiz,
        'n_active': summary['active'][0],
        'n_quiet': n_quiet,
        'summary': summary,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()
    result = run(args.checkpoint)
    if result and result.get('passed') is not None:
        sys.exit(0 if result['passed'] else 1)
