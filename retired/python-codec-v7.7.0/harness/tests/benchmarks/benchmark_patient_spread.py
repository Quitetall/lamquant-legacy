#!/usr/bin/env python3
"""
LamQuant Gen 7.5 — D5: Per-Patient Generalization Spread
=========================================================
Diagnostic D5: Is variance across held-out patients acceptable?

For each held-out patient, encode + decode 20 L3 windows, compute Pearson R
per window, then average within the patient. Reports per-patient R, along
with the population mean, std, min, max, and range.

PASS criteria:
  - range (max - min) < 0.10
  - min R > 0.80

Usage:
  python benchmark_patient_spread.py
  python benchmark_patient_spread.py --checkpoint path/to/model.ckpt
"""

import torch
import os
import sys
import numpy as np
from scipy.stats import pearsonr
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

# Add benchmarks dir so we can import fsq_encode
sys.path.insert(0, os.path.dirname(__file__))
from benchmark_compression_ratio import fsq_encode

PATIENTS = {
    'chb15': 'chbmit_chb15_01_q31.npz',
    'chb16': 'chbmit_chb16_01_q31.npz',
    'chb17': 'chbmit_chb17a_03_q31.npz',
    'chb18': 'chbmit_chb18_01_q31.npz',
    'chb19': 'chbmit_chb19_01_q31.npz',
    'chb20': 'chbmit_chb20_01_q31.npz',
}

FSQ_L = 16  # quantization levels for encode/decode


def load_l3_windows(path, max_windows=20):
    """Load up to max_windows L3 windows from a q31 NPZ file."""
    with np.load(path) as data:
        l3 = data['l3']  # [N, 21, 313]
    n = min(l3.shape[0], max_windows)
    idx = np.linspace(0, l3.shape[0] - 1, n, dtype=int)
    return l3[idx]  # [n, 21, 313]


def encode_decode_windows(model, windows, device):
    """Encode + decode a batch of L3 windows.

    Returns reconstructed numpy array with same shape as windows.
    """
    x = torch.from_numpy(windows).float().to(device)
    target_len = windows.shape[2]

    # Per-window to avoid OOM on large batches; collect latent then decode
    recons = []
    with torch.no_grad():
        for wi in range(x.shape[0]):
            xi = x[wi:wi + 1]  # [1, 21, 313]
            lat = model.encode(xi, quantize=True)  # [1, 32, 79]
            lat_np = lat[0].cpu().numpy()           # [32, 79]

            # FSQ encode → decode (simulate quantization)
            syms, vmin, vmax = fsq_encode(lat_np, FSQ_L)
            span = vmax - vmin + 1e-8
            lat_rec = vmin + (syms.astype(np.float32) + 0.5) * span / FSQ_L
            lat_t = torch.from_numpy(lat_rec).unsqueeze(0).float().to(device)

            dec = model.decode(lat_t, target_len=target_len, quantize=True)
            recons.append(dec[0].cpu().numpy())  # [21, 313]

    return np.stack(recons, axis=0)  # [n, 21, 313]


def patient_pearson_r(original, reconstructed):
    """Compute Pearson R per window, return list of R values.

    Per-window: flatten all channels and time samples then pearsonr.
    """
    rs = []
    n_win = original.shape[0]
    for wi in range(n_win):
        orig_flat = original[wi].flatten()
        recon_flat = reconstructed[wi].flatten()
        if np.std(orig_flat) < 1e-8 or np.std(recon_flat) < 1e-8:
            continue
        r, _ = pearsonr(orig_flat, recon_flat)
        rs.append(r)
    return rs


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

    # Load model
    model = TernaryMobileNetV5_Subband.from_checkpoint(checkpoint_path, device=device).eval()
    print(f"[*] D5: Per-Patient Generalization Spread on {device}")
    print(f"    Checkpoint: {checkpoint_path}")
    print(f"    Patients  : {', '.join(sorted(available.keys()))}")
    print(f"    FSQ levels: L={FSQ_L}")
    print(f"    Windows   : 20 per patient")
    print()

    # Per-patient evaluation
    patient_mean_r = {}
    for patient, path in sorted(available.items()):
        windows = load_l3_windows(path, max_windows=20)
        recon = encode_decode_windows(model, windows, device)
        rs = patient_pearson_r(windows, recon)
        patient_mean_r[patient] = float(np.mean(rs)) if rs else float('nan')

    # Aggregate statistics
    r_values = [v for v in patient_mean_r.values() if not np.isnan(v)]
    if not r_values:
        print("[FAIL] No valid R values computed.")
        return None

    mean_r = float(np.mean(r_values))
    std_r = float(np.std(r_values))
    min_r = float(np.min(r_values))
    max_r = float(np.max(r_values))
    range_r = max_r - min_r

    # === Report ===
    print("=" * 68)
    print(" D5: PER-PATIENT GENERALIZATION SPREAD")
    print("=" * 68)
    print()
    print(f"  {'Patient':<10} {'Mean R':>8}   {'Windows':>8}")
    print(f"  {'-'*35}")
    for patient in sorted(patient_mean_r):
        r = patient_mean_r[patient]
        mark = " <-- min" if r == min_r else (" <-- max" if r == max_r else "")
        print(f"  {patient:<10} {r:>8.4f}   {'20':>8}{mark}")

    print()
    print(f"  Mean R   : {mean_r:.4f}")
    print(f"  Std R    : {std_r:.4f}")
    print(f"  Min R    : {min_r:.4f}")
    print(f"  Max R    : {max_r:.4f}")
    print(f"  Range    : {range_r:.4f}  (threshold < 0.10)")
    print()

    # Pass/fail
    pass_range = range_r < 0.10
    pass_min = min_r > 0.80
    passed = pass_range and pass_min

    if passed:
        print(f"  [PASS] Range={range_r:.4f} < 0.10 AND min R={min_r:.4f} > 0.80")
    else:
        if not pass_range:
            print(f"  [FAIL] Range={range_r:.4f} >= 0.10 — patient spread too large")
        if not pass_min:
            print(f"  [FAIL] Min R={min_r:.4f} <= 0.80 — worst patient below threshold")

    print()
    print("=" * 68)

    return {
        'passed': passed,
        'per_patient_r': patient_mean_r,
        'mean_r': mean_r,
        'std_r': std_r,
        'min_r': min_r,
        'max_r': max_r,
        'range_r': range_r,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()
    result = run(args.checkpoint)
    if result:
        sys.exit(0 if result['passed'] else 1)
