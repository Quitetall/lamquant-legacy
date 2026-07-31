#!/usr/bin/env python3
"""
LamQuant Gen 7.5 — D6: Seizure vs Quiescent Reconstruction Quality
===================================================================
Diagnostic D6: Does reconstruction quality hold during seizures?

Maps each L3 window back to its raw sample range using the seizure_mask,
then encodes + decodes the window and computes Pearson R. Windows are
binned as seizure or quiescent. Results are pooled across all patients.

PASS criteria:
  - delta (quiescent_R - seizure_R) < 0.15
  - at least 5 seizure windows total

Usage:
  python benchmark_seizure_vs_quiescent.py
  python benchmark_seizure_vs_quiescent.py --checkpoint path/to/model.ckpt
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

FSQ_L = 16


def load_patient_data(path):
    """Load L3 windows and seizure_mask from a q31 NPZ file.

    Returns (l3, seizure_mask): l3 shape [N, 21, 313], mask shape [900000].
    """
    with np.load(path) as data:
        l3 = data['l3']           # [N, 21, 313]
        seizure_mask = data['seizure_mask']   # [900000]
    return l3, seizure_mask


def encode_decode_single(model, window_np, device):
    """Encode + decode a single L3 window [21, 313] via FSQ."""
    x = torch.from_numpy(window_np[np.newaxis]).float().to(device)  # [1, 21, 313]
    target_len = window_np.shape[1]

    with torch.no_grad():
        lat = model.encode(x, quantize=True)       # [1, 32, 79]
        lat_np = lat[0].cpu().numpy()               # [32, 79]

        syms, vmin, vmax = fsq_encode(lat_np, FSQ_L)
        span = vmax - vmin + 1e-8
        lat_rec = vmin + (syms.astype(np.float32) + 0.5) * span / FSQ_L
        lat_t = torch.from_numpy(lat_rec).unsqueeze(0).float().to(device)

        dec = model.decode(lat_t, target_len=target_len, quantize=True)

    return dec[0].cpu().numpy()  # [21, 313]


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
    print(f"[*] D6: Seizure vs Quiescent Reconstruction Quality on {device}")
    print(f"    Checkpoint: {checkpoint_path}")
    print(f"    Patients  : {', '.join(sorted(available.keys()))}")
    print(f"    FSQ levels: L={FSQ_L}")
    print()

    seizure_R_list = []
    quiescent_R_list = []

    # Per-patient summary table rows
    summary_rows = []

    for patient, path in sorted(available.items()):
        l3, seizure_mask = load_patient_data(path)
        n_l3_windows = l3.shape[0]
        stride = max(1, len(seizure_mask) // n_l3_windows)

        pat_seiz_r = []
        pat_quie_r = []

        for wi in range(n_l3_windows):
            start = wi * stride
            end = min(start + stride, len(seizure_mask))
            is_seizure = bool(np.any(seizure_mask[start:end] > 0))

            # Encode + decode
            recon = encode_decode_single(model, l3[wi], device)

            # Pearson R: flatten channels and time
            orig_flat = l3[wi].flatten()
            recon_flat = recon.flatten()
            if np.std(orig_flat) < 1e-8 or np.std(recon_flat) < 1e-8:
                continue
            r, _ = pearsonr(orig_flat, recon_flat)

            if is_seizure:
                pat_seiz_r.append(r)
                seizure_R_list.append(r)
            else:
                pat_quie_r.append(r)
                quiescent_R_list.append(r)

        summary_rows.append((
            patient,
            len(pat_seiz_r),
            float(np.mean(pat_seiz_r)) if pat_seiz_r else float('nan'),
            len(pat_quie_r),
            float(np.mean(pat_quie_r)) if pat_quie_r else float('nan'),
        ))

    # === Report ===
    print("=" * 78)
    print(" D6: SEIZURE vs QUIESCENT RECONSTRUCTION QUALITY")
    print("=" * 78)
    print()
    print(f"  {'Patient':<10} {'Seiz Win':>9} {'Seiz R':>8}   {'Quie Win':>9} {'Quie R':>8}")
    print(f"  {'-'*55}")
    for (pat, n_s, r_s, n_q, r_q) in summary_rows:
        r_s_str = f"{r_s:.4f}" if not np.isnan(r_s) else "  N/A  "
        r_q_str = f"{r_q:.4f}" if not np.isnan(r_q) else "  N/A  "
        print(f"  {pat:<10} {n_s:>9} {r_s_str:>8}   {n_q:>9} {r_q_str:>8}")

    print()

    total_seizure = len(seizure_R_list)
    total_quiescent = len(quiescent_R_list)
    mean_seiz_r = float(np.mean(seizure_R_list)) if seizure_R_list else float('nan')
    mean_quie_r = float(np.mean(quiescent_R_list)) if quiescent_R_list else float('nan')
    delta = mean_quie_r - mean_seiz_r if (seizure_R_list and quiescent_R_list) else float('nan')

    print(f"  Total seizure windows  : {total_seizure}")
    print(f"  Total quiescent windows: {total_quiescent}")
    print()
    print(f"  Mean R (seizure)   : {mean_seiz_r:.4f}" if not np.isnan(mean_seiz_r) else "  Mean R (seizure)   : N/A")
    print(f"  Mean R (quiescent) : {mean_quie_r:.4f}" if not np.isnan(mean_quie_r) else "  Mean R (quiescent) : N/A")
    print(f"  Delta (quie - seiz): {delta:.4f}  (threshold < 0.15)" if not np.isnan(delta) else "  Delta (quie - seiz): N/A")
    print()

    # Pass/fail
    if total_seizure < 5:
        print(f"  [SKIP] Only {total_seizure} seizure windows found (need >= 5).")
        print("         Cannot evaluate seizure degradation.")
        print()
        print("=" * 78)
        return {
            'passed': None,
            'skipped': True,
            'reason': f'Only {total_seizure} seizure windows (need >= 5)',
            'total_seizure_windows': total_seizure,
            'total_quiescent_windows': total_quiescent,
            'mean_seizure_r': mean_seiz_r,
            'mean_quiescent_r': mean_quie_r,
            'delta': delta,
        }

    passed = (not np.isnan(delta)) and (delta < 0.15)

    if passed:
        print(f"  [PASS] Delta={delta:.4f} < 0.15 — seizure R within acceptable range")
    else:
        print(f"  [FAIL] Delta={delta:.4f} >= 0.15 — seizure reconstruction quality drops too much")

    print()
    print("=" * 78)

    return {
        'passed': passed,
        'total_seizure_windows': total_seizure,
        'total_quiescent_windows': total_quiescent,
        'mean_seizure_r': mean_seiz_r,
        'mean_quiescent_r': mean_quie_r,
        'delta': delta,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()
    result = run(args.checkpoint)
    if result and result.get('passed') is not None:
        sys.exit(0 if result['passed'] else 1)
