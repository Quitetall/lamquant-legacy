#!/usr/bin/env python3
"""
LamQuant Gen 7.5 — D14: Cayley Rotation Effectiveness
======================================================
Diagnostic D14: Does the learned rotation justify 4 KB firmware cost?

Encodes holdout-patient L3 windows with the full model (rotation enabled),
then temporarily zeros out rotation_A to simulate no rotation, re-encodes,
and compares FSQ symbol entropy and dead-code count.

PASS criteria (either of):
  - dead_rot < dead_norot  (rotation reduces dead codes), OR
  - entropy_delta > 0.05 bps (rotation increases entropy by at least 0.05 bits)

Usage:
  python benchmark_cayley_rotation.py
  python benchmark_cayley_rotation.py --checkpoint path/to/model.ckpt
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


def load_l3_windows(path, max_windows=20):
    """Load up to max_windows L3 windows from a q31 NPZ file."""
    with np.load(path) as data:
        l3 = data['l3']  # [N, 21, 313]
    n = min(l3.shape[0], max_windows)
    idx = np.linspace(0, l3.shape[0] - 1, n, dtype=int)
    return l3[idx]  # [n, 21, 313]


def compute_entropy_and_dead(symbols_flat, L):
    """Compute empirical entropy (bits per symbol) and dead-code count.

    symbols_flat: 1D array of integer symbols in [0, L).
    Returns (entropy_bps, n_dead_codes).
    """
    counts = np.bincount(symbols_flat.astype(np.int32), minlength=L)
    total = counts.sum()
    if total == 0:
        return 0.0, L
    probs = counts / total
    dead = int(np.sum(counts == 0))
    # Shannon entropy: only sum over non-zero bins
    nonzero = probs[probs > 0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero)))
    return entropy, dead


def encode_all_patients(model, available, device):
    """Encode all patient windows, return flat symbol array [32, N_total_timesteps]."""
    all_syms_list = []  # each: [32, 79] uint16

    for patient, path in sorted(available.items()):
        windows = load_l3_windows(path, max_windows=20)
        x = torch.from_numpy(windows).float().to(device)
        with torch.no_grad():
            latent = model.encode(x, quantize=True)  # [n, 32, 79]
        lat_np = latent.cpu().numpy()
        for wi in range(lat_np.shape[0]):
            syms, _, _ = fsq_encode(lat_np[wi], FSQ_L)  # [32, 79]
            all_syms_list.append(syms)

    # Stack → [total_windows, 32, 79]
    all_syms = np.stack(all_syms_list, axis=0)
    # Flatten to [32, total_windows * 79] for channel-level analysis
    total_windows = all_syms.shape[0]
    flat = all_syms.transpose(1, 0, 2).reshape(32, -1)  # [32, total_windows * 79]
    return flat, total_windows


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
    print(f"[*] D14: Cayley Rotation Effectiveness on {device}")
    print(f"    Checkpoint: {checkpoint_path}")
    print(f"    Patients  : {', '.join(sorted(available.keys()))}")
    print(f"    FSQ levels: L={FSQ_L}")
    print()

    # ---- Step 1: Encode with rotation (default) ----
    print("  Encoding with rotation enabled...")
    syms_rot, total_windows = encode_all_patients(model, available, device)
    # syms_rot: [32, total_windows * 79]

    # Global entropy and dead codes across all dims
    syms_rot_flat = syms_rot.flatten()
    H_rot, dead_rot = compute_entropy_and_dead(syms_rot_flat, FSQ_L)

    # Per-dim dead codes
    dead_rot_per_dim = [
        compute_entropy_and_dead(syms_rot[d], FSQ_L)[1]
        for d in range(32)
    ]

    # ---- Step 2: Disable rotation ----
    print("  Disabling rotation (zeroing rotation_A)...")
    saved_A = model.rotation_A.data.clone()
    model.rotation_A.data.zero_()

    syms_norot, _ = encode_all_patients(model, available, device)
    syms_norot_flat = syms_norot.flatten()
    H_norot, dead_norot = compute_entropy_and_dead(syms_norot_flat, FSQ_L)

    dead_norot_per_dim = [
        compute_entropy_and_dead(syms_norot[d], FSQ_L)[1]
        for d in range(32)
    ]

    # ---- Step 3: Restore rotation ----
    model.rotation_A.data.copy_(saved_A)
    print("  Rotation restored.")
    print()

    # ---- Derived metrics ----
    entropy_delta = H_rot - H_norot
    dead_code_delta = dead_rot - dead_norot  # negative means rotation reduced dead codes

    # ---- Report ----
    print("=" * 72)
    print(" D14: CAYLEY ROTATION EFFECTIVENESS")
    print(f" FSQ L={FSQ_L}, 32 latent dims, {total_windows} windows across all patients")
    print("=" * 72)
    print()
    print(f"  {'Metric':<30} {'With Rotation':>15}  {'No Rotation':>13}  {'Delta':>8}")
    print(f"  {'-'*70}")
    print(f"  {'Entropy (bps)':<30} {H_rot:>15.4f}  {H_norot:>13.4f}  {entropy_delta:>+8.4f}")
    print(f"  {'Dead codes (global)':<30} {dead_rot:>15d}  {dead_norot:>13d}  {dead_code_delta:>+8d}")
    print(f"  {'Dead codes (sum per dim)':<30} {sum(dead_rot_per_dim):>15d}  {sum(dead_norot_per_dim):>13d}  {sum(dead_rot_per_dim)-sum(dead_norot_per_dim):>+8d}")
    print()

    # Per-dim breakdown (compact)
    print(f"  Per-dim dead codes (with rotation vs without):")
    dim_header = "  Dim : " + " ".join(f"{d:>4}" for d in range(32))
    print(dim_header)
    rot_row = "  Rot : " + " ".join(f"{dead_rot_per_dim[d]:>4}" for d in range(32))
    norot_row = "  Nort: " + " ".join(f"{dead_norot_per_dim[d]:>4}" for d in range(32))
    print(rot_row)
    print(norot_row)
    print()

    # Pass/fail
    pass_dead = dead_rot < dead_norot
    pass_entropy = entropy_delta > 0.05

    passed = pass_dead or pass_entropy

    print(f"  Entropy delta    : {entropy_delta:+.4f} bps  (PASS threshold > 0.05)")
    print(f"  Dead-code delta  : {dead_code_delta:+d}  (PASS if negative, i.e. rotation reduces dead codes)")
    print()

    if passed:
        reasons = []
        if pass_dead:
            reasons.append(f"dead_rot ({dead_rot}) < dead_norot ({dead_norot})")
        if pass_entropy:
            reasons.append(f"entropy_delta ({entropy_delta:.4f}) > 0.05 bps")
        print(f"  [PASS] Rotation justified: {'; '.join(reasons)}")
    else:
        print(f"  [FAIL] Rotation NOT justified:")
        print(f"         dead_rot={dead_rot} >= dead_norot={dead_norot}  (no dead-code reduction)")
        print(f"         entropy_delta={entropy_delta:.4f} <= 0.05 bps  (no meaningful entropy gain)")
        print()
        print("  Interpretation: rotation_A may not have converged or training was")
        print("  cut short. Check that hardening included rotation gradient updates.")

    print()
    print("=" * 72)

    return {
        'passed': passed,
        'H_rot': H_rot,
        'H_norot': H_norot,
        'entropy_delta': entropy_delta,
        'dead_rot': dead_rot,
        'dead_norot': dead_norot,
        'dead_code_delta': dead_code_delta,
        'total_windows': total_windows,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()
    result = run(args.checkpoint)
    if result:
        sys.exit(0 if result['passed'] else 1)
