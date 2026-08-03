#!/usr/bin/env python3
"""
LamQuant Gen 7.5 — D9: Latent Space Utilization
================================================
Diagnostic D9: Are latent dimensions wasted?

Encodes all holdout-patient L3 windows, applies FSQ at L=16, then groups
the 32 latent channels into 8 groups of 4. For each group computes:
  - Symbol utilization (% of 16 levels actually used)
  - Mutual information with the adjacent group

PASS criteria:
  - mean utilization > 60%
  - no single group < 25%

Usage:
  python benchmark_latent_utilization.py
  python benchmark_latent_utilization.py --checkpoint path/to/model.ckpt
"""

import torch
import os
import sys
import numpy as np
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

FSQ_L = 16           # FSQ quantization levels
N_GROUPS = 8         # groups of 4 dims each (8 * 4 = 32 latent dims)
DIMS_PER_GROUP = 4


def load_l3_windows(path, max_windows=20):
    """Load up to max_windows L3 windows from a q31 NPZ file."""
    with np.load(path) as data:
        l3 = data['l3']  # [N, 21, 313]
    n = min(l3.shape[0], max_windows)
    idx = np.linspace(0, l3.shape[0] - 1, n, dtype=int)
    return l3[idx]  # [n, 21, 313]


def mutual_information(syms_a, syms_b, L):
    """Compute mutual information (bits) between two discrete sequences.

    syms_a, syms_b: 1D arrays of integer symbols in [0, L).
    Uses joint histogram + marginals.
    """
    assert len(syms_a) == len(syms_b)
    joint = np.zeros((L, L), dtype=np.float64)
    for a, b in zip(syms_a.astype(int), syms_b.astype(int)):
        joint[a, b] += 1.0
    total = joint.sum()
    if total < 1:
        return 0.0
    joint /= total

    p_a = joint.sum(axis=1)  # marginal over a
    p_b = joint.sum(axis=0)  # marginal over b

    mi = 0.0
    for i in range(L):
        for j in range(L):
            if joint[i, j] > 0 and p_a[i] > 0 and p_b[j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (p_a[i] * p_b[j]))
    return float(mi)


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
    print(f"[*] D9: Latent Space Utilization on {device}")
    print(f"    Checkpoint: {checkpoint_path}")
    print(f"    Patients  : {', '.join(sorted(available.keys()))}")
    print(f"    FSQ levels: L={FSQ_L}")
    print(f"    Groups    : {N_GROUPS} x {DIMS_PER_GROUP} dims")
    print()

    # Collect all latent FSQ symbols AND raw continuous latents
    all_symbols = []   # each entry is [32, 79] uint16
    all_latents = []   # each entry is [32, 79] float32 (for kurtosis)

    for patient, path in sorted(available.items()):
        windows = load_l3_windows(path, max_windows=20)
        x = torch.from_numpy(windows).float().to(device)

        with torch.no_grad():
            latent = model.encode(x, quantize=True)  # [n, 32, 79]

        lat_np = latent.cpu().numpy()  # [n, 32, 79]
        for wi in range(lat_np.shape[0]):
            syms, _, _ = fsq_encode(lat_np[wi], FSQ_L)
            all_symbols.append(syms)
            all_latents.append(lat_np[wi])

    # Stack: [total_windows, 32, 79]
    all_symbols = np.stack(all_symbols, axis=0)
    all_latents_np = np.stack(all_latents, axis=0)  # [total_windows, 32, 79]
    total_windows, n_latent_dims, n_timesteps = all_symbols.shape
    assert n_latent_dims == 32, f"Expected 32 latent dims, got {n_latent_dims}"

    print(f"    Total windows encoded: {total_windows}")
    print()

    # Per-group analysis
    # Group g covers dims [g*4, g*4+4)
    group_utilizations = []
    group_symbols_flat = []  # [N_GROUPS, total_windows * n_timesteps * DIMS_PER_GROUP]

    from scipy.stats import kurtosis as scipy_kurtosis

    group_kurtosis = []  # per-group excess kurtosis (0=Gaussian, >0=peaked, <0=flat)

    for g in range(N_GROUPS):
        dim_start = g * DIMS_PER_GROUP
        dim_end = dim_start + DIMS_PER_GROUP
        # FSQ symbols
        syms_group = all_symbols[:, dim_start:dim_end, :]
        symbols_flat = syms_group.flatten().astype(np.int32)
        n_unique = len(np.unique(symbols_flat))
        utilization = n_unique / FSQ_L * 100.0
        group_utilizations.append(utilization)
        group_symbols_flat.append(symbols_flat)
        # Kurtosis on raw continuous latent (before FSQ)
        lat_group = all_latents_np[:, dim_start:dim_end, :].flatten()
        kurt = float(scipy_kurtosis(lat_group, fisher=True))  # excess kurtosis (Gaussian=0)
        group_kurtosis.append(kurt)

    # Mutual information between adjacent groups (g, g+1)
    # Use only the timestep dimension for MI (consistent length)
    # Per-timestep: shape [total_windows * n_timesteps] per dim
    # We take the mean of the 4 dims in each group as a representative sequence
    group_mi = []
    for g in range(N_GROUPS - 1):
        dim_start_g = g * DIMS_PER_GROUP
        dim_start_h = (g + 1) * DIMS_PER_GROUP

        # Representative: first dim of each group across all windows and timesteps
        # [total_windows, 79] → flatten
        seq_g = all_symbols[:, dim_start_g, :].flatten().astype(np.int32)
        seq_h = all_symbols[:, dim_start_h, :].flatten().astype(np.int32)

        mi = mutual_information(seq_g, seq_h, FSQ_L)
        group_mi.append(mi)

    # Statistics
    mean_utilization = float(np.mean(group_utilizations))
    min_utilization = float(np.min(group_utilizations))
    dead_code_total = sum(FSQ_L - int(round(u / 100.0 * FSQ_L)) for u in group_utilizations)

    # === Report ===
    print("=" * 72)
    print(" D9: LATENT SPACE UTILIZATION")
    print(f" FSQ L={FSQ_L}, 32 dims grouped into {N_GROUPS} groups of {DIMS_PER_GROUP}")
    print("=" * 72)
    print()
    print(f"  {'Group':<8} {'Dims':<12} {'N Unique':>9} {'Util %':>8} {'Kurtosis':>9}   {'MI→next':>10}")
    print(f"  {'-'*60}")

    for g in range(N_GROUPS):
        dim_start = g * DIMS_PER_GROUP
        dim_end = dim_start + DIMS_PER_GROUP - 1
        dim_str = f"{dim_start}-{dim_end}"
        n_unique = int(round(group_utilizations[g] / 100.0 * FSQ_L))
        util = group_utilizations[g]
        kurt = group_kurtosis[g]
        mi_str = f"{group_mi[g]:.3f}" if g < len(group_mi) else "---"
        kurt_tag = "Laplacian" if kurt > 3 else "peaked" if kurt > 1 else "Gaussian" if kurt > -0.5 else "flat"
        warn = " <-- LOW" if util < 25.0 else ""
        print(f"  {'G'+str(g):<8} {dim_str:<12} {n_unique:>9} {util:>7.1f}% {kurt:>8.2f}   {mi_str:>10}  {kurt_tag}{warn}")

    print()
    mean_kurtosis = float(np.mean(group_kurtosis))
    max_kurtosis = float(np.max(group_kurtosis))
    n_laplacian = sum(1 for k in group_kurtosis if k > 3)

    print(f"  Mean utilization : {mean_utilization:.1f}%  (threshold > 60%)")
    print(f"  Min  utilization : {min_utilization:.1f}%  (threshold > 25%)")
    print(f"  Mean kurtosis    : {mean_kurtosis:.2f}  (>3 = Laplacian, good for residual FSQ)")
    print(f"  Max kurtosis     : {max_kurtosis:.2f}  ({n_laplacian}/{N_GROUPS} groups Laplacian)")
    if n_laplacian >= 4:
        print(f"  Residual FSQ     : RECOMMENDED — {n_laplacian} groups have Laplacian latents")
    elif n_laplacian >= 2:
        print(f"  Residual FSQ     : WORTH TESTING — {n_laplacian} groups have Laplacian latents")
    else:
        print(f"  Residual FSQ     : NOT RECOMMENDED — latents too flat/Gaussian")
    print(f"  Dead codes total : {dead_code_total}  (sum of unused bins per group)")
    print(f"  Mean MI adjacent : {np.mean(group_mi):.3f} bits")
    print()

    # Pass/fail
    pass_mean = mean_utilization > 60.0
    pass_min = min_utilization >= 25.0
    passed = pass_mean and pass_min

    if passed:
        print(f"  [PASS] Mean={mean_utilization:.1f}% > 60% AND min={min_utilization:.1f}% >= 25%")
    else:
        if not pass_mean:
            print(f"  [FAIL] Mean utilization={mean_utilization:.1f}% <= 60%")
        if not pass_min:
            print(f"  [FAIL] Min utilization={min_utilization:.1f}% < 25% (dead dimension group)")

    print()
    print("=" * 72)

    return {
        'passed': passed,
        'mean_utilization': mean_utilization,
        'min_utilization': min_utilization,
        'dead_code_total': dead_code_total,
        'group_utilizations': group_utilizations,
        'group_mi': group_mi,
        'group_kurtosis': group_kurtosis,
        'mean_kurtosis': mean_kurtosis,
        'max_kurtosis': max_kurtosis,
        'n_laplacian_groups': n_laplacian,
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
