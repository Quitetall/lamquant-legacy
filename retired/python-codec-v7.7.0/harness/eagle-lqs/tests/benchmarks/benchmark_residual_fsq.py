#!/usr/bin/env python3
"""
LamQuant — Residual FSQ Experiment
===================================
Compares single-stage FSQ against multi-stage residual FSQ on the
holdout patient set.

The hypothesis: two-stage L=2 gives better R at similar or fewer
bits than single-stage L=3, because the residual is sparse (Laplacian
latent distribution) and compresses well with rANS.

Tests:
  1. Single-stage L=2, L=3, L=4, L=8, L=16 (baselines)
  2. Two-stage L=2+L=2 (residual)
  3. Two-stage L=2+L=3 (residual)
  4. Two-stage L=3+L=3 (residual, for Mode 2)
  5. Per-group adaptive: high-variance groups get 2 stages, low-variance get 1

Reports R, total compressed bytes, effective bits-per-sample, and whether
residual FSQ dominates single-stage on the R-D curve.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from scipy.stats import pearsonr
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
sys.path.insert(0, os.path.join(ROOT_DIR, 'ai_models', 'student'))
sys.path.insert(0, os.path.dirname(__file__))

from lamquant_codec.models.encoder import TernaryMobileNetV5_Subband

PATIENTS = {
    'chb15': 'chbmit_chb15_01_q31.npz',
    'chb16': 'chbmit_chb16_01_q31.npz',
    'chb17': 'chbmit_chb17a_03_q31.npz',
    'chb18': 'chbmit_chb18_01_q31.npz',
    'chb19': 'chbmit_chb19_01_q31.npz',
    'chb20': 'chbmit_chb20_01_q31.npz',
}


def fsq_encode(lat_np, L):
    """Our scalar FSQ: per-tensor min/max normalization + uniform binning."""
    vmin, vmax = float(lat_np.min()), float(lat_np.max())
    span = vmax - vmin + 1e-8
    syms = np.clip(((lat_np - vmin) / span * L).astype(np.int32), 0, L - 1)
    return syms.astype(np.uint16), vmin, vmax


def fsq_decode(syms, L, vmin, vmax):
    """Reconstruct from FSQ symbols using midpoint dequantization."""
    span = vmax - vmin + 1e-8
    return vmin + (syms.astype(np.float32) + 0.5) * span / L


def rans_compressed_size(symbols, L):
    """Estimate rANS compressed size in bytes from symbol distribution."""
    flat = symbols.flatten().astype(np.int32)
    counts = np.bincount(flat, minlength=L)
    total = len(flat)
    # Entropy in bits
    probs = counts / total
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    # Compressed size ≈ entropy * num_symbols / 8 + overhead
    compressed_bits = entropy * total
    overhead = L * 2 + 18  # freq table (L × uint16) + header
    return int(compressed_bits / 8) + overhead, entropy


def compute_r(original, reconstructed):
    """Pearson R between flattened arrays."""
    o = original.flatten()
    r = reconstructed.flatten()
    if np.std(o) < 1e-8 or np.std(r) < 1e-8:
        return 0.0
    return float(pearsonr(o, r)[0])


def single_stage_fsq(lat_np, L):
    """Standard single-stage FSQ. Returns (recon, compressed_bytes, entropy)."""
    syms, vmin, vmax = fsq_encode(lat_np, L)
    recon = fsq_decode(syms, L, vmin, vmax)
    comp_bytes, entropy = rans_compressed_size(syms, L)
    return recon, comp_bytes, entropy


def residual_fsq(lat_np, L1, L2):
    """Two-stage residual FSQ.

    Stage 1: quantize at L1 levels
    Stage 2: compute residual, quantize residual at L2 levels
    Returns (recon, total_compressed_bytes, combined_entropy)
    """
    # Stage 1
    syms1, vmin1, vmax1 = fsq_encode(lat_np, L1)
    recon1 = fsq_decode(syms1, L1, vmin1, vmax1)
    comp1, ent1 = rans_compressed_size(syms1, L1)

    # Residual
    residual = lat_np - recon1

    # Stage 2: quantize residual
    syms2, vmin2, vmax2 = fsq_encode(residual, L2)
    recon2 = fsq_decode(syms2, L2, vmin2, vmax2)
    comp2, ent2 = rans_compressed_size(syms2, L2)

    # Final reconstruction
    recon_final = recon1 + recon2

    # Total compressed size: both stages + extra header for stage 2 range
    total_bytes = comp1 + comp2 + 8  # 8 bytes for vmin2/vmax2 floats
    combined_entropy = ent1 + ent2

    return recon_final, total_bytes, combined_entropy


def per_group_adaptive_residual(lat_np, L1=2, L2=2, variance_threshold=None):
    """Per-group adaptive: high-variance groups get 2 stages, low-variance get 1.

    lat_np: [32, 79] — 8 groups of 4 dimensions each.
    """
    n_dims, T = lat_np.shape
    n_groups = 8
    dims_per_group = n_dims // n_groups

    # Compute per-group variance
    group_var = []
    for g in range(n_groups):
        d_start = g * dims_per_group
        d_end = d_start + dims_per_group
        group_var.append(np.var(lat_np[d_start:d_end]))

    if variance_threshold is None:
        variance_threshold = np.median(group_var)

    recon = np.zeros_like(lat_np)
    total_bytes = 0
    stages_used = []

    for g in range(n_groups):
        d_start = g * dims_per_group
        d_end = d_start + dims_per_group
        group_lat = lat_np[d_start:d_end]

        if group_var[g] > variance_threshold:
            # High variance: 2-stage residual
            group_recon, group_bytes, _ = residual_fsq(group_lat, L1, L2)
            stages_used.append(2)
        else:
            # Low variance: 1-stage
            group_recon, group_bytes, _ = single_stage_fsq(group_lat, L1 * L2)
            stages_used.append(1)

        recon[d_start:d_end] = group_recon
        total_bytes += group_bytes

    return recon, total_bytes, stages_used


def run(checkpoint_path=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Find checkpoint
    if checkpoint_path is None:
        for p in [
            os.path.join(ROOT_DIR, 'ai_models/student/student_hardened.ckpt'),
            os.path.join(ROOT_DIR, 'weights/student_subband.ckpt'),
            os.path.join(ROOT_DIR, 'weights/student_subband_fast.ckpt'),
        ]:
            if os.path.exists(p):
                checkpoint_path = p
                break
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print("[SKIP] No checkpoint found.")
        return None

    # Find patient data
    q31_dir = os.path.join(ROOT_DIR, 'ai_models/dataset_sim/q31_events')
    available = {}
    for name, fname in PATIENTS.items():
        p = os.path.join(q31_dir, fname)
        if os.path.exists(p):
            available[name] = p
    if len(available) < 3:
        print(f"[SKIP] Need at least 3 patients, found {len(available)}")
        return None

    # Load model and encode
    model = TernaryMobileNetV5_Subband.from_checkpoint(checkpoint_path, device=device).eval()
    print(f"[*] Residual FSQ Experiment on {device}")
    print(f"    Checkpoint: {checkpoint_path}")
    print(f"    Patients: {', '.join(sorted(available.keys()))}")

    # Collect latents and originals
    all_latents = []
    all_originals = []
    for patient, path in sorted(available.items()):
        with np.load(path) as data:
            l3 = data['l3'][:5]  # 5 windows per patient
        x = torch.from_numpy(l3).float().to(device)
        with torch.no_grad():
            lat = model.encode(x, quantize=True).cpu().numpy()
            recon_baseline = model(x, quantize=True).cpu().numpy()
        all_latents.append(lat)
        all_originals.append(l3)

    latents = np.concatenate(all_latents, axis=0)  # [N, 32, 79]
    originals = np.concatenate(all_originals, axis=0)  # [N, 21, 313]
    N = latents.shape[0]
    raw_l3_bytes = 21 * 313 * 4  # float32 per window

    print(f"    Windows: {N}")
    print()

    # ========================================
    # Single-stage baselines
    # ========================================
    print("=" * 70)
    print(" SINGLE-STAGE FSQ BASELINES")
    print("=" * 70)
    print(f"{'L':>4} {'R':>8} {'Bytes':>8} {'BPS':>8} {'Entropy':>8} {'CR':>8}")
    print("-" * 70)

    single_results = {}
    for L in [2, 3, 4, 6, 8, 12, 16]:
        rs, bytess = [], []
        for i in range(N):
            recon_lat, comp_bytes, entropy = single_stage_fsq(latents[i], L)
            # Decode through model
            recon_l3 = model.decode(
                torch.from_numpy(recon_lat[None]).float().to(device),
                target_len=313, quantize=True
            ).detach().cpu().numpy()[0]
            r = compute_r(originals[i], recon_l3)
            rs.append(r)
            bytess.append(comp_bytes)
        mean_r = np.mean(rs)
        mean_bytes = np.mean(bytess)
        bps = mean_bytes * 8 / (21 * 313)
        cr = raw_l3_bytes / mean_bytes
        single_results[L] = {'R': mean_r, 'bytes': mean_bytes, 'bps': bps, 'cr': cr}
        print(f"{L:>4} {mean_r:>8.4f} {mean_bytes:>8.0f} {bps:>8.3f} {'—':>8} {cr:>7.1f}x")

    # ========================================
    # Residual FSQ
    # ========================================
    print()
    print("=" * 70)
    print(" RESIDUAL FSQ (TWO-STAGE)")
    print("=" * 70)
    print(f"{'Config':>10} {'R':>8} {'Bytes':>8} {'BPS':>8} {'CR':>8} {'vs best single':>16}")
    print("-" * 70)

    residual_results = {}
    for L1, L2 in [(2, 2), (2, 3), (3, 2), (3, 3), (4, 2), (4, 4)]:
        rs, bytess = [], []
        for i in range(N):
            recon_lat, comp_bytes, _ = residual_fsq(latents[i], L1, L2)
            recon_l3 = model.decode(
                torch.from_numpy(recon_lat[None]).float().to(device),
                target_len=313, quantize=True
            ).detach().cpu().numpy()[0]
            r = compute_r(originals[i], recon_l3)
            rs.append(r)
            bytess.append(comp_bytes)
        mean_r = np.mean(rs)
        mean_bytes = np.mean(bytess)
        bps = mean_bytes * 8 / (21 * 313)
        cr = raw_l3_bytes / mean_bytes

        # Compare to best single-stage at similar bitrate
        effective_L = L1 * L2
        single_r = single_results.get(effective_L, single_results.get(L1, {})).get('R', 0)
        delta = mean_r - single_r

        config = f"L={L1}+{L2}"
        residual_results[(L1, L2)] = {'R': mean_r, 'bytes': mean_bytes, 'bps': bps,
                                       'cr': cr, 'delta_vs_single': delta}
        better = "BETTER" if delta > 0.005 else "similar" if delta > -0.005 else "worse"
        print(f"{config:>10} {mean_r:>8.4f} {mean_bytes:>8.0f} {bps:>8.3f} {cr:>7.1f}x "
              f"{delta:>+8.4f} ({better})")

    # ========================================
    # Per-group adaptive
    # ========================================
    print()
    print("=" * 70)
    print(" PER-GROUP ADAPTIVE RESIDUAL")
    print("=" * 70)

    rs, bytess, all_stages = [], [], []
    for i in range(N):
        recon_lat, comp_bytes, stages = per_group_adaptive_residual(latents[i], L1=2, L2=2)
        recon_l3 = model.decode(
            torch.from_numpy(recon_lat[None]).float().to(device),
            target_len=313, quantize=True
        ).detach().cpu().numpy()[0]
        r = compute_r(originals[i], recon_l3)
        rs.append(r)
        bytess.append(comp_bytes)
        all_stages.append(stages)

    mean_r = np.mean(rs)
    mean_bytes = np.mean(bytess)
    bps = mean_bytes * 8 / (21 * 313)
    cr = raw_l3_bytes / mean_bytes
    stage_counts = np.array(all_stages)
    avg_2stage = np.mean(stage_counts == 2, axis=0)

    print(f"  Adaptive L=2+2 (variance threshold): R={mean_r:.4f}, {mean_bytes:.0f} bytes, "
          f"{bps:.3f} bps, {cr:.1f}x CR")
    print(f"  Per-group 2-stage usage: {['%.0f%%' % (p*100) for p in avg_2stage]}")

    # ========================================
    # Summary
    # ========================================
    print()
    print("=" * 70)
    print(" R-D COMPARISON: Does residual FSQ dominate single-stage?")
    print("=" * 70)

    # Find cases where residual beats single at matched bitrate
    wins = 0
    for (L1, L2), res in residual_results.items():
        # Find closest single-stage by bytes
        closest_L = min(single_results.keys(),
                        key=lambda L: abs(single_results[L]['bytes'] - res['bytes']))
        single_at_match = single_results[closest_L]
        if res['R'] > single_at_match['R'] + 0.005:
            wins += 1
            print(f"  WIN: L={L1}+{L2} (R={res['R']:.4f}, {res['bytes']:.0f}B) > "
                  f"single L={closest_L} (R={single_at_match['R']:.4f}, {single_at_match['bytes']:.0f}B)")
        elif res['R'] > single_at_match['R'] - 0.005:
            print(f"  TIE: L={L1}+{L2} (R={res['R']:.4f}, {res['bytes']:.0f}B) ≈ "
                  f"single L={closest_L} (R={single_at_match['R']:.4f}, {single_at_match['bytes']:.0f}B)")

    passed = wins > 0
    print()
    if passed:
        print(f"  VERDICT: Residual FSQ dominates at {wins} operating point(s). Worth adopting.")
    else:
        print(f"  VERDICT: No clear wins. Single-stage FSQ is sufficient for current latent distribution.")

    return {
        'single_stage': single_results,
        'residual': residual_results,
        'adaptive_R': mean_r,
        'wins': wins,
        'passed': passed,
    }


if __name__ == '__main__':
    result = run()
    if result:
        sys.exit(0 if result['passed'] else 1)
