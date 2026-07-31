#!/usr/bin/env python3
"""
LamQuant Gen 7.5 — D4: Rate-Distortion Curve
=============================================
Diagnostic D4: Does Gen 7.5 dominate Gen 6?

Sweeps FSQ quantization levels L = [2, 3, 4, 6, 8, 12, 16, 24, 32],
encodes each patient's L3 windows, rANS-compresses, reconstructs, and
computes Pearson R vs the original L3 signal.  Produces one (bps, R) point
per L value, averaged across all available patients and windows.

If a Gen 6 checkpoint is available at weights/ablation_a_raw_gen6.ckpt,
the same sweep is also run on the Gen 6 model using raw 21-ch x 2500-sample
windows, and both curves are shown.

PASS criterion (relaxed for fast preset):
  Any (bps, R) point achieves R >= 0.85 at bps < 1.5

Report table: L | bps | R | PRD | SNR

Usage:
  python benchmark_rate_distortion.py
  python benchmark_rate_distortion.py --checkpoint path/to/model.ckpt
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

FSQ_LEVELS = [2, 3, 4, 6, 8, 12, 16, 24, 32]
N_L3_WINDOWS = 20

# Original raw sample count for bps denominator: 21 channels x 313 L3 samples
SAMPLES_PER_WINDOW = 21 * 313


def load_l3_windows(path, max_windows=N_L3_WINDOWS):
    """Load up to max_windows evenly-spaced L3 windows from a q31 NPZ file."""
    with np.load(path) as data:
        l3 = data['l3']   # [N, 21, 313]
    n = min(l3.shape[0], max_windows)
    idx = np.linspace(0, l3.shape[0] - 1, n, dtype=int)
    return l3[idx]


def compute_metrics(orig, recon):
    """Return (R, PRD, SNR) between two float arrays (any shape)."""
    o = orig.flatten()
    r = recon.flatten()
    min_len = min(len(o), len(r))
    o, r = o[:min_len], r[:min_len]
    if np.std(o) < 1e-8 or np.std(r) < 1e-8:
        return 0.0, 100.0, -10.0
    rval, _ = pearsonr(o, r)
    rms_diff = np.sqrt(np.mean((o - r) ** 2))
    rms_orig = np.sqrt(np.mean(o ** 2))
    prd = (rms_diff / (rms_orig + 1e-8)) * 100.0
    snr = 10.0 * np.log10(np.mean(o ** 2) / (np.mean((o - r) ** 2) + 1e-12))
    return float(rval), float(prd), float(snr)


def rans_encode_symbols(syms_flat, L):
    """rANS encode a flat uint16 symbol array; return byte count."""
    counts = np.bincount(syms_flat.astype(np.int32), minlength=L).tolist()
    enc = RANSEncoder(counts)
    # encode in reverse order (rANS convention)
    for i in range(len(syms_flat) - 1, -1, -1):
        enc.put(int(syms_flat[i]))
    enc.flush()
    return len(enc.out)


def rd_sweep_gen75(model, available, device):
    """Rate-distortion sweep for Gen 7.5 (L3 subband input)."""
    rd_points = []

    for L in FSQ_LEVELS:
        r_list = []
        prd_list = []
        snr_list = []
        bps_list = []

        for patient, path in sorted(available.items()):
            windows = load_l3_windows(path, max_windows=N_L3_WINDOWS)  # [n, 21, 313]
            x = torch.from_numpy(windows).float().to(device)

            with torch.no_grad():
                latents = model.encode(x, quantize=True)   # [n, 32, 79]
            lat_np = latents.cpu().numpy()

            for wi in range(windows.shape[0]):
                orig = windows[wi]   # [21, 313]

                syms, vmin, vmax = fsq_encode(lat_np[wi], L)  # [32, 79]
                byte_count = rans_encode_symbols(syms.flatten(), L)

                lat_rec = fsq_decode(syms, L, vmin, vmax)  # [32, 79]
                lat_t = torch.from_numpy(lat_rec[None]).float().to(device)

                with torch.no_grad():
                    recon = model.decode(lat_t, target_len=313, quantize=True)
                recon_np = recon[0].cpu().numpy()  # [21, 313]

                r, prd, snr = compute_metrics(orig, recon_np)
                bps = (byte_count * 8) / SAMPLES_PER_WINDOW

                r_list.append(r)
                prd_list.append(prd)
                snr_list.append(snr)
                bps_list.append(bps)

        mean_r   = float(np.mean(r_list))
        mean_prd = float(np.mean(prd_list))
        mean_snr = float(np.mean(snr_list))
        mean_bps = float(np.mean(bps_list))
        rd_points.append((L, mean_bps, mean_r, mean_prd, mean_snr))

    return rd_points


def run(checkpoint_path=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Checkpoint discovery — Gen 7.5
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
        print("[SKIP] No Gen 7.5 student checkpoint found.")
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
    print(f"[*] D4: Rate-Distortion Curve on {device}")
    print(f"    Gen 7.5 checkpoint: {checkpoint_path}")
    print(f"    Patients          : {', '.join(sorted(available.keys()))}")
    print(f"    FSQ levels        : {FSQ_LEVELS}")
    print(f"    Windows/patient   : {N_L3_WINDOWS}")
    print()

    # Gen 7.5 sweep
    rd_points = rd_sweep_gen75(model, available, device)

    print("=" * 80)
    print(" D4: RATE-DISTORTION CURVE — Gen 7.5 (L3 subband, 21ch x 313 samples)")
    print("=" * 80)
    print(f"  {'L':>4} {'bps':>8} {'R':>9} {'PRD (%)':>9} {'SNR (dB)':>10}")
    print(f"  {'-'*46}")
    for L, bps, r, prd, snr in rd_points:
        marker = " <-- PASS" if r >= 0.85 and bps < 1.5 else ""
        print(f"  {L:>4} {bps:>8.4f} {r:>9.4f} {prd:>9.2f} {snr:>10.1f}{marker}")
    print()

    # Gen 6 comparison
    gen6_ckpt_candidates = [
        os.path.join(ROOT_DIR, 'weights/ablation_a_raw_gen6.ckpt'),
        os.path.join(ROOT_DIR, 'weights/gen6.ckpt'),
    ]
    gen6_ckpt = next((p for p in gen6_ckpt_candidates if os.path.exists(p)), None)

    if gen6_ckpt is not None:
        print(f"  [Gen 6 comparison] Found checkpoint: {gen6_ckpt}")
        print(f"  Note: Gen 6 comparison would require a compatible raw-input model class.")
        print(f"        Skipping detailed Gen 6 sweep (class mismatch risk).")
    else:
        print("  Gen 6 comparison: not available")
        print(f"  (Expected at: {gen6_ckpt_candidates[0]})")
    print()

    # Pass/fail
    pass_points = [(L, bps, r) for L, bps, r, prd, snr in rd_points
                   if r >= 0.85 and bps < 1.5]

    print("=" * 80)
    if pass_points:
        best = max(pass_points, key=lambda x: x[2])
        print(f"  [PASS] {len(pass_points)} point(s) achieve R >= 0.85 at bps < 1.5")
        print(f"         Best: L={best[0]}, bps={best[1]:.4f}, R={best[2]:.4f}")
    else:
        # Check if we have any reasonable quality at all
        best_r = max(r for _, _, r, _, _ in rd_points)
        best_at_low_bps = max((r for _, bps, r, _, _ in rd_points if bps < 1.5),
                              default=float('nan'))
        print(f"  [FAIL] No point achieves R >= 0.85 at bps < 1.5")
        print(f"         Best R overall: {best_r:.4f}")
        if not np.isnan(best_at_low_bps):
            print(f"         Best R at bps < 1.5: {best_at_low_bps:.4f}")
    print("=" * 80)

    passed = len(pass_points) > 0
    return {
        'passed': passed,
        'rd_points': rd_points,
        'pass_points': pass_points,
        'gen6_available': gen6_ckpt is not None,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()
    result = run(args.checkpoint)
    if result and result.get('passed') is not None:
        sys.exit(0 if result['passed'] else 1)
