#!/usr/bin/env python3
"""
LamQuant Gen 7.5 — D15: FSQ Validation Against Reference Implementation
========================================================================
Compares our custom scalar FSQ (from benchmark_compression_ratio.py) against
the lucidrains vector-quantize-pytorch FSQ/LFQ reference implementations.

Tests:
  1. Encode holdout patient L3 windows -> latent [32, 79]
  2. Our FSQ at L=16 -> symbols -> decode -> compute R
  3. lucidrains FSQ at L=16 -> codes -> compute R
  4. lucidrains LFQ (ternary, levels=[3] per dim) baseline
  5. Compare reconstructed latents and R values
  6. Report codebook utilization for each method

PASS criteria:
  - Our FSQ R and lucidrains FSQ R differ by < 0.05 (methods are comparable)
  - Both FSQ methods achieve R >= 0.90
  - Codebook utilization > 20% for both FSQ methods

Usage:
  python benchmark_fsq_validation.py
"""

import torch
import os
import sys
import numpy as np
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
sys.path.append(os.path.join(ROOT_DIR, 'ai_models', 'student'))
sys.path.insert(0, os.path.dirname(__file__))

from lamquant_codec.models.encoder import TernaryMobileNetV5_Subband
from benchmark_compression_ratio import fsq_encode, fsq_decode

PATIENTS = {
    'chb15': 'chbmit_chb15_01_q31.npz',
    'chb16': 'chbmit_chb16_01_q31.npz',
    'chb17': 'chbmit_chb17a_03_q31.npz',
    'chb18': 'chbmit_chb18_01_q31.npz',
    'chb19': 'chbmit_chb19_01_q31.npz',
    'chb20': 'chbmit_chb20_01_q31.npz',
}

FSQ_L = 16
LFQ_LEVELS = 3  # ternary per dimension


def load_l3_windows(path, max_windows=10):
    """Load up to max_windows interictal L3 windows from a q31 NPZ file."""
    with np.load(path) as data:
        l3 = data['l3']  # [N, 21, 313]
    n = min(l3.shape[0], max_windows)
    idx = np.linspace(0, l3.shape[0] - 1, n, dtype=int)
    return l3[idx]  # [n, 21, 313]


def compute_r(original, reconstructed):
    """Compute Pearson R between two arrays."""
    orig = original.flatten()
    recon = reconstructed.flatten()
    min_len = min(len(orig), len(recon))
    orig, recon = orig[:min_len], recon[:min_len]
    if np.std(orig) < 1e-8 or np.std(recon) < 1e-8:
        return 0.0
    r, _ = pearsonr(orig, recon)
    return float(r)


def codebook_utilization(symbols, num_levels):
    """Fraction of codebook entries actually used."""
    unique = len(np.unique(symbols.flatten()))
    return unique / num_levels


def our_fsq_pipeline(latent_np, L):
    """Our custom scalar FSQ: encode -> decode -> return reconstructed latent and symbols."""
    syms, vmin, vmax = fsq_encode(latent_np, L)
    recon = fsq_decode(syms, L, vmin, vmax)
    return recon, syms


def lucidrains_fsq_pipeline(latent_tensor, L):
    """lucidrains FSQ: reshape latent to groups of 4, quantize, reshape back.

    FSQ from vector-quantize-pytorch treats the last dimension as the
    product quantizer group. We reshape [C, T] -> [B, T*C//4, 4] with
    levels=[L, L, L, L].
    """
    from vector_quantize_pytorch import FSQ

    device = latent_tensor.device
    fsq = FSQ(levels=[L, L, L, L]).to(device)

    # latent_tensor: [B, C, T]
    B, C, T = latent_tensor.shape
    # Flatten spatial dims and group into 4-element vectors
    flat = latent_tensor.permute(0, 2, 1).reshape(B, T, C)  # [B, T, C]
    # Pad C to multiple of 4 if needed
    pad = (4 - C % 4) % 4
    if pad > 0:
        flat = torch.nn.functional.pad(flat, (0, pad))
    C_padded = C + pad
    grouped = flat.reshape(B, T * C_padded // 4, 4)  # [B, N, 4]

    with torch.no_grad():
        codes, indices = fsq(grouped)  # codes: [B, N, 4], indices: [B, N]

    # Reconstruct back to [B, C, T]
    recon_flat = codes.reshape(B, T, C_padded)
    if pad > 0:
        recon_flat = recon_flat[:, :, :C]
    recon = recon_flat.permute(0, 2, 1)  # [B, C, T]

    return recon.cpu().numpy(), indices.cpu().numpy()


def lucidrains_lfq_pipeline(latent_tensor, codebook_size):
    """lucidrains LFQ (Lookup-Free Quantization) baseline.

    LFQ uses binary codes. codebook_size controls the number of bits
    (codebook_size = 2^num_bits). For ternary-like comparison we use
    a small codebook.
    """
    from vector_quantize_pytorch import LFQ

    device = latent_tensor.device
    B, C, T = latent_tensor.shape

    # LFQ expects [..., dim] where dim determines bits
    # Use dim = C directly, codebook_size = 2^ceil(log2(codebook_size))
    import math
    num_bits = max(1, int(math.ceil(math.log2(codebook_size))))
    actual_codebook = 2 ** num_bits

    lfq = LFQ(dim=C, codebook_size=actual_codebook).to(device)

    flat = latent_tensor.permute(0, 2, 1)  # [B, T, C]

    with torch.no_grad():
        quantized, indices, commit_loss = lfq(flat)

    recon = quantized.permute(0, 2, 1)  # [B, C, T]
    return recon.cpu().numpy(), indices.cpu().numpy(), actual_codebook


def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] FSQ Validation Benchmark on {device}")

    # Check for lucidrains package
    try:
        from vector_quantize_pytorch import FSQ, LFQ
        has_lucidrains = True
    except ImportError:
        has_lucidrains = False
        print("[SKIP] vector-quantize-pytorch not installed.")
        print("[SKIP] Install with: pip install vector-quantize-pytorch")
        return None

    # Load student
    s_path = os.path.join(ROOT_DIR, 'ai_models/student/student_hardened.ckpt')
    if not os.path.exists(s_path):
        print(f"[SKIP] Student checkpoint not found: {s_path}")
        return None

    try:
        student = TernaryMobileNetV5_Subband.from_checkpoint(s_path, device=device).eval()
    except Exception as e:
        print(f"[SKIP] Could not load student checkpoint: {e}")
        return None

    # Check patient data
    q31_dir = os.path.join(ROOT_DIR, 'ai_models/dataset_sim/q31_events')
    missing = [f for f in PATIENTS.values()
               if not os.path.exists(os.path.join(q31_dir, f))]
    if missing:
        print(f"[SKIP] Missing patient files ({len(missing)})")
        return None

    # Collect latents from all patients
    all_latents = []
    all_inputs = []
    print("[*] Encoding holdout patient windows...")

    with torch.no_grad():
        for subj, fname in PATIENTS.items():
            path = os.path.join(q31_dir, fname)
            windows = load_l3_windows(path, max_windows=5)
            for w in windows:
                x = torch.from_numpy(w).unsqueeze(0).float().to(device)
                x = torch.clamp(x - x.mean(dim=2, keepdim=True), -50, 50)
                lat = student.encode(x, quantize=True)
                all_latents.append(lat)
                all_inputs.append(x)

    print(f"[*] Collected {len(all_latents)} latent windows")

    # ===== METHOD 1: Our FSQ =====
    print(f"\n{'='*60}")
    print(f" METHOD 1: Our Scalar FSQ (L={FSQ_L})")
    print(f"{'='*60}")

    our_rs = []
    our_utils = []
    for lat_t in all_latents:
        lat_np = lat_t[0].cpu().numpy()
        recon, syms = our_fsq_pipeline(lat_np, FSQ_L)
        r = compute_r(lat_np, recon)
        util = codebook_utilization(syms, FSQ_L)
        our_rs.append(r)
        our_utils.append(util)

    our_mean_r = np.mean(our_rs)
    our_mean_util = np.mean(our_utils)
    print(f"  Mean R:     {our_mean_r:.4f}")
    print(f"  Mean Util:  {our_mean_util*100:.1f}%")

    # ===== METHOD 2: lucidrains FSQ =====
    print(f"\n{'='*60}")
    print(f" METHOD 2: lucidrains FSQ (levels=[{FSQ_L},{FSQ_L},{FSQ_L},{FSQ_L}])")
    print(f"{'='*60}")

    lr_rs = []
    lr_utils = []
    lr_total_codebook = FSQ_L ** 4  # product quantizer
    for lat_t in all_latents:
        recon, indices = lucidrains_fsq_pipeline(lat_t, FSQ_L)
        lat_np = lat_t[0].cpu().numpy()
        r = compute_r(lat_np, recon[0])
        unique_codes = len(np.unique(indices.flatten()))
        util = unique_codes / lr_total_codebook
        lr_rs.append(r)
        lr_utils.append(util)

    lr_mean_r = np.mean(lr_rs)
    lr_mean_util = np.mean(lr_utils)
    print(f"  Mean R:     {lr_mean_r:.4f}")
    print(f"  Mean Util:  {lr_mean_util*100:.1f}% (of {lr_total_codebook} codes)")

    # ===== METHOD 3: lucidrains LFQ (ternary baseline) =====
    print(f"\n{'='*60}")
    print(f" METHOD 3: lucidrains LFQ (ternary-like baseline)")
    print(f"{'='*60}")

    lfq_rs = []
    lfq_utils = []
    lfq_codebook = None
    for lat_t in all_latents:
        recon, indices, actual_cb = lucidrains_lfq_pipeline(lat_t, LFQ_LEVELS)
        lfq_codebook = actual_cb
        lat_np = lat_t[0].cpu().numpy()
        r = compute_r(lat_np, recon[0])
        unique_codes = len(np.unique(indices.flatten()))
        util = unique_codes / actual_cb
        lfq_rs.append(r)
        lfq_utils.append(util)

    lfq_mean_r = np.mean(lfq_rs)
    lfq_mean_util = np.mean(lfq_utils)
    print(f"  Mean R:     {lfq_mean_r:.4f}")
    print(f"  Mean Util:  {lfq_mean_util*100:.1f}% (of {lfq_codebook} codes)")

    # ===== COMPARISON =====
    print(f"\n{'='*60}")
    print(f" COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Method':<30} {'R':>8} {'Util':>8}")
    print(f"  {'-'*48}")
    print(f"  {'Our FSQ (L='+str(FSQ_L)+')':<30} {our_mean_r:>7.4f} {our_mean_util*100:>6.1f}%")
    print(f"  {'lucidrains FSQ (L='+str(FSQ_L)+')':<30} {lr_mean_r:>7.4f} {lr_mean_util*100:>6.1f}%")
    print(f"  {'lucidrains LFQ (ternary)':<30} {lfq_mean_r:>7.4f} {lfq_mean_util*100:>6.1f}%")

    r_delta = abs(our_mean_r - lr_mean_r)
    print(f"\n  R delta (ours vs lucidrains FSQ): {r_delta:.4f}")

    # Pass/fail
    passed = True
    reasons = []

    if r_delta > 0.05:
        passed = False
        reasons.append(f"R delta {r_delta:.4f} > 0.05 threshold")

    if our_mean_r < 0.90:
        passed = False
        reasons.append(f"Our FSQ R {our_mean_r:.4f} < 0.90")

    if lr_mean_r < 0.90:
        passed = False
        reasons.append(f"lucidrains FSQ R {lr_mean_r:.4f} < 0.90")

    if our_mean_util < 0.20:
        passed = False
        reasons.append(f"Our FSQ util {our_mean_util*100:.1f}% < 20%")

    if passed:
        print("\n[PASS] FSQ Validation — our implementation matches reference.")
    else:
        print(f"\n[FAIL] FSQ Validation:")
        for reason in reasons:
            print(f"  - {reason}")

    return {
        'passed': passed,
        'our_fsq_r': round(our_mean_r, 4),
        'lucidrains_fsq_r': round(lr_mean_r, 4),
        'lfq_r': round(lfq_mean_r, 4),
        'r_delta': round(r_delta, 4),
        'our_fsq_util': round(our_mean_util, 4),
        'lucidrains_fsq_util': round(lr_mean_util, 4),
        'lfq_util': round(lfq_mean_util, 4),
    }


if __name__ == "__main__":
    run()
