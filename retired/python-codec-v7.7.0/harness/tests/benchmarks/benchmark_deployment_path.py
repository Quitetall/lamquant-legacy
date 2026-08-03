#!/usr/bin/env python3
"""
LamQuant Gen 7 — Deployment Path Benchmark
===========================================
Tests the deployment signal path: ternary encoder on RP2350, decoder on base station.

Two decoder options are tested:
  Route A: Student's own decoder (always available, no teacher needed)
  Route B: Teacher's FP32 decoder (requires latent alignment, better capacity)

The encoder runs with quantize=True (ternary weights, LSQ alphas).
For Route B, the student's strided latent [B, 32, T/4] is upsampled
to [B, 32, T] via linear interpolation before feeding the teacher decoder.
"""
import torch
import torch.nn.functional as F
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
sys.path.append(os.path.join(ROOT_DIR, 'ai_models', 'oracle'))

from lamquant_codec.models.encoder import TernaryMobileNetV5_Subband
try:
    from train_teacher import FP32OracleAutoEncoder
    HAS_TEACHER_MODULE = True
except Exception as _e:
    FP32OracleAutoEncoder = None
    HAS_TEACHER_MODULE = False
    print(f"[!] train_teacher not importable: {_e}")


def compute_metrics(x, recon):
    """Per-channel R, PRD, SNR between [1, C, T] tensors."""
    r_vals = []
    for ch in range(x.shape[1]):
        xa = x[0, ch].cpu().numpy()
        ya = recon[0, ch].cpu().numpy()
        min_t = min(len(xa), len(ya))
        xa, ya = xa[:min_t], ya[:min_t]
        if np.std(xa) < 1e-6 or np.std(ya) < 1e-6:
            continue
        r, _ = pearsonr(xa, ya)
        if not np.isnan(r):
            r_vals.append(r)
    mean_r = np.mean(r_vals) if r_vals else 0.0

    min_t = min(x.shape[2], recon.shape[2])
    diff = x[:, :, :min_t] - recon[:, :, :min_t]
    prd = (torch.sqrt(torch.mean(diff**2)) /
           (torch.sqrt(torch.mean(x[:, :, :min_t]**2)) + 1e-8)).item() * 100
    snr = (10 * torch.log10(
        torch.mean(x[:, :, :min_t]**2) /
        (torch.mean(diff**2) + 1e-12))).item()

    return round(mean_r, 4), round(prd, 2), round(snr, 1)


def load_patient_slice(path):
    """Load a 313-sample interictal L3 subband window from a Q31 NPZ file."""
    with np.load(path) as data:
        l3 = data['l3']          # [N_windows, 21, 313], float32
        seizure_mask = data['seizure_mask']
        n_windows = l3.shape[0]
        approx_stride = max(1, len(seizure_mask) // n_windows)
        win_idx = 0
        for i in range(n_windows):
            start_s = i * approx_stride
            end_s = min(start_s + approx_stride, len(seizure_mask))
            if not np.any(seizure_mask[start_s:end_s]):
                win_idx = i
                break
        return l3[win_idx:win_idx + 1]  # [1, 21, 313]


def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Deployment Path Benchmark on {device}")

    # Load student
    s_path = os.path.join(ROOT_DIR, 'ai_models/student/student_hardened.ckpt')
    if not os.path.exists(s_path):
        print(f"[SKIP] Student checkpoint not found: {s_path}")
        print("[SKIP] Benchmark Deployment Path requires a trained student_hardened.ckpt.")
        return None

    PATIENTS = [
        ('chb15', 'ai_models/dataset_sim/q31_events/chbmit_chb15_01_q31.npz'),
        ('chb16', 'ai_models/dataset_sim/q31_events/chbmit_chb16_01_q31.npz'),
        ('chb17', 'ai_models/dataset_sim/q31_events/chbmit_chb17a_03_q31.npz'),
        ('chb18', 'ai_models/dataset_sim/q31_events/chbmit_chb18_01_q31.npz'),
        ('chb19', 'ai_models/dataset_sim/q31_events/chbmit_chb19_01_q31.npz'),
        ('chb20', 'ai_models/dataset_sim/q31_events/chbmit_chb20_01_q31.npz'),
    ]
    missing = [p for _, p in PATIENTS
               if not os.path.exists(os.path.join(ROOT_DIR, p))]
    if missing:
        print(f"[SKIP] Missing patient files ({len(missing)}):")
        for p in missing:
            print(f"         {p}")
        print("[SKIP] Benchmark Deployment Path requires ai_models/dataset_sim/q31_events/chbmit_*.npz.")
        return None

    student = TernaryMobileNetV5_Subband.from_checkpoint(s_path, device=device).eval()

    # Load teacher (optional — Route B)
    has_teacher = False
    if HAS_TEACHER_MODULE:
        try:
            teacher = FP32OracleAutoEncoder().to(device).eval()
            t_enc = os.path.join(ROOT_DIR, 'ai_models/oracle/teacher_best.ckpt')
            t_dec = os.path.join(ROOT_DIR, 'ai_models/oracle/decoder_best.ckpt')
            if os.path.exists(t_enc) and os.path.exists(t_dec):
                sd = {}
                for k, v in torch.load(t_enc, map_location=device).items():
                    sd[f"encoder.{k}"] = v
                for k, v in torch.load(t_dec, map_location=device).items():
                    sd[f"decoder.{k}"] = v
                teacher.load_state_dict(sd)
                has_teacher = True
            else:
                print("[*] Teacher checkpoints not present — Route B will be skipped.")
        except Exception as e:
            print(f"[!] Teacher not loaded: {e}")
    else:
        print("[*] Teacher module unavailable — Route B will be skipped.")

    # Check encoder output shape (use 313 samples — L3 subband window length)
    with torch.no_grad():
        test = torch.randn(1, 21, 313).to(device)
        lat = student.encode(test, quantize=True)
        stride = test.shape[2] // lat.shape[2]
        print(f"[*] Student encoder: {list(test.shape)} → {list(lat.shape)} (stride {stride}x)")

    # ======================== ROUTE A: Student Decoder ========================
    print(f"\n{'='*70}")
    print(f" ROUTE A: Ternary Encoder → Student Decoder (on base station)")
    print(f"{'='*70}")
    print(f"{'Subject':<10} {'R':>8} {'PRD':>8} {'SNR':>8}")
    print("-" * 40)

    route_a_results = []
    for subject, rel_path in PATIENTS:
        path = os.path.join(ROOT_DIR, rel_path)
        sig = load_patient_slice(path)
        x = torch.from_numpy(sig).float().to(device)
        x_ac = x - torch.mean(x, dim=2, keepdim=True)
        x_shackle = torch.clamp(x_ac, -50.0, 50.0)

        with torch.no_grad():
            # Full student round-trip (encoder + decoder)
            recon = student(x_shackle, quantize=True)
            recon_ac = recon - torch.mean(recon, dim=2, keepdim=True)

        r, prd, snr = compute_metrics(x_shackle, recon_ac)
        print(f"{subject:<10} {r:>7.4f} {prd:>7.1f}% {snr:>7.1f}dB")
        route_a_results.append({
            'subject': subject, 'r': r, 'prd': prd, 'snr': snr
        })

    a_rs = [r['r'] for r in route_a_results]
    print(f"\n[*] Route A — Mean R: {np.mean(a_rs):.4f}  Min R: {min(a_rs):.4f}")

    # ======================== ROUTE B: Teacher Decoder ========================
    route_b_results = []
    if has_teacher:
        print(f"\n{'='*70}")
        print(f" ROUTE B: Ternary Encoder → Teacher FP32 Decoder (on base station)")
        print(f" (Student latent upsampled {stride}x to match teacher)")
        print(f"{'='*70}")
        print(f"{'Subject':<10} {'R':>8} {'PRD':>8} {'SNR':>8}")
        print("-" * 40)

        for subject, rel_path in PATIENTS:
            path = os.path.join(ROOT_DIR, rel_path)
            sig = load_patient_slice(path)
            x = torch.from_numpy(sig).float().to(device)
            x_ac = x - torch.mean(x, dim=2, keepdim=True)
            x_shackle = torch.clamp(x_ac, -50.0, 50.0)

            with torch.no_grad():
                latent = student.encode(x_shackle, quantize=True)
                # Upsample strided latent to full T for teacher decoder
                lat_up = F.interpolate(latent, size=x_shackle.shape[2],
                                       mode='linear', align_corners=False)
                recon = teacher.decoder(lat_up)
                recon_ac = recon - torch.mean(recon, dim=2, keepdim=True)

            r, prd, snr = compute_metrics(x_shackle, recon_ac)
            print(f"{subject:<10} {r:>7.4f} {prd:>7.1f}% {snr:>7.1f}dB")
            route_b_results.append({
                'subject': subject, 'r': r, 'prd': prd, 'snr': snr
            })

        b_rs = [r['r'] for r in route_b_results]
        print(f"\n[*] Route B — Mean R: {np.mean(b_rs):.4f}  Min R: {min(b_rs):.4f}")
    else:
        print(f"\n[*] Route B skipped (no teacher checkpoint)")

    # ======================== SUMMARY ========================
    print(f"\n{'='*70}")
    print(f" SUMMARY")
    print(f"{'='*70}")
    print(f"  Route A (Student Decoder):  Mean R = {np.mean(a_rs):.4f}  Min R = {min(a_rs):.4f}")
    if route_b_results:
        b_rs = [r['r'] for r in route_b_results]
        print(f"  Route B (Teacher Decoder):  Mean R = {np.mean(b_rs):.4f}  Min R = {min(b_rs):.4f}")
        better = "A" if np.mean(a_rs) > np.mean(b_rs) else "B"
        print(f"  Recommended: Route {better}")

    # Pass/fail on Route A (always available)
    R_FLOOR = 0.85
    min_a = min(a_rs)
    if min_a >= R_FLOOR:
        print(f"\n[PASS] Deployment Path (Route A Min R = {min_a:.4f} >= {R_FLOOR})")
    else:
        print(f"\n[FAIL] Deployment Path (Route A Min R = {min_a:.4f} < {R_FLOOR})")
        sys.exit(1)

    return route_a_results, route_b_results


if __name__ == "__main__":
    run()
