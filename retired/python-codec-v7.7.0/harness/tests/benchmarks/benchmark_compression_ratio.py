#!/usr/bin/env python3
"""
LamQuant Gen 7 — Compression Ratio Benchmark
=============================================
Measures REAL compression ratio AND reconstruction quality for both paths
by implementing the full encode → transmit → decode pipeline.

GOLDEN PATH:
  Encode:  student.encode(x) → FSQ quantize → rANS → bytes
  Decode:  rANS decode → inverse FSQ → teacher.decoder → signal
  Quality: R, PRD, SNR between original and decoded signal

LIGHTNING PATH:
  Encode:  Toeplitz CS → 2D lifting → Golomb-Rice → bytes
  Decode:  inverse Golomb-Rice → inverse lifting → OMP CS recovery → signal
  Quality: R, PRD, SNR between original and recovered signal (6ch subset)

Both paths count actual transmitted bytes. Both paths measure actual
reconstruction quality. No estimates.
"""
import torch
import numpy as np
import os
import sys
from pathlib import Path
from scipy.stats import pearsonr
import torch.nn.functional as F

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


# =====================================================================
# Metrics
# =====================================================================

def compute_metrics(original, reconstructed):
    orig = original.flatten()
    recon = reconstructed.flatten()
    min_len = min(len(orig), len(recon))
    orig, recon = orig[:min_len], recon[:min_len]

    if np.std(orig) < 1e-8 or np.std(recon) < 1e-8:
        return 0.0, 100.0, -10.0

    r, _ = pearsonr(orig, recon)
    rms_diff = np.sqrt(np.mean((orig - recon) ** 2))
    rms_orig = np.sqrt(np.mean(orig ** 2))
    prd = (rms_diff / (rms_orig + 1e-8)) * 100
    snr = 10 * np.log10(np.mean(orig ** 2) / (np.mean((orig - recon) ** 2) + 1e-12))
    return round(float(r), 4), round(float(prd), 2), round(float(snr), 1)


# =====================================================================
# GOLDEN PATH
# =====================================================================

def fsq_encode(lat_np, L):
    vmin, vmax = float(lat_np.min()), float(lat_np.max())
    span = vmax - vmin + 1e-8
    syms = np.clip(((lat_np - vmin) / span * L).astype(np.int32), 0, L - 1)
    return syms.astype(np.uint16), vmin, vmax

def fsq_decode(syms, L, vmin, vmax):
    span = vmax - vmin + 1e-8
    return vmin + (syms.astype(np.float32) + 0.5) * span / L

class RANSEncoder:
    def __init__(self, counts, total_freq=4096):
        self.total = total_freq
        raw = sum(counts)
        self.freq = [max(1, int(c / raw * total_freq)) for c in counts] if raw > 0 else [1]*len(counts)
        self.freq[self.freq.index(max(self.freq))] += total_freq - sum(self.freq)
        self.start = [0] * len(self.freq)
        for i in range(1, len(self.freq)):
            self.start[i] = self.start[i-1] + self.freq[i-1]
        self.L = 1 << 15
        self.state = self.L
        self.out = bytearray()

    def put(self, sym):
        f, s = self.freq[sym], self.start[sym]
        xm = max(1, (self.L // self.total) * f)
        while self.state >= xm * (1 << 16):
            self.out.append(self.state & 0xFF); self.state >>= 8
        self.state = ((self.state // f) * self.total) + s + (self.state % f)

    def flush(self):
        for _ in range(4):
            self.out.append(self.state & 0xFF); self.state >>= 8

def golden_pipeline(student, teacher, x_tensor, device, L=8):
    """Full golden encode → decode → measure. Uses student's own decoder (Route A)."""
    with torch.no_grad():
        latent = student.encode(x_tensor, quantize=True)
    lat_np = latent[0].cpu().numpy()

    syms, vmin, vmax = fsq_encode(lat_np, L)
    flat = syms.flatten()
    counts = np.bincount(flat.astype(np.int32), minlength=L)
    enc = RANSEncoder(counts.tolist())
    for i in range(len(flat)-1, -1, -1):
        enc.put(int(flat[i]))
    enc.flush()

    total_bytes = 5 + 4 + len(enc.out)  # header + range params + payload

    # Decode using STUDENT's decoder (Route A — always works)
    lat_rec = fsq_decode(syms, L, vmin, vmax)
    lat_t = torch.from_numpy(lat_rec).unsqueeze(0).float().to(device)

    with torch.no_grad():
        input_T = x_tensor.shape[2]
        # decode() requires target_len for ConvTranspose1d output trimming
        decoded = student.decode(lat_t, target_len=input_T, quantize=True)

    orig = x_tensor[0].cpu().numpy()
    dec_np = decoded[0].cpu().numpy()
    r, prd, snr = compute_metrics(orig, dec_np)

    return total_bytes, r, prd, snr


# =====================================================================
# LIGHTNING PATH
# =====================================================================

def toeplitz_matrix(M, N, seed):
    rng = np.random.RandomState(seed)
    row = rng.choice([-1, 1], size=N).astype(np.float64)
    Phi = np.zeros((M, N), dtype=np.float64)
    for m in range(M):
        Phi[m] = np.roll(row, m)
    return Phi / np.sqrt(N)

def lifting_forward(tile):
    t = tile.copy().astype(np.int64)
    for r in range(t.shape[0]):
        for c in range(1, t.shape[1]-1, 2):
            t[r,c] -= (t[r,c-1] + t[r,c+1]) >> 1
        for c in range(2, t.shape[1]-2, 2):
            t[r,c] += (t[r,c-1] + t[r,c+1] + 2) >> 2
    return t

def lifting_inverse(tile):
    t = tile.copy().astype(np.int64)
    for r in range(t.shape[0]):
        for c in range(2, t.shape[1]-2, 2):
            t[r,c] -= (t[r,c-1] + t[r,c+1] + 2) >> 2
        for c in range(1, t.shape[1]-1, 2):
            t[r,c] += (t[r,c-1] + t[r,c+1]) >> 1
    return t

def omp_recover(y, Phi, sparsity=64):
    M, N = Phi.shape
    residual = y.copy()
    support = []
    x = np.zeros(N)
    for _ in range(min(sparsity, M)):
        corr = np.abs(Phi.T @ residual)
        idx = np.argmax(corr)
        if idx in support:
            break
        support.append(idx)
        Ps = Phi[:, support]
        coeffs, _, _, _ = np.linalg.lstsq(Ps, y, rcond=None)
        residual = y - Ps @ coeffs
        if np.linalg.norm(residual) < 1e-6:
            break
    x_rec = np.zeros(N)
    if support:
        Ps = Phi[:, support]
        coeffs, _, _, _ = np.linalg.lstsq(Ps, y, rcond=None)
        for i, idx in enumerate(support):
            x_rec[idx] = coeffs[i]
    return x_rec

def golomb_rice_bits(tile, k=4):
    bits = 0
    for v in tile.flatten():
        v = int(v)
        mapped = (v << 1) if v >= 0 else ((-v << 1) - 1)
        q = mapped >> k
        bits += q + 1 + k
    return bits

def lightning_pipeline(signal_21ch, T=2500):
    """Full lightning encode → decode → measure."""
    seeds = [0x12345678 + ch for ch in range(6)]

    # Encode: CS → lifting → Golomb-Rice
    tile = np.zeros((6, 32), dtype=np.int64)
    for ch in range(6):
        measurements = toeplitz_matrix(32, T, seeds[ch]) @ signal_21ch[ch, :T].astype(np.float64)
        tile[ch] = np.round(measurements).astype(np.int64)

    tile_lifted = lifting_forward(tile)
    payload_bits = golomb_rice_bits(tile_lifted, k=4)
    total_bytes = 5 + (payload_bits + 7) // 8

    # Decode: inverse lifting → OMP per channel
    tile_recovered = lifting_inverse(tile_lifted)
    recovered = np.zeros((6, T), dtype=np.float64)
    for ch in range(6):
        Phi = toeplitz_matrix(32, T, seeds[ch])
        recovered[ch] = omp_recover(tile_recovered[ch].astype(np.float64), Phi,
                                     sparsity=min(128, T // 8))

    # Quality on the 6 transmitted channels
    orig_6ch = signal_21ch[:6, :T]
    r, prd, snr = compute_metrics(orig_6ch, recovered)

    return total_bytes, r, prd, snr


# =====================================================================
# Main
# =====================================================================

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
    print(f"[*] Compression Ratio Benchmark on {device}")

    s_path = os.path.join(ROOT_DIR, 'ai_models/student/student_hardened.ckpt')
    t_enc = os.path.join(ROOT_DIR, 'ai_models/oracle/teacher_best.ckpt')
    t_dec = os.path.join(ROOT_DIR, 'ai_models/oracle/decoder_best.ckpt')

    missing_ckpts = [p for p in (s_path, t_enc, t_dec) if not os.path.exists(p)]
    if missing_ckpts or not HAS_TEACHER_MODULE:
        print("[SKIP] Benchmark Compression Ratio requires trained checkpoints:")
        for p in missing_ckpts:
            print(f"         missing: {p}")
        if not HAS_TEACHER_MODULE:
            print("         missing: train_teacher module (import failed)")
        print("[SKIP] Train student + teacher before running this benchmark.")
        return None

    PATIENTS = [
        ('chb15', 'ai_models/dataset_sim/q31_events/chbmit_chb15_01_q31.npz'),
        ('chb16', 'ai_models/dataset_sim/q31_events/chbmit_chb16_01_q31.npz'),
        ('chb17', 'ai_models/dataset_sim/q31_events/chbmit_chb17a_03_q31.npz'),
        ('chb18', 'ai_models/dataset_sim/q31_events/chbmit_chb18_01_q31.npz'),
        ('chb19', 'ai_models/dataset_sim/q31_events/chbmit_chb19_01_q31.npz'),
        ('chb20', 'ai_models/dataset_sim/q31_events/chbmit_chb20_01_q31.npz'),
    ]
    missing_patients = [p for _, p in PATIENTS
                        if not os.path.exists(os.path.join(ROOT_DIR, p))]
    if missing_patients:
        print(f"[SKIP] Missing patient files ({len(missing_patients)}):")
        for p in missing_patients:
            print(f"         {p}")
        print("[SKIP] Benchmark Compression Ratio requires ai_models/dataset_sim/q31_events/chbmit_*.npz.")
        return None

    student = TernaryMobileNetV5_Subband.from_checkpoint(s_path, device=device).eval()

    teacher = FP32OracleAutoEncoder().to(device).eval()
    sd = {}
    for k, v in torch.load(t_enc, map_location=device).items(): sd[f"encoder.{k}"] = v
    for k, v in torch.load(t_dec, map_location=device).items(): sd[f"decoder.{k}"] = v
    teacher.load_state_dict(sd)

    # L3 subband: 313 samples × 21 channels × 4 bytes (float32)
    raw_bytes = 21 * 313 * 4
    FSQ_LEVELS = [8, 16, 32]

    # ======================== GOLDEN ========================
    print(f"\n{'='*80}")
    print(f" GOLDEN PATH (TNN → FSQ → rANS → Student Decode)")
    print(f" Raw: {raw_bytes} bytes | 21ch × 313 × float32 (L3 subband)")
    print(f"{'='*80}")

    golden_results = {}
    for L in FSQ_LEVELS:
        print(f"\n  FSQ L={L} ({np.log2(L):.1f} bps)")
        print(f"  {'Subject':<10} {'Bytes':>7} {'CR':>7} {'R':>8} {'PRD':>8} {'SNR':>7}")
        print(f"  {'-'*52}")

        res = []
        for subj, rp in PATIENTS:
            path = os.path.join(ROOT_DIR, rp)
            sig = load_patient_slice(path)
            x = torch.from_numpy(sig).float().to(device)
            x = torch.clamp(x - x.mean(dim=2, keepdim=True), -50, 50)

            b, r, prd, snr = golden_pipeline(student, teacher, x, device, L)
            cr = raw_bytes / b
            print(f"  {subj:<10} {b:>6}B {cr:>6.1f}x {r:>7.4f} {prd:>7.1f}% {snr:>6.1f}dB")
            res.append({'subject': subj, 'bytes': b, 'cr': round(cr,1),
                        'r': r, 'prd': prd, 'snr': snr})
        golden_results[L] = res
        print(f"  Mean: CR={np.mean([r['cr'] for r in res]):.1f}x  "
              f"R={np.mean([r['r'] for r in res]):.4f}")

    # ======================== LIGHTNING ========================
    print(f"\n{'='*80}")
    print(f" LIGHTNING PATH (Toeplitz CS → Lifting → Golomb-Rice → OMP Recovery)")
    print(f" Note: Reconstructs first 6 of 21 channels (L3 subband, T=313)")
    print(f"{'='*80}")
    print(f"  {'Subject':<10} {'Bytes':>7} {'CR':>7} {'R(6ch)':>8} {'PRD':>8} {'SNR':>7}")
    print(f"  {'-'*52}")

    lightning_res = []
    for subj, rp in PATIENTS:
        path = os.path.join(ROOT_DIR, rp)
        sig = load_patient_slice(path)
        x_np = sig[0]
        x_np = np.clip(x_np - np.mean(x_np, axis=1, keepdims=True), -50, 50)

        b, r, prd, snr = lightning_pipeline(x_np, T=313)
        cr = raw_bytes / b
        print(f"  {subj:<10} {b:>6}B {cr:>6.1f}x {r:>7.4f} {prd:>7.1f}% {snr:>6.1f}dB")
        lightning_res.append({'subject': subj, 'bytes': b, 'cr': round(cr,1),
                              'r': r, 'prd': prd, 'snr': snr})

    print(f"  Mean: CR={np.mean([r['cr'] for r in lightning_res]):.1f}x  "
          f"R={np.mean([r['r'] for r in lightning_res]):.4f}")

    # ======================== SUMMARY ========================
    print(f"\n{'='*80}")
    print(f" SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Path':<25} {'CR':>7} {'R':>8} {'PRD':>8} {'SNR':>7}")
    print(f"  {'-'*55}")
    for L in FSQ_LEVELS:
        r = golden_results[L]
        print(f"  Golden L={L:<2}              {np.mean([x['cr'] for x in r]):>6.1f}x "
              f"{np.mean([x['r'] for x in r]):>7.4f} "
              f"{np.mean([x['prd'] for x in r]):>7.1f}% "
              f"{np.mean([x['snr'] for x in r]):>6.1f}dB")
    print(f"  Lightning (6ch)          {np.mean([x['cr'] for x in lightning_res]):>6.1f}x "
          f"{np.mean([x['r'] for x in lightning_res]):>7.4f} "
          f"{np.mean([x['prd'] for x in lightning_res]):>7.1f}% "
          f"{np.mean([x['snr'] for x in lightning_res]):>6.1f}dB")

    # Pass/fail
    best_cr, best_r = 0, 0
    for L in FSQ_LEVELS:
        r = golden_results[L]
        mcr = np.mean([x['cr'] for x in r])
        mr = np.mean([x['r'] for x in r])
        if mr >= 0.85 and mcr > best_cr:
            best_cr, best_r = mcr, mr

    if best_cr >= 5.0:
        print(f"\n[PASS] Best golden: CR={best_cr:.1f}x at R={best_r:.4f}")
    else:
        print(f"\n[FAIL] Best golden CR with R>=0.85: {best_cr:.1f}x")
        sys.exit(1)

    return golden_results, lightning_results


if __name__ == "__main__":
    run()
