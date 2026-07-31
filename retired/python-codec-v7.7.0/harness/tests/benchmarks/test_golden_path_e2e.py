#!/usr/bin/env python3
"""
LamQuant Gen 7 — Golden Path End-to-End Test
=============================================
Validates the complete encode→compress→decompress→decode pipeline.

Cycle per patient:
  1. Load interictal EEG window (Q31 → float)
  2. Student encoder (quantize=True) → latent [B, 32, T/8]
  3. FSQ quantization (L=16) → bin indices
  4. rANS entropy coding → byte payload
  5. rANS decode → FSQ inverse → latent
  6. Student decoder → reconstructed [B, 21, T]
  7. Quality metrics: Pearson R, PRD (%), SNR (dB)

Pass criteria:
  - R ≥ 0.85
  - PRD ≤ 40%
  - Compression ratio ≥ 5.0x
"""
import os
import sys
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'ai_models', 'student'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'ai_models'))

from lamquant_codec.models.encoder import TernaryMobileNetV5_Subband
from precompute_l3_fast import preprocess_subband_single

# --- Metrics ---

def pearson_r(x, y):
    """Per-channel Pearson correlation, averaged."""
    rs = []
    for ch in range(x.shape[1]):
        a = x[0, ch].numpy()
        b = y[0, ch].numpy()
        a = a - a.mean()
        b = b - b.mean()
        denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
        if denom < 1e-12:
            rs.append(0.0)
        else:
            rs.append(float(np.sum(a * b) / denom))
    return np.mean(rs)


def prd_percent(original, reconstructed):
    """Percent Root-mean-square Difference."""
    diff = original - reconstructed
    num = torch.sqrt(torch.mean(diff ** 2))
    den = torch.sqrt(torch.mean(original ** 2))
    if den < 1e-12:
        return 0.0
    return float(num / den * 100.0)


def snr_db(original, reconstructed):
    """Signal-to-Noise Ratio in dB."""
    noise = original - reconstructed
    sig_power = torch.mean(original ** 2)
    noise_power = torch.mean(noise ** 2)
    if noise_power < 1e-12:
        return 100.0
    return float(10.0 * torch.log10(sig_power / noise_power))


# --- Simple rANS codec (Python reference) ---

def build_freq_table(latent_flat, num_levels=16, total_freq=4096):
    """Build frequency table from latent distribution."""
    vmin, vmax = float(latent_flat.min()), float(latent_flat.max())
    span = vmax - vmin + 1e-8
    normalized = (latent_flat - vmin) / span
    bins = np.clip((normalized * num_levels).astype(np.int32), 0, num_levels - 1)
    counts = np.bincount(bins, minlength=num_levels)
    freq = np.maximum(1, (counts / counts.sum() * total_freq).astype(np.int32))
    diff = total_freq - freq.sum()
    freq[np.argmax(freq)] += diff
    start = np.zeros(num_levels, dtype=np.int32)
    for i in range(1, num_levels):
        start[i] = start[i - 1] + freq[i - 1]
    return freq, start, vmin, vmax


def fsq_quantize(latent_flat, vmin, vmax, num_levels=16):
    """Uniform scalar quantization."""
    span = vmax - vmin + 1e-8
    normalized = (latent_flat - vmin) / span
    return np.clip((normalized * num_levels).astype(np.int32), 0, num_levels - 1)


def fsq_dequantize(bins, vmin, vmax, num_levels=16):
    """Inverse: bin index → center of bin."""
    span = vmax - vmin + 1e-8
    centers = vmin + (bins.astype(np.float32) + 0.5) * span / num_levels
    return centers


def rans_encode(symbols, freq, start, total_freq=4096):
    """Simple rANS encoder (returns byte list)."""
    RANS_L = 1 << 15
    state = RANS_L
    output_bytes = []

    for sym in reversed(symbols):
        f = int(freq[sym])
        s = int(start[sym])
        x_max = (RANS_L // total_freq) * f
        if x_max == 0:
            x_max = 1
        while state >= x_max * (1 << 16):
            output_bytes.append(state & 0xFF)
            state >>= 8
        state = ((state // f) * total_freq) + s + (state % f)

    # Flush state
    for _ in range(4):
        output_bytes.append(state & 0xFF)
        state >>= 8

    return output_bytes


def rans_decode(encoded_bytes, num_symbols, freq, start, total_freq=4096):
    """Simple rANS decoder."""
    RANS_L = 1 << 15
    byte_idx = len(encoded_bytes) - 1

    # Read initial state
    state = 0
    for i in range(4):
        if byte_idx >= 0:
            state = (state << 8) | encoded_bytes[byte_idx]
            byte_idx -= 1

    symbols = []
    for _ in range(num_symbols):
        # Find symbol from cumulative freq
        slot = state % total_freq
        sym = 0
        for s in range(len(start)):
            if s + 1 < len(start) and start[s + 1] <= slot:
                continue
            sym = s
            break

        f = int(freq[sym])
        s_val = int(start[sym])
        state = f * (state // total_freq) + (state % total_freq) - s_val

        # Renormalize
        while state < RANS_L and byte_idx >= 0:
            state = (state << 8) | encoded_bytes[byte_idx]
            byte_idx -= 1

        symbols.append(sym)

    return symbols


# --- Main E2E Test ---

def load_dataset():
    """Load held-out Q31 dataset, applying L3 subband preprocessing."""
    data_dir = os.path.join(ROOT_DIR, 'ai_models', 'dataset_sim', 'q31_held_out')
    if not os.path.isdir(data_dir):
        print(f"[SKIP] Dataset directory not found: {data_dir}")
        return []

    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
    if not files:
        print("[SKIP] No .npz files found in q31_held_out/")
        return []

    datasets = []
    for fname in files:
        npz = np.load(os.path.join(data_dir, fname))
        if 'l3' in npz:
            # File already has precomputed L3 subband (first window)
            eeg = npz['l3'][0] if npz['l3'].ndim == 3 else npz['l3']  # [21, 313]
        else:
            # Raw Q31 int32 data — normalize then apply L3 preprocessing
            raw = npz['data'].astype(np.float32)  # [21, 2500]
            eeg_mv = raw / 2147483647.0 * 1000.0
            eeg, _, _ = preprocess_subband_single(eeg_mv)  # [21, 313]
        # DC removal per channel + silicon shackle
        eeg = eeg - eeg.mean(axis=1, keepdims=True)
        eeg = np.clip(eeg, -50.0, 50.0)
        datasets.append((fname, torch.tensor(eeg).unsqueeze(0)))  # [1, 21, 313]

    return datasets


def run_golden_path_e2e():
    """Run the full golden path end-to-end test."""
    print("=" * 60)
    print("LamQuant Gen 7 — Golden Path E2E Test")
    print("=" * 60)

    # Load model
    ckpt_path = os.path.join(ROOT_DIR, 'ai_models', 'student', 'student_hardened.ckpt')
    if not os.path.exists(ckpt_path):
        print(f"[SKIP] Student checkpoint not found: {ckpt_path}")
        return True  # Skip gracefully

    model = TernaryMobileNetV5_Subband.from_checkpoint(ckpt_path, device='cpu').eval()

    datasets = load_dataset()
    if not datasets:
        print("[SKIP] No test data available")
        return True

    num_levels = 16
    total_freq = 4096
    all_pass = True
    results = []

    with torch.no_grad():
        for fname, x in datasets:
            T = x.shape[2]  # 313 (L3 subband)
            raw_bytes = 21 * T * 4  # 21ch × T × 4 bytes (float32 L3)

            # 1. Encode
            latent = model.encode(x, quantize=True)  # [1, 32, T/4]
            lat_np = latent.numpy().flatten()

            # 2. FSQ quantize
            freq, start, vmin, vmax = build_freq_table(lat_np, num_levels, total_freq)
            symbols = fsq_quantize(lat_np, vmin, vmax, num_levels)

            # 3. rANS compress
            encoded = rans_encode(symbols, freq, start, total_freq)
            compressed_bytes = len(encoded) + 5 + 4  # header + range params + payload
            cr = raw_bytes / compressed_bytes

            # 4. rANS decompress
            decoded_symbols = rans_decode(encoded, len(symbols), freq, start, total_freq)

            # 5. FSQ dequantize
            lat_reconstructed = fsq_dequantize(
                np.array(decoded_symbols), vmin, vmax, num_levels)
            lat_tensor = torch.tensor(
                lat_reconstructed.reshape(latent.shape), dtype=torch.float32)

            # 6. Decode (target_len required by new model to trim ConvTranspose1d output)
            recon = model.decode(lat_tensor, target_len=T)  # [1, 21, T]

            # Handle potential length mismatch from stride rounding
            min_T = min(x.shape[2], recon.shape[2])
            x_trim = x[:, :, :min_T]
            recon_trim = recon[:, :, :min_T]

            # 7. Quality metrics
            r = pearson_r(x_trim, recon_trim)
            prd = prd_percent(x_trim, recon_trim)
            snr = snr_db(x_trim, recon_trim)

            results.append({
                'file': fname, 'r': r, 'prd': prd, 'snr': snr,
                'cr': cr, 'bytes': compressed_bytes,
            })

            status = "PASS" if (r >= 0.85 and prd <= 40.0 and cr >= 5.0) else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  {fname}: R={r:.4f}  PRD={prd:.1f}%  SNR={snr:.1f}dB  "
                  f"CR={cr:.1f}x  [{status}]")

    # Summary
    print("\n--- Summary ---")
    mean_r = np.mean([r['r'] for r in results])
    mean_prd = np.mean([r['prd'] for r in results])
    mean_cr = np.mean([r['cr'] for r in results])
    print(f"  Mean R: {mean_r:.4f}  Mean PRD: {mean_prd:.1f}%  Mean CR: {mean_cr:.1f}x")
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


if __name__ == "__main__":
    success = run_golden_path_e2e()
    sys.exit(0 if success else 1)
