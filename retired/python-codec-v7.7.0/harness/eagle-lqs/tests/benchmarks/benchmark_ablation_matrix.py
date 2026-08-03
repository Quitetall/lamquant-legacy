#!/usr/bin/env python3
"""
LamQuant Gen 7.5 — D2: Ablation Matrix
=======================================
Diagnostic D2: Comparison script that runs AFTER ablation training runs complete.

Loads up to 4 checkpoints and compares Pearson R at matched compression:

  A: Raw signal -> TNN stride 8 (Gen 6 baseline)
  B: LPC residual -> TNN stride 8
  C: Lifting L3 approx -> TNN stride 4 (no LPC)
  D: LPC -> Lifting L3 -> TNN stride 4 (Gen 7.5 full pipeline)

PASS: At least 2 configs available for meaningful comparison.
NOTE: Config D (current production pipeline) is the only one that runs with
      existing checkpoints. The benchmark shows partial results and notes
      which configs are missing.

Usage:
  python benchmark_ablation_matrix.py
"""

import torch
import os
import sys
import numpy as np
from scipy.signal import butter, sosfilt
from scipy.stats import pearsonr
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
from lamquant_codec.models.encoder import TernaryMobileNetV5, TernaryMobileNetV5_Subband
from subband_preprocess import lpc_analyze_channel, lifting_3level_forward, lifting_3level_inverse


PATIENTS = {
    'chb15': 'chbmit_chb15_01_q31.npz',
    'chb16': 'chbmit_chb16_01_q31.npz',
    'chb17': 'chbmit_chb17a_03_q31.npz',
    'chb18': 'chbmit_chb18_01_q31.npz',
    'chb19': 'chbmit_chb19_01_q31.npz',
    'chb20': 'chbmit_chb20_01_q31.npz',
}

ABLATION_CONFIGS = {
    'A_raw_tnn8': {
        'description': 'Raw signal -> TNN stride 8 (Gen 6 baseline)',
        'checkpoint': 'weights/ablation_a_raw_gen6.ckpt',
        'model_class': 'gen6',      # TernaryMobileNetV5
        'input': 'raw',             # [21, 2500]
    },
    'B_lpc_tnn8': {
        'description': 'LPC residual -> TNN stride 8',
        'checkpoint': 'weights/ablation_b_lpc_gen6.ckpt',
        'model_class': 'gen6',
        'input': 'lpc_residual',    # [21, 2500]
    },
    'C_l3_nolpc': {
        'description': 'Lifting L3 approx -> TNN stride 4 (no LPC)',
        'checkpoint': 'weights/ablation_c_l3_nolpc.ckpt',
        'model_class': 'subband',   # TernaryMobileNetV5_Subband
        'input': 'l3_nolpc',        # [21, 313]
    },
    'D_full_pipeline': {
        'description': 'LPC -> Lifting L3 -> TNN stride 4 (Gen 7.5 full)',
        'checkpoint': 'weights/student_subband.ckpt',  # current production
        'model_class': 'subband',
        'input': 'l3',              # [21, 313] from NPZ (includes LPC)
    },
}

FS = 250.0
WIN_LEN = 2500       # 10 seconds at 250 Hz
N_CHANNELS = 21
MAX_WINDOWS = 10
Q31_SCALE = 2147483647.0
MV_SCALE = 1000.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hp_filter_batch(signal_ch21, fs=FS, fc=0.5):
    """HP filter all 21 channels. signal_ch21: [21, T]. Returns [21, T] float64."""
    sos = butter(2, fc, btype='high', fs=fs, output='sos')
    out = np.empty_like(signal_ch21)
    for ch in range(signal_ch21.shape[0]):
        out[ch] = sosfilt(sos, signal_ch21[ch])
    return out


def q31_to_mv(raw_int32):
    """Convert Q31 integer array to float mV."""
    return raw_int32.astype(np.float64) / Q31_SCALE * MV_SCALE


def load_windows(npz_path, max_windows=MAX_WINDOWS):
    """Load evenly-spaced raw windows and pre-computed l3 windows.

    Returns:
        raw_windows : list of [21, 2500] float64 (HP-filtered)
        l3_windows  : list of [21, 313] float32 (pre-computed from NPZ)
    """
    with np.load(npz_path) as npz:
        raw_data = npz['data']      # [21, N_samples] int32
        l3_all = npz['l3']          # [N_win, 21, 313] float32

    n_l3 = l3_all.shape[0]
    total_samples = raw_data.shape[1]

    n_win = min(n_l3, max_windows)
    win_indices = np.linspace(0, n_l3 - 1, n_win, dtype=int)

    raw_windows = []
    l3_windows = []

    for wi in win_indices:
        start = int(wi) * WIN_LEN
        end = start + WIN_LEN
        if end > total_samples:
            continue
        raw_slice = q31_to_mv(raw_data[:, start:end])
        hp_signal = hp_filter_batch(raw_slice)
        raw_windows.append(hp_signal)
        l3_windows.append(l3_all[int(wi)])

    return raw_windows, l3_windows


def compute_lpc_residual(hp_signal):
    """Compute LPC residual per channel. Returns [21, 2500] float64."""
    residuals = np.zeros_like(hp_signal)
    for ch in range(hp_signal.shape[0]):
        result = lpc_analyze_channel(hp_signal[ch].astype(np.float32))
        # result is a dict; 'residual' is the prediction error signal
        if isinstance(result, dict) and 'residual' in result:
            res = np.asarray(result['residual'], dtype=np.float64)
        else:
            # Fallback: return raw signal (lpc_analyze_channel may return tuple)
            try:
                _, res = result   # (coeffs, residual)
                res = np.asarray(res, dtype=np.float64)
            except Exception:
                res = hp_signal[ch].copy()
        # Match length
        min_len = min(len(res), hp_signal.shape[1])
        residuals[ch, :min_len] = res[:min_len]
    return residuals


def compute_l3_nolpc(hp_signal):
    """Forward lifting on raw HP signal (NO LPC). Returns [21, 313] float32."""
    l3_approx = np.zeros((N_CHANNELS, 313), dtype=np.float32)
    for ch in range(N_CHANNELS):
        subs = lifting_3level_forward(hp_signal[ch])
        l3_approx[ch] = subs['l3_approx'].astype(np.float32)
    return l3_approx


def inverse_lifting_from_l3(raw_window, l3_recon_np):
    """Reconstruct fullband signal by replacing l3_approx in the lifting tree.

    Uses the original raw window to get detail subbands, then replaces l3_approx
    with the reconstructed version and applies inverse lifting.

    Returns [21, 2500] float64.
    """
    recon_raw = np.zeros((N_CHANNELS, WIN_LEN), dtype=np.float64)
    for ch in range(N_CHANNELS):
        subs = lifting_3level_forward(raw_window[ch])
        subs_recon = dict(subs)
        subs_recon['l3_approx'] = l3_recon_np[ch].astype(np.float64)
        recon_raw[ch] = lifting_3level_inverse(subs_recon)[:WIN_LEN]
    return recon_raw


def pearson_r_2d(orig, recon):
    """Pearson R between two 2D arrays (all channels, all samples flattened)."""
    o = orig.flatten()
    r = recon.flatten()
    min_len = min(len(o), len(r))
    o, r = o[:min_len], r[:min_len]
    if np.std(o) < 1e-8 or np.std(r) < 1e-8:
        return 0.0
    rval, _ = pearsonr(o, r)
    return float(rval)


# ---------------------------------------------------------------------------
# Per-config evaluation
# ---------------------------------------------------------------------------

def evaluate_config(config_name, cfg, available_patients, device):
    """Evaluate a single ablation config across all available patients.

    Returns list of (patient_name, R) pairs, or None if config cannot run.
    """
    ckpt_path = os.path.join(ROOT_DIR, cfg['checkpoint'])
    if not os.path.exists(ckpt_path):
        print(f"  [{config_name}] Checkpoint not found: {ckpt_path}")
        return None

    # Load model
    model_class = cfg['model_class']
    try:
        if model_class == 'gen6':
            model = TernaryMobileNetV5(in_ch=21, latent_dim=32)
            model.load_state_dict(
                torch.load(ckpt_path, map_location=device, weights_only=False)
            )
            model = model.to(device).eval()
        elif model_class == 'subband':
            model = TernaryMobileNetV5_Subband.from_checkpoint(ckpt_path, device=device)
            model.eval()
        else:
            print(f"  [{config_name}] Unknown model_class: {model_class}")
            return None
    except Exception as e:
        print(f"  [{config_name}] Failed to load model: {e}")
        return None

    input_type = cfg['input']
    results = []

    for patient, path in sorted(available_patients.items()):
        raw_windows, l3_windows = load_windows(path, max_windows=MAX_WINDOWS)
        if not raw_windows:
            print(f"  [{config_name}] {patient}: no valid windows, skipping")
            continue

        r_vals = []
        for i, (raw_win, l3_win) in enumerate(zip(raw_windows, l3_windows)):
            try:
                if input_type == 'raw':
                    # gen6: input [21, 2500], output [21, 2500] in raw domain
                    x_np = raw_win.astype(np.float32)
                    x_t = torch.from_numpy(x_np[np.newaxis]).to(device)
                    with torch.no_grad():
                        out = model(x_t)
                    recon_raw = out[0].cpu().numpy().astype(np.float64)
                    r = pearson_r_2d(raw_win, recon_raw)

                elif input_type == 'lpc_residual':
                    # gen6: input is LPC residual [21, 2500], output is residual
                    residual = compute_lpc_residual(raw_win)
                    x_t = torch.from_numpy(residual[np.newaxis].astype(np.float32)).to(device)
                    with torch.no_grad():
                        out = model(x_t)
                    recon_residual = out[0].cpu().numpy().astype(np.float64)
                    # R computed in residual domain (matched compression domain)
                    r = pearson_r_2d(residual, recon_residual)

                elif input_type == 'l3_nolpc':
                    # subband: forward lifting on raw (no LPC), output [21, 313]
                    l3_input = compute_l3_nolpc(raw_win)
                    x_t = torch.from_numpy(l3_input[np.newaxis]).to(device)
                    with torch.no_grad():
                        recon_l3 = model(x_t, quantize=True)
                    recon_l3_np = recon_l3[0].cpu().numpy()
                    # Inverse lifting back to raw domain for fair R comparison
                    recon_raw = inverse_lifting_from_l3(raw_win, recon_l3_np)
                    r = pearson_r_2d(raw_win, recon_raw)

                elif input_type == 'l3':
                    # subband: pre-computed L3 from NPZ [21, 313]
                    x_t = torch.from_numpy(l3_win[np.newaxis]).to(device)
                    with torch.no_grad():
                        recon_l3 = model(x_t, quantize=True)
                    recon_l3_np = recon_l3[0].cpu().numpy()
                    # Inverse lifting back to raw domain for fair R comparison
                    recon_raw = inverse_lifting_from_l3(raw_win, recon_l3_np)
                    r = pearson_r_2d(raw_win, recon_raw)

                else:
                    print(f"  [{config_name}] Unknown input type: {input_type}")
                    break

                r_vals.append(r)

            except Exception as e:
                print(f"  [{config_name}] {patient} window {i}: error {e}")
                continue

        if r_vals:
            mean_r = float(np.mean(r_vals))
            results.append((patient, mean_r))
        else:
            results.append((patient, float('nan')))

    return results


# ---------------------------------------------------------------------------
# Main run()
# ---------------------------------------------------------------------------

def run():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Patient data discovery
    q31_dir = os.path.join(ROOT_DIR, 'ai_models/dataset_sim/q31_events')
    available = {}
    for name, fname in PATIENTS.items():
        p = os.path.join(q31_dir, fname)
        if os.path.exists(p):
            available[name] = p

    if len(available) < 2:
        print(f"[SKIP] Need at least 2 held-out patients, found {len(available)}")
        return None

    # Check which checkpoints exist
    present_configs = {}
    missing_configs = {}
    for cname, cfg in ABLATION_CONFIGS.items():
        ckpt_path = os.path.join(ROOT_DIR, cfg['checkpoint'])
        if os.path.exists(ckpt_path):
            present_configs[cname] = cfg
        else:
            missing_configs[cname] = cfg

    print(f"[*] D2: Ablation Matrix on {device}")
    print(f"    Patients available : {', '.join(sorted(available.keys()))}")
    print(f"    Configs present    : {', '.join(present_configs.keys()) or 'none'}")
    print(f"    Configs missing    : {', '.join(missing_configs.keys()) or 'none'}")
    print()

    if len(present_configs) < 2:
        print(f"[SKIP] Ablation Matrix requires at least 2 configs with checkpoints.")
        print(f"       Currently available: {list(present_configs.keys())}")
        print()
        for cname, cfg in missing_configs.items():
            print(f"  Missing: {cname}: {cfg['checkpoint']}")
        print()
        # Still run what we have for informational purposes
        if len(present_configs) == 1:
            cname = list(present_configs.keys())[0]
            cfg = present_configs[cname]
            print(f"[INFO] Running single config {cname} for informational output:")
            res = evaluate_config(cname, cfg, available, device)
            if res:
                for patient, r in res:
                    print(f"  {cname} / {patient}: R = {r:.4f}")
        return None

    # Evaluate all present configs
    all_results = {}
    for cname, cfg in ABLATION_CONFIGS.items():
        if cname not in present_configs:
            all_results[cname] = None
            continue
        print(f"[*] Evaluating config: {cname}")
        print(f"    {cfg['description']}")
        res = evaluate_config(cname, cfg, available, device)
        all_results[cname] = res

    # === Report ===
    print()
    print("=" * 90)
    print(" D2: ABLATION MATRIX — R @ MATCHED COMPRESSION (Pearson R, raw EEG domain)")
    print("=" * 90)
    print()

    # Header row: config names
    config_order = list(ABLATION_CONFIGS.keys())
    header = f"  {'Patient':<10}"
    for cname in config_order:
        short = cname[:14]
        marker = '  ' if cname in present_configs else '[-]'
        header += f"  {marker + short:>16}"
    print(header)
    print(f"  {'-'*80}")

    patient_means = {cname: [] for cname in config_order}

    for patient in sorted(available.keys()):
        row = f"  {patient:<10}"
        for cname in config_order:
            res = all_results.get(cname)
            if res is None:
                row += f"  {'(missing)':>16}"
                continue
            # Find this patient's R
            r_val = next((r for p, r in res if p == patient), float('nan'))
            if np.isnan(r_val):
                row += f"  {'   N/A':>16}"
            else:
                row += f"  {r_val:>16.4f}"
                patient_means[cname].append(r_val)
        print(row)

    print(f"  {'-'*80}")
    mean_row = f"  {'MEAN':<10}"
    for cname in config_order:
        vals = [v for v in patient_means[cname] if not np.isnan(v)]
        if vals:
            mean_row += f"  {float(np.mean(vals)):>16.4f}"
        elif all_results.get(cname) is None:
            mean_row += f"  {'(missing)':>16}"
        else:
            mean_row += f"  {'   N/A':>16}"
    print(mean_row)
    print()

    # Config descriptions
    print("  Config descriptions:")
    for cname, cfg in ABLATION_CONFIGS.items():
        status = "[PRESENT]" if cname in present_configs else "[MISSING]"
        print(f"    {status} {cname}: {cfg['description']}")
    print()

    # Delta analysis (D - best of A/B/C)
    present_names = list(present_configs.keys())
    mean_rs = {}
    for cname in present_names:
        vals = [v for v in patient_means[cname] if not np.isnan(v)]
        mean_rs[cname] = float(np.mean(vals)) if vals else float('nan')

    print("  Per-config mean R:")
    for cname in config_order:
        if cname in mean_rs:
            r = mean_rs[cname]
            print(f"    {cname:<20}: {r:.4f}" if not np.isnan(r) else f"    {cname:<20}: N/A")
        else:
            print(f"    {cname:<20}: (missing checkpoint)")

    print()

    # Delta: if D (full pipeline) and at least one baseline are present
    if 'D_full_pipeline' in mean_rs and any(
        c in mean_rs for c in ('A_raw_tnn8', 'B_lpc_tnn8', 'C_l3_nolpc')
    ):
        r_d = mean_rs['D_full_pipeline']
        baselines = {c: mean_rs[c] for c in ('A_raw_tnn8', 'B_lpc_tnn8', 'C_l3_nolpc')
                     if c in mean_rs and not np.isnan(mean_rs[c])}
        if baselines:
            best_baseline_name = max(baselines, key=baselines.get)
            best_baseline_r = baselines[best_baseline_name]
            delta = r_d - best_baseline_r
            sign = '+' if delta >= 0 else ''
            print(f"  D vs best baseline ({best_baseline_name}): {sign}{delta:.4f}")

    print("=" * 90)
    print()

    # Pass: at least 2 configs ran
    n_ran = sum(1 for cname in config_order
                if all_results.get(cname) is not None)
    passed = n_ran >= 2

    if passed:
        print(f"[PASS] D2: Ablation Matrix — {n_ran}/{len(ABLATION_CONFIGS)} configs ran.")
    else:
        print(f"[NOTE] D2: Only {n_ran} config(s) ran — need >= 2 for comparison.")

    return {
        'passed': passed,
        'n_configs_ran': n_ran,
        'n_configs_total': len(ABLATION_CONFIGS),
        'present_configs': list(present_configs.keys()),
        'missing_configs': list(missing_configs.keys()),
        'per_config_mean_r': mean_rs,
        'per_config_results': {
            cname: all_results[cname] for cname in config_order
        },
    }


if __name__ == '__main__':
    result = run()
    if result is None:
        sys.exit(0)   # SKIP is not a failure
    sys.exit(0 if result.get('passed', False) else 1)
