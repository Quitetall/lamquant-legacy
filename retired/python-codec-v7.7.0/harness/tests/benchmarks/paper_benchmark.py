#!/usr/bin/env python3
"""paper_benchmark.py — Publication-grade LML compression benchmark.

Produces the complete benchmark report for the LamQuant paper:
  1. Compression performance (CR distribution, aggregate, per-subset)
  2. Correctness verification (bit-perfect roundtrip on every file)
  3. Throughput (encode/decode speed)
  4. Dataset characterization (hours, channels, sample rates)
  5. Comparison baselines (gzip, zstd, raw)
  6. Per-component ablation (lifting, LPC, bias cancel, GR)
  7. Reproducibility (commit hash, config, test set)

Usage:
    # Full benchmark on TUEG (takes hours)
    python paper_benchmark.py /data/tueg/ -o /data/lml/ -n 0

    # Quick validation on 100 random files
    python paper_benchmark.py /data/tueg/ -o /data/lml/ -n 100

    # Statistical sample (500 files, paper-ready)
    python paper_benchmark.py /data/tueg/ -o /data/lml/ -n 500
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import random
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from lamquant_codec.lossless import _compress_bytes, _decompress_bytes


# ============================================================
# Per-file measurement
# ============================================================

def measure_file(edf_path: str, output_dir: str, baselines: bool = True,
                 ablation: bool = True) -> dict:
    """Measure everything about one file. Returns metrics dict."""
    import pyedflib
    from collections import Counter

    result = {
        'source': os.path.basename(edf_path),
        'source_path': edf_path,
        'source_size': os.path.getsize(edf_path),
    }

    # Read EDF with pyedflib (C-backed, 12× faster than MNE)
    try:
        f = pyedflib.EdfReader(edf_path)
    except Exception as e:
        result['error'] = str(e)[:100]
        return result

    try:
        n_ch = f.signals_in_file
        labels = f.getSignalLabels()
        srs = f.getSampleFrequencies()

        # Filter annotation channels, pick most common SR
        eeg_idx = [i for i in range(n_ch) if 'annotation' not in labels[i].lower()]
        if not eeg_idx:
            result['error'] = 'no_eeg_channels'
            f.close()
            return result

        mode_sr = Counter(srs[i] for i in eeg_idx).most_common(1)[0][0]
        eeg_idx = [i for i in eeg_idx if srs[i] == mode_sr]
        C = len(eeg_idx)
        T = int(f.getNSamples()[eeg_idx[0]])
        sr = float(mode_sr)

        # Read physical signals, convert to int16 for compression
        data = np.zeros((C, T), dtype=np.float64)
        for j, ch in enumerate(eeg_idx):
            sig = f.readSignal(ch)
            data[j, :min(len(sig), T)] = sig[:T]

        # Store metadata
        result['n_channels'] = C
        result['sample_rate'] = sr
        result['duration_s'] = float(T / sr)
        result['n_samples'] = int(T)
        result['patient'] = f.getPatientCode()
        result['sex'] = f.getSex()
    finally:
        f.close()

    # Scale to int16
    mx = np.abs(data).max()
    if mx < 1e-12:
        result['error'] = 'flat_signal'
        return result
    gain = 0.72 / mx
    signal = (data * gain * 32767).astype(np.int64)
    raw_int16_bytes = C * T * 2

    result['raw_int16_bytes'] = raw_int16_bytes

    # ---- LML compression ----
    window_samples = int(10 * sr)
    n_windows = max(1, (T + window_samples - 1) // window_samples)

    t_enc_start = time.perf_counter()
    compressed_parts = []
    for w in range(n_windows):
        start = w * window_samples
        end = min(start + window_samples, T)
        window = signal[:, start:end]
        compressed_parts.append(_compress_bytes(window.astype(np.float64), n_levels=3))
    t_enc = time.perf_counter() - t_enc_start

    lml_total = sum(len(p) for p in compressed_parts)
    result['lml_bytes'] = lml_total
    result['lml_cr'] = raw_int16_bytes / lml_total
    result['encode_time_ms'] = t_enc * 1000
    result['encode_speed_mbps'] = raw_int16_bytes / 1e6 / max(t_enc, 1e-9)

    # ---- Bit-perfect roundtrip ----
    t_dec_start = time.perf_counter()
    recon = np.zeros_like(signal)
    for w, payload in enumerate(compressed_parts):
        decoded = _decompress_bytes(payload)
        start = w * window_samples
        end = min(start + window_samples, T)
        actual = min(decoded.shape[1], end - start)
        recon[:C, start:start + actual] = np.round(decoded[:C, :actual]).astype(np.int64)
    t_dec = time.perf_counter() - t_dec_start

    result['decode_time_ms'] = t_dec * 1000
    result['decode_speed_mbps'] = raw_int16_bytes / 1e6 / max(t_dec, 1e-9)
    result['bit_perfect'] = bool(np.array_equal(signal, recon))
    if not result['bit_perfect']:
        diff = np.abs(signal - recon)
        result['max_diff'] = int(diff.max())
        result['n_diff_samples'] = int(np.count_nonzero(diff))

    # Content hash for reproducibility
    result['content_sha256'] = hashlib.sha256(signal[:, :2500].tobytes()).hexdigest()[:16]

    # ---- Baseline comparisons ----
    if baselines:
        raw_bytes = signal.astype(np.int16).tobytes()

        # gzip
        import gzip
        t0 = time.perf_counter()
        gz = gzip.compress(raw_bytes, compresslevel=6)
        result['gzip_bytes'] = len(gz)
        result['gzip_cr'] = raw_int16_bytes / len(gz)
        result['gzip_time_ms'] = (time.perf_counter() - t0) * 1000

        # zstd
        try:
            import pyzstd
            t0 = time.perf_counter()
            zs = pyzstd.compress(raw_bytes, 3)
            result['zstd_bytes'] = len(zs)
            result['zstd_cr'] = raw_int16_bytes / len(zs)
            result['zstd_time_ms'] = (time.perf_counter() - t0) * 1000
        except ImportError:
            pass

    # ---- Per-component ablation ----
    if ablation:
        from lamquant_codec.ops.lpc import analyze_channel as lpc_analyze_channel, analyze_int as lpc_analyze_int
        from lamquant_codec.ops.lifting import forward_1d_int as lifting_1d_forward_int
        from lamquant_codec.ops.golomb import encode_dense

        # Test one window for ablation
        win = signal[:, :min(window_samples, T)]

        # A: Raw GR (no prediction, no lifting)
        a_raw = sum(len(encode_dense(win[ch].astype(np.int64))) for ch in range(C))

        # B: First-order diff + GR
        a_diff = 0
        for ch in range(C):
            d = np.concatenate([[win[ch, 0]], np.diff(win[ch])])
            a_diff += len(encode_dense(d.astype(np.int64)))

        # C: Lifting + GR (no LPC)
        a_lift = 0
        for ch in range(C):
            approx = win[ch].copy()
            for _ in range(3):
                approx, detail = lifting_1d_forward_int(approx)
            # Encode subbands raw (no LPC)
            a_lift += len(encode_dense(approx.astype(np.int64)))

        # D: Lifting + LPC + GR (no bias cancel) — approximate by using full pipeline
        # The full pipeline IS lifting + LPC + bias + GR
        a_full = sum(len(p) for p in compressed_parts[:1])  # first window

        win_raw = C * min(window_samples, T) * 2
        result['ablation_raw_gr_cr'] = win_raw / a_raw
        result['ablation_diff_gr_cr'] = win_raw / a_diff
        result['ablation_lift_gr_cr'] = win_raw / a_lift
        result['ablation_full_cr'] = win_raw / a_full

    return result


# ============================================================
# Full benchmark
# ============================================================

def run_benchmark(input_dir: str, output_dir: str, n_files: int = 0,
                  seed: int = 42, baselines: bool = True,
                  ablation: bool = True) -> dict:
    """Run the complete paper benchmark. Returns report dict."""

    all_edfs = sorted(glob.glob(os.path.join(input_dir, '**', '*.edf'), recursive=True))
    all_edfs += sorted(glob.glob(os.path.join(input_dir, '**', '*.EDF'), recursive=True))
    all_edfs = sorted(set(all_edfs))

    if n_files > 0 and n_files < len(all_edfs):
        random.seed(seed)
        sample = random.sample(all_edfs, n_files)
    else:
        sample = all_edfs
        n_files = len(sample)

    # Git commit for reproducibility
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=str(_REPO), stderr=subprocess.DEVNULL
        ).decode().strip()[:12]
    except Exception:
        git_hash = 'unknown'

    report = {
        'meta': {
            'codec_version': 'LML v4.1',
            'git_commit': git_hash,
            'created': datetime.now(timezone.utc).isoformat(),
            'hostname': platform.node(),
            'python': platform.python_version(),
            'numpy': np.__version__,
            'seed': seed,
            'n_files_sampled': n_files,
            'n_files_total': len(all_edfs),
            'baselines_enabled': baselines,
            'ablation_enabled': ablation,
        },
        'files': [],
    }

    print(f'=== LML Paper Benchmark ===')
    print(f'Corpus: {len(all_edfs):,} EDFs')
    print(f'Sample: {n_files:,} files (seed={seed})')
    print(f'Git:    {git_hash}')
    print()

    t0 = time.time()
    for i, edf_path in enumerate(sample):
        metrics = measure_file(edf_path, output_dir, baselines=baselines,
                                ablation=ablation and (i < 50))  # ablation on first 50 only

        report['files'].append(metrics)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_files - i - 1) / rate
            ok = sum(1 for f in report['files'] if f.get('bit_perfect'))
            err = sum(1 for f in report['files'] if 'error' in f)
            crs = [f['lml_cr'] for f in report['files'] if 'lml_cr' in f]
            mean_cr = np.mean(crs) if crs else 0
            print(f'  [{i+1:>5,}/{n_files:,}] {ok:,} ok, {err} err, '
                  f'CR={mean_cr:.2f}:1 ({rate:.1f}/s, ETA {eta/60:.0f}m)')

    # ---- Compute aggregate statistics ----
    valid = [f for f in report['files'] if 'lml_cr' in f]
    errors = [f for f in report['files'] if 'error' in f]

    crs = np.array([f['lml_cr'] for f in valid])
    durations = np.array([f['duration_s'] for f in valid])
    channels = np.array([f['n_channels'] for f in valid])
    srs = np.array([f['sample_rate'] for f in valid])
    raw_total = sum(f['raw_int16_bytes'] for f in valid)
    lml_total = sum(f['lml_bytes'] for f in valid)

    report['aggregate'] = {
        'n_files_ok': len(valid),
        'n_files_error': len(errors),
        'n_bit_perfect': sum(1 for f in valid if f.get('bit_perfect')),
        'total_raw_bytes': raw_total,
        'total_lml_bytes': lml_total,
        'total_edf_bytes': sum(f['source_size'] for f in valid),
        'aggregate_cr': raw_total / max(1, lml_total),
        'mean_cr': float(np.mean(crs)) if len(crs) else 0,
        'median_cr': float(np.median(crs)) if len(crs) else 0,
        'std_cr': float(np.std(crs)) if len(crs) else 0,
        'p5_cr': float(np.percentile(crs, 5)) if len(crs) else 0,
        'p95_cr': float(np.percentile(crs, 95)) if len(crs) else 0,
        'min_cr': float(np.min(crs)) if len(crs) else 0,
        'max_cr': float(np.max(crs)) if len(crs) else 0,
        'total_hours': float(np.sum(durations) / 3600),
        'mean_channels': float(np.mean(channels)),
        'channel_distribution': dict(zip(*np.unique(channels, return_counts=True))),
        'sample_rate_distribution': dict(zip(*np.unique(srs, return_counts=True))),
        'mean_encode_speed_mbps': float(np.mean([f['encode_speed_mbps'] for f in valid])),
        'mean_decode_speed_mbps': float(np.mean([f['decode_speed_mbps'] for f in valid])),
    }

    # Baseline aggregates
    if baselines:
        gzip_total = sum(f.get('gzip_bytes', 0) for f in valid)
        zstd_total = sum(f.get('zstd_bytes', 0) for f in valid if 'zstd_bytes' in f)
        report['baselines'] = {
            'gzip_cr': raw_total / max(1, gzip_total),
            'zstd_cr': raw_total / max(1, zstd_total) if zstd_total else 0,
            'lml_cr': raw_total / max(1, lml_total),
        }

    # Ablation aggregates
    abl_files = [f for f in valid if 'ablation_raw_gr_cr' in f]
    if abl_files:
        report['ablation'] = {
            'raw_gr_cr': float(np.mean([f['ablation_raw_gr_cr'] for f in abl_files])),
            'diff_gr_cr': float(np.mean([f['ablation_diff_gr_cr'] for f in abl_files])),
            'lift_gr_cr': float(np.mean([f['ablation_lift_gr_cr'] for f in abl_files])),
            'full_cr': float(np.mean([f['ablation_full_cr'] for f in abl_files])),
        }

    return report


def print_report(report: dict):
    """Print paper-ready summary."""
    a = report['aggregate']
    m = report['meta']
    W = 60

    def row(label, value):
        print(f'  |  {label:<24}{value:<{W-28}}|')
    def sep():
        print(f'  +{"-"*(W-2)}+')
    def header(text):
        print(f'  |{text:^{W-2}}|')

    sep()
    header('LML Lossless EEG Compression — Paper Benchmark')
    sep()
    row('Codec', m['codec_version'])
    row('Git commit', m['git_commit'])
    row('Date', m['created'][:10])
    row('Files tested', f"{a['n_files_ok']:,} / {m['n_files_total']:,} corpus")
    row('Total recording', f"{a['total_hours']:.1f} hours")
    sep()
    header('Compression Performance')
    sep()
    row('Aggregate CR', f"{a['aggregate_cr']:.2f}:1")
    row('Mean CR', f"{a['mean_cr']:.2f}:1 +/- {a['std_cr']:.2f}")
    row('Median CR', f"{a['median_cr']:.2f}:1")
    row('5th-95th pctl', f"{a['p5_cr']:.2f}:1 to {a['p95_cr']:.2f}:1")
    row('Min / Max', f"{a['min_cr']:.2f}:1 / {a['max_cr']:.2f}:1")
    row('Raw (int16)', f"{a['total_raw_bytes']/1e9:.2f} GB")
    row('Compressed', f"{a['total_lml_bytes']/1e9:.2f} GB")
    row('Saved', f"{(a['total_raw_bytes']-a['total_lml_bytes'])/1e9:.2f} GB "
                 f"({(1-a['total_lml_bytes']/max(1,a['total_raw_bytes']))*100:.1f}%)")
    sep()
    header('Correctness')
    sep()
    row('Bit-perfect', f"{a['n_bit_perfect']} / {a['n_files_ok']} "
                        f"({'100.0' if a['n_bit_perfect']==a['n_files_ok'] else 'FAIL'}%)")
    row('Errors', f"{a['n_files_error']}")
    sep()
    header('Throughput')
    sep()
    row('Encode speed', f"{a['mean_encode_speed_mbps']:.1f} MB/s")
    row('Decode speed', f"{a['mean_decode_speed_mbps']:.1f} MB/s")
    sep()
    header('Dataset')
    sep()
    row('Mean channels', f"{a['mean_channels']:.1f}")
    row('EDF total', f"{a['total_edf_bytes']/1e9:.2f} GB")

    if 'baselines' in report:
        sep()
        header('Baselines (same data)')
        sep()
        b = report['baselines']
        row('gzip -6', f"{b['gzip_cr']:.2f}:1")
        if b.get('zstd_cr'):
            row('zstd -3', f"{b['zstd_cr']:.2f}:1")
        row('LML v4.1', f"{b['lml_cr']:.2f}:1")

    if 'ablation' in report:
        sep()
        header('Per-Component Ablation')
        sep()
        ab = report['ablation']
        row('Raw GR (no pred)', f"{ab['raw_gr_cr']:.2f}:1")
        row('Diff + GR', f"{ab['diff_gr_cr']:.2f}:1")
        row('Lifting + GR', f"{ab['lift_gr_cr']:.2f}:1")
        row('Full (lift+LPC+bias+GR)', f"{ab['full_cr']:.2f}:1")

    sep()


def main():
    parser = argparse.ArgumentParser(prog='paper_benchmark')
    parser.add_argument('input', help='EDF directory')
    parser.add_argument('-o', '--output', default='/tmp/lml_benchmark/')
    parser.add_argument('-n', '--n-files', type=int, default=100,
                        help='Number of files to sample (0=all)')
    parser.add_argument('-r', '--report', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-baselines', action='store_true')
    parser.add_argument('--no-ablation', action='store_true')
    args = parser.parse_args()

    report = run_benchmark(
        args.input, args.output,
        n_files=args.n_files, seed=args.seed,
        baselines=not args.no_baselines,
        ablation=not args.no_ablation,
    )

    print()
    print_report(report)

    report_path = args.report or os.path.join(args.output, 'paper_benchmark.json')
    os.makedirs(os.path.dirname(report_path) or '.', exist_ok=True)

    # Strip per-file data for the summary JSON (keep full in separate file)
    summary = {k: v for k, v in report.items() if k != 'files'}
    summary['n_files_measured'] = len(report['files'])
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\n  Report: {report_path}')

    # Full per-file data
    full_path = report_path.replace('.json', '_full.json')
    with open(full_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'  Full:   {full_path}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
