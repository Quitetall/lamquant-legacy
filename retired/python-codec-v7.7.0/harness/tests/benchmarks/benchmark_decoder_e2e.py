#!/usr/bin/env python3
"""
LamQuant Gen 7 — Heavy Decoder End-to-End Benchmark
====================================================
Benchmarks decoder performance under realistic load:
  - Route A (student decoder): latency, throughput, memory
  - Route B (teacher decoder): same metrics + latent upsampling cost

Runs 100 windows (10 seconds at 250Hz × 21ch), measures:
  - Per-window decode latency (ms)
  - Peak memory (MB)
  - Mean Pearson R across all windows
  - p95 latency
"""
import os
import sys
import time
import tracemalloc
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'ai_models', 'student'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'ai_models', 'oracle'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'ai_models'))

from lamquant_codec.models.encoder import TernaryMobileNetV5_Subband


def pearson_r_batch(x, y):
    """Per-channel Pearson R averaged over channels."""
    x_np = x[0].numpy()
    y_np = y[0].numpy()
    rs = []
    for ch in range(x_np.shape[0]):
        a = x_np[ch] - x_np[ch].mean()
        b = y_np[ch] - y_np[ch].mean()
        d = np.sqrt(np.sum(a**2) * np.sum(b**2))
        rs.append(float(np.sum(a * b) / d) if d > 1e-12 else 0.0)
    return np.mean(rs)


def benchmark_route_a(model, num_windows=100):
    """Benchmark student encoder+decoder (Route A)."""
    print("\n--- Route A: Student Decoder ---")
    model.eval()

    latencies = []
    rs = []

    tracemalloc.start()
    with torch.no_grad():
        for i in range(num_windows):
            # Subband model expects [B, 21, 313] (L3 subband input)
            x = torch.clamp(torch.randn(1, 21, 313) * 20, -50, 50)

            t0 = time.perf_counter()
            latent = model.encode(x, quantize=True)
            recon = model.decode(latent, target_len=x.shape[2])
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000)  # ms
            min_T = min(x.shape[2], recon.shape[2])
            rs.append(pearson_r_batch(x[:, :, :min_T], recon[:, :, :min_T]))

    _, peak_mb = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb /= 1024 * 1024

    latencies = np.array(latencies)
    print(f"  Windows:     {num_windows}")
    print(f"  Mean latency: {latencies.mean():.2f} ms")
    print(f"  p50 latency:  {np.percentile(latencies, 50):.2f} ms")
    print(f"  p95 latency:  {np.percentile(latencies, 95):.2f} ms")
    print(f"  Peak memory:  {peak_mb:.1f} MB")
    print(f"  Mean R:        {np.mean(rs):.4f}")

    return {
        'mean_latency_ms': float(latencies.mean()),
        'p95_latency_ms': float(np.percentile(latencies, 95)),
        'peak_memory_mb': peak_mb,
        'mean_r': float(np.mean(rs)),
    }


def benchmark_route_b(student_model, num_windows=100):
    """Benchmark teacher decoder (Route B) with latent upsampling."""
    print("\n--- Route B: Teacher Decoder ---")

    teacher_ckpt = os.path.join(ROOT_DIR, 'ai_models', 'oracle', 'teacher_best.ckpt')
    if not os.path.exists(teacher_ckpt):
        print("  [SKIP] Teacher checkpoint not found")
        return None

    try:
        from train_teacher import FP32OracleAutoEncoder
        teacher = FP32OracleAutoEncoder(in_ch=21, enc_dim=32)
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location='cpu'))
        teacher.eval()
    except Exception as e:
        print(f"  [SKIP] Could not load teacher: {e}")
        return None

    latencies = []
    rs = []

    tracemalloc.start()
    with torch.no_grad():
        for i in range(num_windows):
            # Subband model expects [B, 21, 313] (L3 subband input)
            x = torch.clamp(torch.randn(1, 21, 313) * 20, -50, 50)

            t0 = time.perf_counter()
            latent = student_model.encode(x, quantize=True)  # [1, 32, T/4]
            # Upsample to full length for teacher decoder
            latent_up = torch.nn.functional.interpolate(
                latent, size=x.shape[2], mode='linear', align_corners=False)
            recon = teacher.decode(latent_up)
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000)
            min_T = min(x.shape[2], recon.shape[2])
            rs.append(pearson_r_batch(x[:, :, :min_T], recon[:, :, :min_T]))

    _, peak_mb = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb /= 1024 * 1024

    latencies = np.array(latencies)
    print(f"  Windows:     {num_windows}")
    print(f"  Mean latency: {latencies.mean():.2f} ms")
    print(f"  p50 latency:  {np.percentile(latencies, 50):.2f} ms")
    print(f"  p95 latency:  {np.percentile(latencies, 95):.2f} ms")
    print(f"  Peak memory:  {peak_mb:.1f} MB")
    print(f"  Mean R:        {np.mean(rs):.4f}")

    return {
        'mean_latency_ms': float(latencies.mean()),
        'p95_latency_ms': float(np.percentile(latencies, 95)),
        'peak_memory_mb': peak_mb,
        'mean_r': float(np.mean(rs)),
    }


def main():
    print("=" * 60)
    print("LamQuant Gen 7 — Decoder E2E Benchmark")
    print("=" * 60)

    ckpt_path = os.path.join(ROOT_DIR, 'ai_models', 'student', 'student_hardened.ckpt')
    if not os.path.exists(ckpt_path):
        print(f"[SKIP] Student checkpoint not found: {ckpt_path}")
        return 0

    model = TernaryMobileNetV5_Subband.from_checkpoint(ckpt_path, device='cpu')

    route_a = benchmark_route_a(model)
    route_b = benchmark_route_b(model)

    print("\n--- Summary ---")
    print(f"  Route A: {route_a['mean_latency_ms']:.1f} ms/window, "
          f"R={route_a['mean_r']:.4f}")
    if route_b:
        print(f"  Route B: {route_b['mean_latency_ms']:.1f} ms/window, "
              f"R={route_b['mean_r']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
