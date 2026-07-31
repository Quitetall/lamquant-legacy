#!/usr/bin/env python3
"""
LamQuant Gen 7.5 — D10: XNOR+cpop Kernel Benchmark (Hazard3 stub)
==================================================================
Diagnostic D10: How close is the MAC kernel to theoretical throughput?

This is a theoretical-only analysis.  No hardware is required.

The TernaryMobileNetV5_Subband encoder runs on an RP2350 Hazard3 core.
Ternary weights are stored as 2-bit packed values.  Each MAC reduces to:
  XNOR  (1 instruction) + cpop/popcount (1 instruction) → 2 instructions
At 5 cycles overhead per XNOR+cpop pair, and 16 MACs per 5 cycles (ideal
pipelined throughput), we estimate theoretical wall time at 150 MHz.

Algorithm:
  1. Load model to extract architecture params (width, kernel sizes, shapes)
  2. Compute MACs per encoder layer
  3. Sum → total MACs for one 313-sample L3 window
  4. Theoretical cycles at 16 MACs / 5 cycles (XNOR+cpop on RV32)
  5. Wall time at 150 MHz
  6. Print hardware profiling instructions for real RP2350

PASS: Always passes (informational benchmark — no hardware needed to run).

Usage:
  python benchmark_xnor_cpop_stub.py
  python benchmark_xnor_cpop_stub.py --checkpoint path/to/model.ckpt
"""

import torch
import os
import sys
import numpy as np
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

# RP2350 Hazard3 throughput constants
MACS_PER_CYCLE_IDEAL = 16 / 5    # 16 MACs in 5 cycles (XNOR+cpop pipeline)
CPU_FREQ_HZ = 150e6               # 150 MHz Hazard3


def conv_macs(in_ch, out_ch, kernel_size, output_length, groups=1):
    """MACs for a 1D convolution layer.

    Formula: out_ch * (in_ch / groups) * kernel_size * output_length
    Each multiply-accumulate is one MAC.
    """
    return int(out_ch * (in_ch / groups) * kernel_size * output_length)


def shortcut_macs(in_ch, out_ch, output_length, strided=False):
    """MACs for a 1x1 shortcut conv (or stride-1 identity = 0 MACs)."""
    if in_ch == out_ch and not strided:
        return 0  # Identity shortcut
    return int(out_ch * in_ch * 1 * output_length)


def run(checkpoint_path=None):
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
        print("[SKIP] No student checkpoint found — using default Gen 7.5 dimensions.")
        # Fall back to canonical Gen 7.5 architecture dimensions
        W = 96
        checkpoint_path = None
        width_source = "default (Gen 7.5 canonical)"
    else:
        sd = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        W = sd['focal2.conv.weight'].shape[0] if 'focal2.conv.weight' in sd else 96
        width_source = f"checkpoint ({checkpoint_path})"

    print(f"[*] D10: XNOR+cpop Kernel Benchmark (Hazard3 stub)")
    print(f"    Width source : {width_source}")
    print(f"    Width W      : {W}")
    print(f"    Input        : L3 approximation [21, 313]")
    print(f"    XNOR+cpop    : 16 MACs / 5 cycles (ideal pipeline)")
    print(f"    CPU freq     : {CPU_FREQ_HZ/1e6:.0f} MHz")
    print()

    # Gen 7.5 encoder layer dimensions for input [21, 313]
    #
    #   premix:     21→21,   k=1, s=1    313 → 313
    #   focal1_conv:21→W,    k=3, s=2    313 → 157
    #   focal1_sc:  21→W,    k=1, s=2    (zero-pad shortcut → 0 MACs)
    #   focal2.conv:W→W,     k=5, s=1    157 → 157
    #   focal2.sc:  W→W,     k=1, s=1    (identity, 0 MACs)
    #   focal3.conv:W→W,     k=7, s=2    157 → 79
    #   focal3.sc:  W→W,     k=1, s=2    (stride shortcut → has MACs)
    #   dw_gate:    W→W,     k=3, dw     79 → 79  (depthwise, groups=W)
    #   bneck_v:    W→32,    k=1         79 → 79
    #   bneck_g:    W→32,    k=1         79 → 79

    in_ch     = 21
    latent    = 32

    # Output lengths after each stride
    T_in      = 313
    T_focal1  = (T_in + 1) // 2          # stride 2 → 157
    T_focal2  = T_focal1                  # stride 1 → 157
    T_focal3  = (T_focal2 + 1) // 2      # stride 2 → 79
    T_latent  = T_focal3                  # 79

    layers = [
        # (name,          MACs)
        ('premix',        conv_macs(in_ch, in_ch, 1, T_in)),
        ('focal1_conv',   conv_macs(in_ch, W,     3, T_focal1)),
        ('focal1_sc',     0),   # zero-pad shortcut: no multiply
        ('focal2.conv',   conv_macs(W, W, 5, T_focal2)),
        ('focal2.sc',     shortcut_macs(W, W, T_focal2, strided=False)),
        ('focal3.conv',   conv_macs(W, W, 7, T_focal3)),
        ('focal3.sc',     shortcut_macs(W, W, T_focal3, strided=True)),
        ('dw_gate',       conv_macs(W, W, 3, T_latent, groups=W)),
        ('bneck_v',       conv_macs(W, latent, 1, T_latent)),
        ('bneck_g',       conv_macs(W, latent, 1, T_latent)),
    ]

    total_macs = sum(m for _, m in layers)

    # Theoretical cycles (ternary: XNOR+cpop at 16 MACs / 5 cycles)
    cycles_ideal     = total_macs / MACS_PER_CYCLE_IDEAL
    # Realistic: add 30% overhead for memory latency, branch, norm, etc.
    cycles_realistic = cycles_ideal * 1.30

    wall_ideal_ms    = (cycles_ideal     / CPU_FREQ_HZ) * 1e3
    wall_realistic_ms = (cycles_realistic / CPU_FREQ_HZ) * 1e3

    # Print layer breakdown
    print("=" * 78)
    print(" D10: ENCODER MAC COUNT PER LAYER")
    print(f" Input: L3 approximation [21ch, {T_in} samples], Width W={W}")
    print("=" * 78)
    print(f"  {'Layer':<18} {'Output T':>8} {'MACs':>12} {'% Total':>9}")
    print(f"  {'-'*52}")

    output_lengths = {
        'premix':       T_in,
        'focal1_conv':  T_focal1,
        'focal1_sc':    T_focal1,
        'focal2.conv':  T_focal2,
        'focal2.sc':    T_focal2,
        'focal3.conv':  T_focal3,
        'focal3.sc':    T_focal3,
        'dw_gate':      T_latent,
        'bneck_v':      T_latent,
        'bneck_g':      T_latent,
    }

    for name, macs in layers:
        out_t = output_lengths[name]
        pct = (macs / total_macs * 100) if total_macs > 0 else 0.0
        print(f"  {name:<18} {out_t:>8} {macs:>12,} {pct:>8.1f}%")

    print(f"  {'-'*52}")
    print(f"  {'TOTAL':<18} {'':>8} {total_macs:>12,} {'100.0%':>9}")
    print()

    print("=" * 78)
    print(" THEORETICAL THROUGHPUT")
    print("=" * 78)
    print(f"  Total MACs (encoder, 1 window)  : {total_macs:>12,}")
    print(f"  XNOR+cpop rate                  : {MACS_PER_CYCLE_IDEAL:.1f} MACs/cycle (ideal)")
    print(f"  Cycles (ideal)                  : {cycles_ideal:>12,.0f}")
    print(f"  Cycles (realistic, +30% overhead): {cycles_realistic:>12,.0f}")
    print(f"  Wall time ideal    @ {CPU_FREQ_HZ/1e6:.0f} MHz : {wall_ideal_ms:>8.3f} ms")
    print(f"  Wall time realistic@ {CPU_FREQ_HZ/1e6:.0f} MHz : {wall_realistic_ms:>8.3f} ms")
    print()
    real_time_margin = 10_000.0 / wall_realistic_ms  # 10-second window
    print(f"  Real-time margin (10s window)   : {real_time_margin:>8.1f}x")
    print()

    print("=" * 78)
    print(" HARDWARE PROFILING INSTRUCTIONS (RP2350 / Hazard3)")
    print("=" * 78)
    print()
    print("  To measure actual throughput on real hardware:")
    print()
    print("  1. Build the encoder with -O2 -march=rv32imac_zba_zbb_zbs:")
    print("     $ cmake -DCMAKE_BUILD_TYPE=Release -DPICO_BOARD=pico2 .")
    print("     $ make focal_net_encoder")
    print()
    print("  2. Wrap the encode() call with PICO SDK timer:")
    print("     uint32_t t0 = time_us_32();")
    print("     focal_net_encode(l3_window, latent_out);")
    print("     uint32_t t1 = time_us_32();")
    print("     printf(\"encoder_us=%lu\\n\", t1 - t0);")
    print()
    print("  3. Enable hardware performance counters (Hazard3 PMU):")
    print("     CSR mcountinhibit = 0 to enable mcycle, minstret counters.")
    print("     Read minstret before/after encode() for actual instruction count.")
    print("     IPC = minstret / mcycle — compare to theoretical 16/5 = 3.2 IPC.")
    print()
    print("  4. Check XNOR+cpop pipeline efficiency:")
    print("     Ideal: 1 xor + 1 cpop per 2 ternary weights → 2 instructions.")
    print("     Hazard3 cpop latency: 1 cycle (single-cycle popcount unit).")
    print("     Expected bottleneck: L2 SRAM4 bandwidth (weight streaming).")
    print()
    print("  5. Enable Pico2 overclocking to 200+ MHz (verify VCC stability):")
    print("     set_sys_clock_khz(200000, true);")
    print(f"     Expected wall time at 200 MHz: {cycles_realistic/200e6*1e3:.3f} ms/window")
    print()

    # Always PASS (informational)
    print("[PASS] D10: XNOR+cpop Kernel Benchmark (informational — no hardware needed)")
    print()

    return {
        'passed': True,
        'total_macs': total_macs,
        'cycles_ideal': cycles_ideal,
        'cycles_realistic': cycles_realistic,
        'wall_ideal_ms': wall_ideal_ms,
        'wall_realistic_ms': wall_realistic_ms,
        'real_time_margin': real_time_margin,
        'width': W,
        'layers': layers,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()
    result = run(args.checkpoint)
    sys.exit(0)
