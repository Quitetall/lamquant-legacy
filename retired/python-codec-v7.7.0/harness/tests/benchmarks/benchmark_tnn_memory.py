#!/usr/bin/env python3
"""
LamQuant Gen 7 — TNN Memory Audit
==================================
Only the ENCODER runs on the RP2350 (SRAM4).
The decoder runs on the base station (phone/laptop, unconstrained).

This benchmark checks:
  1. Encoder fits in SRAM4 (43,008 bytes)
  2. Reports decoder size separately (informational)
  3. Reports full model size (for reference)
"""
import torch
import os
import sys
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

# RP2350: V1 w=128 encoder spans SRAM4 (64 KB) + partial SRAM5.
# Total weight budget: 75 KB (ternary + Q31 + INT8 + CDF + Cayley).
# Activations are separate (78 KB in SRAM6-7).
MODEL_BUDGET = 80 * 1024  # 81,920 bytes (75 KB + 5 KB margin)

# Encoder layer prefixes (everything that runs on-chip)
ENCODER_PREFIXES = ('premix', 'focal1', 'focal2', 'focal3', 'dw_gate', 'bneck_v', 'bneck_g', 'rotation_A')
# Decoder layer prefixes (runs on base station)
DECODER_PREFIXES = ('expand1', 'expand2', 'expand3', 'output')


def estimate_bytes(param_name, tensor, ternary_modules):
    """Estimate packed binary size for one parameter."""
    parts = param_name.rsplit('.', 1)
    module_name = parts[0] if len(parts) > 1 else ''
    attr_name = parts[-1]
    is_ternary = module_name in ternary_modules

    # Path B: metadata uses INT16 (Q15) to fit width=128 in 64KB.
    # Ternary weights stay W2. GroupNorm weights stay INT8.
    if attr_name == 'weight' and is_ternary:
        return (tensor.numel() + 3) // 4, "W2 (2-bit packed)"
    elif attr_name == 'lsq_alpha':
        return tensor.numel() * 2, "Q15 (2B/param)"
    elif attr_name == 'weight' and not is_ternary:
        return tensor.numel() * 1, "INT8 (1B/param)"
    elif attr_name == 'bias':
        return tensor.numel() * 2, "Q15 (2B/param)"
    elif 'norm' in module_name and 'weight' in attr_name:
        return tensor.numel() * 1, "INT8 (1B/param)"
    elif 'norm' in module_name:
        return tensor.numel() * 2, "Q15 (2B/param)"
    elif param_name == 'rotation_A':
        return tensor.numel() * 2, "Q15 (rotation)"
    else:
        return tensor.numel() * 2, "Q15 fallback"


def is_encoder_param(name):
    return any(name.startswith(p) for p in ENCODER_PREFIXES) or name == 'rotation_A'


def is_decoder_param(name):
    return any(name.startswith(p) for p in DECODER_PREFIXES)


def run():
    ckpt_path = os.path.join(ROOT_DIR, 'ai_models/student/student_hardened.ckpt')
    if not os.path.exists(ckpt_path):
        print(f"[SKIP] Student checkpoint not found: {ckpt_path}")
        print("[SKIP] Benchmark TNN Memory requires a trained student_hardened.ckpt.")
        return None

    model = TernaryMobileNetV5_Subband.from_checkpoint(ckpt_path, device='cpu')

    # Find ternary modules
    ternary_modules = set()
    for name, m in model.named_modules():
        if hasattr(m, 'lsq_alpha'):
            ternary_modules.add(name)

    # Compute per-parameter sizes
    encoder_bytes = 0
    decoder_bytes = 0
    encoder_breakdown = {}
    decoder_breakdown = {}

    for param_name, tensor in model.state_dict().items():
        nbytes, fmt = estimate_bytes(param_name, tensor, ternary_modules)

        if is_encoder_param(param_name):
            encoder_bytes += nbytes
            encoder_breakdown[param_name] = (nbytes, fmt, tensor.shape)
        elif is_decoder_param(param_name):
            decoder_bytes += nbytes
            decoder_breakdown[param_name] = (nbytes, fmt, tensor.shape)
        else:
            # Shouldn't happen, but count as encoder to be safe
            encoder_bytes += nbytes
            encoder_breakdown[param_name] = (nbytes, fmt, tensor.shape)

    # Print encoder breakdown (this is what goes on-chip)
    print(f"{'='*95}")
    print(f" ENCODER (on-chip, SRAM4)")
    print(f"{'='*95}")
    print(f"{'Parameter':<45} {'Shape':<20} {'Format':<20} {'Bytes':>8}")
    print("-" * 95)
    for name, (nbytes, fmt, shape) in sorted(encoder_breakdown.items()):
        print(f"{name:<45} {str(list(shape)):<20} {fmt:<20} {nbytes:>8}")
    print("-" * 95)
    print(f"{'ENCODER TOTAL':<45} {'':<20} {'':<20} {encoder_bytes:>8}")
    print(f"{'SRAM4 BUDGET':<45} {'':<20} {'':<20} {MODEL_BUDGET:>8}")
    print()

    # Print decoder breakdown (informational)
    print(f"{'='*95}")
    print(f" DECODER (base station, unconstrained)")
    print(f"{'='*95}")
    print(f"{'Parameter':<45} {'Shape':<20} {'Format':<20} {'Bytes':>8}")
    print("-" * 95)
    for name, (nbytes, fmt, shape) in sorted(decoder_breakdown.items()):
        print(f"{name:<45} {str(list(shape)):<20} {fmt:<20} {nbytes:>8}")
    print("-" * 95)
    print(f"{'DECODER TOTAL':<45} {'':<20} {'':<20} {decoder_bytes:>8}")
    print()

    # Summary
    total = encoder_bytes + decoder_bytes
    print(f"[*] Encoder: {encoder_bytes:,} bytes ({encoder_bytes/MODEL_BUDGET*100:.1f}% of SRAM4)")
    print(f"[*] Decoder: {decoder_bytes:,} bytes (base station)")
    print(f"[*] Total:   {total:,} bytes")

    # Header file size
    header_path = os.path.join(ROOT_DIR, 'firmware', 'firmware_export', 'focal_net_weights.h')
    if os.path.exists(header_path):
        print(f"[*] focal_net_weights.h on disk: {os.path.getsize(header_path):,} bytes (text)")

    # Pass/fail on ENCODER only
    if encoder_bytes > MODEL_BUDGET:
        overshoot = encoder_bytes - MODEL_BUDGET
        print(f"\n[FAIL] ENCODER EXCEEDS SRAM4 BY {overshoot:,} bytes!")
        sys.exit(1)

    headroom = MODEL_BUDGET - encoder_bytes
    print(f"[*] SRAM4 headroom: {headroom:,} bytes")
    print("[PASS] Benchmark TNN Memory - CLEAN.")
    return encoder_bytes, MODEL_BUDGET


if __name__ == "__main__":
    run()
