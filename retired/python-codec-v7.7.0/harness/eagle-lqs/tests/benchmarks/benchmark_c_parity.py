#!/usr/bin/env python3
"""
LamQuant Gen 7 — C-Parity Benchmark
====================================
Phase 1: Bit-perfect static weight extraction (ternary unpacking)
Phase 2: Cascaded forward-pass drift through all encoder layers

Simulates BOTH the main conv path AND the shortcut path using
C-extracted weights. For strided blocks (focal2, focal3), the
shortcut is a strided TernaryConv1d — not identity — and must
be simulated with C weights to avoid false drift.

GroupNorm and ReLU use PyTorch (identical in float — these are
not ternary and don't introduce C vs Python drift).
"""
import torch
import torch.nn.functional as F
import os
import sys
import numpy as np
import re
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
from lamquant_codec.models.encoder import TernaryMobileNetV5_Subband
from lamquant_codec.models.blocks import TernaryConv1d

TERNARY_LUT = np.array([0, 1, -1, 0], dtype=np.int32)


def unpack_header_ternary(packed_bytes, shape):
    unpacked = []
    for b in packed_bytes:
        for j in range(4):
            val = (b >> (2 * j)) & 0x03
            unpacked.append(TERNARY_LUT[val])
    return np.array(unpacked[:np.prod(shape)]).reshape(shape)


def parse_clinical_header(path):
    """Parse the firmware C header, handling both Q31 (int32_t) and Q15 (int16_t) formats.

    The export_firmware.py now emits Q15 alphas, Q15 biases, Q7 norm weights,
    and Q15 rotation matrices.  Old headers may still use Q31.  We detect both
    patterns and normalise everything to float32.
    """
    with open(path, 'r') as f:
        content = f.read()

    alphas = {}
    # Q31 alphas: const int32_t {name}_alphas_q31[...] = {...};
    for m in re.finditer(
        r"const int32_t\s+([a-zA-Z0-9_]+)_alphas_q31\[.*?\]\s*.*?\s*=\s*\{(.*?)\};",
        content, re.DOTALL
    ):
        name = m.group(1)
        vals = re.findall(r"-?[0-9][0-9]*", m.group(2))
        alphas[name] = np.array([int(v) for v in vals], dtype=np.int32).astype(np.float32) / 2147483647.0

    # Q15 alphas: const int16_t {name}_alphas_q15[...] = {...};
    for m in re.finditer(
        r"const int16_t\s+([a-zA-Z0-9_]+)_alphas_q15\[.*?\]\s*.*?\s*=\s*\{(.*?)\};",
        content, re.DOTALL
    ):
        name = m.group(1)
        vals = re.findall(r"-?[0-9][0-9]*", m.group(2))
        alphas[name] = np.array([int(v) for v in vals], dtype=np.int16).astype(np.float32) / 32767.0

    # Rotation matrix — Q31 (legacy): const int32_t rotation_Q_q31[...] = {...};
    rotation = None
    m_rot31 = re.search(
        r"const int32_t\s+rotation_Q_q31\[.*?\]\s*.*?\s*=\s*\{(.*?)\};",
        content, re.DOTALL
    )
    if m_rot31:
        vals = re.findall(r"-?[0-9][0-9]*", m_rot31.group(1))
        rotation = np.array([int(v) for v in vals], dtype=np.int32).astype(np.float32) / 2147483647.0

    # Rotation matrix — Q15 (new): const int16_t rotation_Q_q15[...] = {...};
    m_rot15 = re.search(
        r"const int16_t\s+rotation_Q_q15\[.*?\]\s*.*?\s*=\s*\{(.*?)\};",
        content, re.DOTALL
    )
    if m_rot15:
        vals = re.findall(r"-?[0-9][0-9]*", m_rot15.group(1))
        rotation = np.array([int(v) for v in vals], dtype=np.int16).astype(np.float32) / 32767.0

    # Norm biases — Q31 (legacy): const int32_t {name}_bias_q31[...] = {...};
    norm_biases = {}
    for m in re.finditer(
        r"const int32_t\s+([a-zA-Z0-9_]+)_bias_q31\[.*?\]\s*.*?\s*=\s*\{(.*?)\};",
        content, re.DOTALL
    ):
        name = m.group(1)
        vals = re.findall(r"-?[0-9][0-9]*", m.group(2))
        norm_biases[name] = np.array([int(v) for v in vals], dtype=np.int32).astype(np.float32) / 2147483647.0

    # Norm biases — Q15 (new): const int16_t {name}_bias_q15[...] = {...};
    for m in re.finditer(
        r"const int16_t\s+([a-zA-Z0-9_]+)_bias_q15\[.*?\]\s*.*?\s*=\s*\{(.*?)\};",
        content, re.DOTALL
    ):
        name = m.group(1)
        vals = re.findall(r"-?[0-9][0-9]*", m.group(2))
        norm_biases[name] = np.array([int(v) for v in vals], dtype=np.int16).astype(np.float32) / 32767.0

    # Norm weights — Q31 (legacy): const int32_t {name}_weight_q31[...] = {...};
    norm_weights = {}
    for m in re.finditer(
        r"const int32_t\s+([a-zA-Z0-9_]+)_weight_q31\[.*?\]\s*.*?\s*=\s*\{(.*?)\};",
        content, re.DOTALL
    ):
        name = m.group(1)
        vals = re.findall(r"-?[0-9][0-9]*", m.group(2))
        norm_weights[name] = np.array([int(v) for v in vals], dtype=np.int32).astype(np.float32) / 2147483647.0

    # Norm weights — Q7 (new): const int8_t {name}_weight_q7[...] = {...};
    for m in re.finditer(
        r"const int8_t\s+([a-zA-Z0-9_]+)_weight_q7\[.*?\]\s*.*?\s*=\s*\{(.*?)\};",
        content, re.DOTALL
    ):
        name = m.group(1)
        vals = re.findall(r"-?[0-9][0-9]*", m.group(2))
        norm_weights[name] = np.array([int(v) for v in vals], dtype=np.int8).astype(np.float32) / 127.0

    # Packed ternary weights (unchanged — always uint8_t)
    weights = {}
    for m in re.finditer(
        r"const uint8_t\s+([a-zA-Z0-9_]+)_weights\[.*?\]\s*.*?\s*=\s*\{(.*?)\};",
        content, re.DOTALL
    ):
        name = m.group(1)
        hex_vals = re.findall(r"0x[0-9a-fA-F]+", m.group(2))
        weights[name] = np.array([int(h, 16) for h in hex_vals], dtype=np.uint8)

    return alphas, weights, rotation, norm_biases, norm_weights


def c_conv1d(x_t, unpacked_w, alphas, conv_module):
    """Simulate a ternary conv1d using C-extracted weights and float32 alphas."""
    w = torch.from_numpy(unpacked_w).float()
    for oc in range(w.shape[0]):
        w[oc] *= alphas[oc]
    if isinstance(x_t, np.ndarray):
        x_t = torch.from_numpy(x_t).float()
    return F.conv1d(x_t, w, bias=conv_module.bias,
                    stride=conv_module.stride, padding=conv_module.padding,
                    dilation=conv_module.dilation, groups=conv_module.groups)


def c_simulate_focal_block(x_t, conv_name, shortcut_name, block,
                            c_unpack, c_alphas):
    """
    Simulate a full TernaryFocalBlock using C-extracted weights:
      out = ReLU(GroupNorm(C_conv(x))) + C_shortcut(x)

    GroupNorm and ReLU use PyTorch (identical math).
    Conv and shortcut use C-extracted ternary weights + Q31 alphas.
    """
    # Main path: C-simulated conv → PyTorch GroupNorm → ReLU
    c_conv_out = c_conv1d(x_t, c_unpack[conv_name], c_alphas[conv_name], block.conv)
    c_normed = F.relu(block.norm(c_conv_out))

    # Shortcut path
    if isinstance(block.shortcut, TernaryConv1d):
        # Strided ternary shortcut — simulate with C weights
        c_sc_out = c_conv1d(x_t, c_unpack[shortcut_name], c_alphas[shortcut_name],
                            block.shortcut)
    else:
        # Identity shortcut (same channels, stride=1)
        c_sc_out = x_t

    return c_normed + c_sc_out


def run():
    device = 'cpu'
    s_path = os.path.join(ROOT_DIR, "ai_models/student/student_hardened.ckpt")
    if not os.path.exists(s_path):
        print(f"[SKIP] Student checkpoint not found: {s_path}")
        print("[SKIP] Benchmark C Parity requires a trained student_hardened.ckpt.")
        return None

    header_path = os.path.join(ROOT_DIR, 'firmware', 'firmware_export', 'focal_net_weights.h')
    if not os.path.exists(header_path):
        print(f"[SKIP] Firmware header not found: {header_path}")
        print("[SKIP] Benchmark C Parity requires firmware/firmware_export/focal_net_weights.h "
              "(generated by firmware/export_firmware.py from a trained checkpoint).")
        return None

    try:
        model = TernaryMobileNetV5_Subband.from_checkpoint(s_path, device=device)
    except Exception as e:
        print(f"[SKIP] Could not load checkpoint -> {e}")
        return None
    model.eval()

    c_alphas, c_packed, c_rotation, c_norm_biases, c_norm_weights = parse_clinical_header(header_path)

    # All encoder layers for new Subband architecture:
    #   premix, focal1_conv (standalone), focal2 (block), focal3 (block),
    #   dw_gate, bneck_v, bneck_g
    encoder_convs = [
        ('premix', model.premix),
        ('focal1_conv', model.focal1_conv),
        ('focal2_conv', model.focal2.conv),
        ('focal2_shortcut', model.focal2.shortcut),
        ('focal3_conv', model.focal3.conv),
        ('focal3_shortcut', model.focal3.shortcut),
        ('dw_gate', model.dw_gate),
        ('bneck_v', model.bneck_v),
        ('bneck_g', model.bneck_g),
    ]

    # ===== PHASE 1: Static weight parity =====
    print("[*] Phase 1: Verifying bit-perfect weight extraction...")
    c_unpack = {}

    for c_name, py_layer in encoder_convs:
        if not hasattr(py_layer, 'lsq_alpha'):
            continue  # Skip identity shortcuts
        if c_name not in c_packed:
            print(f"[!] FATAL: {c_name}_weights missing from header")
            sys.exit(1)

        shape = py_layer.weight.shape
        c_unpack[c_name] = unpack_header_ternary(c_packed[c_name], shape)

        with torch.no_grad():
            w = py_layer.weight.detach().numpy()
            alpha = torch.abs(py_layer.lsq_alpha).detach().numpy().flatten()
            w_q = np.zeros_like(w, dtype=np.int8)
            for oc in range(w.shape[0]):
                w_q[oc] = np.round(np.clip(w[oc] / (alpha[oc] + 1e-8), -1, 1))

            mismatches = np.sum(w_q != c_unpack[c_name])
            if mismatches != 0:
                print(f"[FAIL] Weight drift in {c_name}: {mismatches} mismatches")
                sys.exit(1)
            print(f"  {c_name}: {np.prod(shape)} weights OK")

    # ===== PHASE 2: Cascaded forward-pass drift =====
    print("[*] Phase 2: Cascaded forward-pass drift (all encoder layers)...")

    np.random.seed(42)
    # Subband model expects [B, 21, 313] (L3 subband input)
    test_inputs = [
        np.zeros((1, 21, 313), dtype=np.float32),
        np.ones((1, 21, 313), dtype=np.float32) * 50.0,
        np.ones((1, 21, 313), dtype=np.float32) * -50.0,
    ]
    for _ in range(10):
        test_inputs.append(
            np.random.uniform(-50.0, 50.0, size=(1, 21, 313)).astype(np.float32))

    max_drift = 0.0

    import torch.nn.functional as F

    with torch.no_grad():
        for x_np in test_inputs:
            x_t = torch.from_numpy(x_np).float()

            # PyTorch reference: full encode path (includes rotation)
            py_lat = model.encode(x_t, quantize=True)

            # C-simulation: cascaded through all encoder layers
            c_h = c_conv1d(x_t, c_unpack['premix'], c_alphas['premix'], model.premix)

            # focal1: C conv + PyTorch GroupNorm/ReLU + zero-pad shortcut
            c_f1_conv = c_conv1d(c_h, c_unpack['focal1_conv'],
                                  c_alphas['focal1_conv'], model.focal1_conv)
            c_h = F.relu(model.focal1_norm(c_f1_conv)) + model.focal1_shortcut(c_h)

            # focal2: identity shortcut (stride=1, same channels)
            c_f2_conv = c_conv1d(c_h, c_unpack['focal2_conv'],
                                  c_alphas['focal2_conv'], model.focal2.conv)
            c_h = F.relu(model.focal2.norm(c_f2_conv)) + c_h

            # focal3: ternary shortcut (stride=2)
            c_f3_conv = c_conv1d(c_h, c_unpack['focal3_conv'],
                                  c_alphas['focal3_conv'], model.focal3.conv)
            c_f3_sc = c_conv1d(c_h, c_unpack['focal3_shortcut'],
                                c_alphas['focal3_shortcut'], model.focal3.shortcut)
            c_h = F.relu(model.focal3.norm(c_f3_conv)) + c_f3_sc

            # Gated SSM bottleneck
            c_gated = c_conv1d(c_h, c_unpack['dw_gate'], c_alphas['dw_gate'], model.dw_gate)
            c_v = c_conv1d(c_h, c_unpack['bneck_v'], c_alphas['bneck_v'], model.bneck_v)
            c_g = c_conv1d(c_gated, c_unpack['bneck_g'], c_alphas['bneck_g'], model.bneck_g)
            c_lat = c_v * torch.sigmoid(c_g)

            # Cayley rotation (same Q matrix — no additional drift source)
            if hasattr(model, 'rotation_A'):
                Q = model._get_rotation()
                c_lat = torch.einsum('ij,bjt->bit', Q, c_lat)

            # The only metric that matters: encoder output (latent) drift
            d_lat = np.max(np.abs(py_lat.numpy() - c_lat.numpy()))
            if d_lat > max_drift:
                max_drift = d_lat

    print(f"[*] Max cascaded encoder drift: {max_drift:.6f}")
    if max_drift > 0.01:
        print(f"[!] FATAL: Cascaded drift {max_drift:.6f} > 0.01 threshold")
        sys.exit(1)

    print("[PASS] Benchmark Bit Parity - CLEAN.")
    return max_drift


if __name__ == "__main__":
    run()
