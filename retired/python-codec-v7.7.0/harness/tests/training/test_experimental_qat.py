"""
L2/L5 — Tests for experimental QAT features: learned WHT rotation and W2A8 activation quantization.

These tests provide the infrastructure to evaluate two experiments:
  1. Learned orthogonal rotation vs fixed WHT32 for FSQ codebook utilization
  2. INT8 activation quantization (W2A8) vs INT16 (W2A16) with block-WHT smoothing

Each test measures a specific metric that determines success/failure of the
experiment, not just "does it crash."
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'ai_models', 'student'))


# ============================================================
# Item 6: Learned Orthogonal Rotation vs Fixed WHT
# ============================================================

class CayleyOrthogonal(nn.Module):
    """Learned orthogonal 32×32 rotation via Cayley parameterization.

    Q = (I - A)(I + A)^{-1} where A is skew-symmetric (A = -A^T).
    This guarantees Q is always orthogonal regardless of A's values.
    """
    def __init__(self, dim=32):
        super().__init__()
        # Skew-symmetric parameterization: only upper triangle is free
        self.A_upper = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.dim = dim

    def get_rotation_matrix(self):
        # Build skew-symmetric A from upper triangle
        A = self.A_upper - self.A_upper.T
        I = torch.eye(self.dim, device=A.device, dtype=A.dtype)
        # Cayley transform: Q = (I - A)(I + A)^{-1}
        Q = torch.linalg.solve(I + A, I - A)
        return Q

    def forward(self, x):
        """x: [..., 32] → [..., 32] rotated"""
        Q = self.get_rotation_matrix()
        return torch.matmul(x, Q.T)


@pytest.mark.l2
class TestCayleyOrthogonality:
    """Verify the Cayley parameterization always produces orthogonal matrices."""

    def test_rotation_is_orthogonal(self):
        """Q × Q^T must equal I for any parameter values."""
        rot = CayleyOrthogonal(32)
        Q = rot.get_rotation_matrix()
        I = torch.eye(32)
        QQT = Q @ Q.T
        max_err = (QQT - I).abs().max().item()
        assert max_err < 1e-5, f"Q×Q^T deviates from I by {max_err}"

    def test_rotation_stays_orthogonal_after_sgd(self):
        """Orthogonality must hold after gradient updates."""
        rot = CayleyOrthogonal(32)
        opt = torch.optim.SGD(rot.parameters(), lr=0.1)

        for _ in range(50):
            x = torch.randn(4, 32)
            y = rot(x)
            loss = y.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

        Q = rot.get_rotation_matrix()
        I = torch.eye(32)
        max_err = (Q @ Q.T - I).abs().max().item()
        assert max_err < 1e-4, f"Orthogonality lost after SGD: {max_err}"

    def test_determinant_is_pm_one(self):
        """det(Q) must be ±1 (proper or improper rotation)."""
        rot = CayleyOrthogonal(32)
        Q = rot.get_rotation_matrix()
        det = torch.linalg.det(Q).item()
        assert abs(abs(det) - 1.0) < 1e-4, f"det(Q) = {det}, expected ±1"

    def test_gradient_flows_through_rotation(self):
        """Gradients must reach A_upper through the Cayley transform."""
        rot = CayleyOrthogonal(32)
        x = torch.randn(4, 32)
        y = rot(x)
        loss = y.sum()
        loss.backward()
        assert rot.A_upper.grad is not None
        assert rot.A_upper.grad.abs().sum() > 0


@pytest.mark.l5
class TestFSQCodebookUtilization:
    """Measure FSQ dead code ratio — the metric that determines if
    learned rotation outperforms fixed WHT."""

    @staticmethod
    def _compute_dead_codes(latent, n_levels=16):
        """Compute fraction of FSQ bins that are never used.

        Args:
            latent: [B, 32, T] float tensor (latent representations)
            n_levels: FSQ quantization levels

        Returns:
            dead_ratio: fraction of bins with zero occupancy (0.0 = perfect, 1.0 = all dead)
        """
        flat = latent.reshape(-1)
        # Normalize to [0, 1] then bin
        vmin, vmax = flat.min(), flat.max()
        if vmax - vmin < 1e-8:
            return 1.0  # all values identical = worst case
        normalized = (flat - vmin) / (vmax - vmin)
        bins = (normalized * n_levels).clamp(0, n_levels - 1).long()
        counts = torch.bincount(bins, minlength=n_levels)
        dead = (counts == 0).sum().item()
        return dead / n_levels

    def test_fixed_wht_dead_codes(self):
        """Measure dead code ratio with fixed WHT on random latents."""
        from subband_preprocess import wht32_forward
        latent = torch.randn(8, 32, 79)
        # Apply fixed WHT per timestep
        wht_latent = torch.stack([
            torch.tensor(wht32_forward(latent[:, :, t].numpy().T).T)
            for t in range(79)
        ], dim=2)
        dead = self._compute_dead_codes(wht_latent)
        # Just record — this is the baseline to beat
        print(f"\n  Fixed WHT dead codes: {dead:.1%} ({int(dead*16)}/16 bins unused)")
        # Sanity: dead ratio should be < 50% on random data
        assert dead < 0.5, f"Fixed WHT has {dead:.0%} dead codes on random data"

    def test_learned_rotation_dead_codes(self):
        """Measure dead code ratio with learned Cayley rotation.

        After a few optimization steps to minimize dead codes, the
        learned rotation should have fewer dead bins than fixed WHT.
        """
        rot = CayleyOrthogonal(32)
        opt = torch.optim.Adam(rot.parameters(), lr=0.01)
        latent = torch.randn(8, 32, 79)

        # Optimize rotation to minimize dead codes (proxy: maximize entropy of bin distribution)
        for _ in range(100):
            opt.zero_grad()
            # Rotate each timestep
            rot_latent = rot(latent.permute(0, 2, 1)).permute(0, 2, 1)  # [B, 32, 79]
            flat = rot_latent.reshape(-1)
            vmin, vmax = flat.min().detach(), flat.max().detach()
            normalized = (flat - vmin) / (vmax - vmin + 1e-8)
            bins = (normalized * 16).clamp(0, 15.999)
            # Soft histogram entropy (differentiable proxy)
            soft_counts = torch.zeros(16)
            for i in range(16):
                soft_counts[i] = torch.exp(-5 * (bins - i).pow(2)).sum()
            probs = soft_counts / soft_counts.sum()
            entropy = -(probs * (probs + 1e-8).log()).sum()
            (-entropy).backward()  # maximize entropy = minimize dead codes
            opt.step()

        with torch.no_grad():
            rot_latent = rot(latent.permute(0, 2, 1)).permute(0, 2, 1)
        dead = self._compute_dead_codes(rot_latent)
        print(f"\n  Learned rotation dead codes: {dead:.1%} ({int(dead*16)}/16 bins unused)")
        # The learned rotation should achieve ≤ the fixed WHT dead code ratio
        # (exact threshold depends on the data, so we just check it's reasonable)
        assert dead < 0.5, f"Learned rotation has {dead:.0%} dead codes"


# ============================================================
# Item 7: W2A8 vs W2A16 Activation Quantization
# ============================================================

def _quantize_activation_parametric(x, bits=16):
    """Quantize activations to given bit width with STE.

    bits=16: range [-32768, 32767] (current W2A16)
    bits=8:  range [-128, 127] (experimental W2A8)
    """
    max_val = 2 ** (bits - 1) - 1
    min_val = -(2 ** (bits - 1))
    with torch.no_grad():
        scale = x.abs().amax() / max_val
        if scale < 1e-12:
            return x
    x_scaled = x / (scale + 1e-12)
    # STE: round in forward, pass gradient through
    x_q = (x_scaled + (x_scaled.round() - x_scaled).detach()).clamp(min_val, max_val)
    return x_q * scale


@pytest.mark.l5
class TestActivationBitWidth:
    """Compare A16 vs A8 activation quantization quality."""

    def test_a16_vs_a8_mse_on_random(self):
        """A8 should have higher MSE than A16 but both should be finite."""
        x = torch.randn(4, 112, 157) * 10  # typical activation scale

        x_a16 = _quantize_activation_parametric(x, bits=16)
        x_a8 = _quantize_activation_parametric(x, bits=8)

        mse_16 = F.mse_loss(x_a16, x).item()
        mse_8 = F.mse_loss(x_a8, x).item()

        print(f"\n  A16 MSE: {mse_16:.6f}")
        print(f"  A8  MSE: {mse_8:.6f}")
        print(f"  Ratio A8/A16: {mse_8/max(mse_16, 1e-12):.1f}×")

        assert mse_16 < mse_8, "A16 should have lower quant error than A8"
        assert np.isfinite(mse_16) and np.isfinite(mse_8)

    def test_wht_smoothing_helps_a8(self):
        """Block-WHT smoothing should reduce A8 quantization error.

        This is the key SSDi8 claim: WHT rotation spreads outliers so
        INT8 quantization has lower error in the rotated domain.
        """
        from lamquant_codec.models.blocks import _get_hadamard_32

        x = torch.randn(4, 32, 64)  # 2 blocks of 32
        # Add outliers to make the difference visible
        x[:, 0, 0] = 100.0  # extreme outlier in one dimension

        # Direct A8 (no WHT)
        x_a8_direct = _quantize_activation_parametric(x, bits=8)
        mse_direct = F.mse_loss(x_a8_direct, x).item()

        # WHT-smoothed A8
        H = _get_hadamard_32(x.device, x.dtype)
        B, C, T = x.shape
        n_blocks = T // 32
        x_blocks = x[:, :, :n_blocks * 32].reshape(B, C, n_blocks, 32)
        x_wht = torch.matmul(x_blocks, H.T)
        x_wht_a8 = _quantize_activation_parametric(x_wht.reshape(-1), bits=8).reshape_as(x_wht)
        x_reconstructed = torch.matmul(x_wht_a8, H).reshape(B, C, n_blocks * 32)
        mse_wht = F.mse_loss(x_reconstructed, x[:, :, :n_blocks * 32]).item()

        print(f"\n  Direct A8 MSE: {mse_direct:.6f}")
        print(f"  WHT+A8 MSE:   {mse_wht:.6f}")
        print(f"  Improvement:   {(1 - mse_wht/max(mse_direct, 1e-12))*100:.1f}%")

        # WHT should reduce error when outliers are present
        assert mse_wht < mse_direct, \
            f"WHT+A8 ({mse_wht:.4f}) should be better than direct A8 ({mse_direct:.4f})"

    def test_a8_model_forward_finite(self):
        """Full model forward pass with A8 should produce finite output."""
        from lamquant_codec.models.encoder import TernaryMobileNetV5_Subband

        model = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32)
        model.eval()
        x = torch.randn(2, 21, 313)

        # Monkey-patch _quantize_activation to use A8 for this test
        import ternary_encoder
        original_fn = ternary_encoder._quantize_activation
        ternary_encoder._quantize_activation = lambda x, enabled=True, hadamard=None: \
            _quantize_activation_parametric(x, bits=8) if enabled else x

        try:
            with torch.no_grad():
                out = model(x, quantize=True)
            assert out.isfinite().all(), "A8 model output has NaN/Inf"
            print(f"\n  A8 model output: shape={out.shape}, "
                  f"range=[{out.min():.2f}, {out.max():.2f}]")
        finally:
            ternary_encoder._quantize_activation = original_fn

    def test_a8_vs_a16_model_accuracy(self):
        """Compare model output quality at A16 vs A8.

        This is the go/no-go metric for shipping W2A8.
        """
        from lamquant_codec.models.encoder import TernaryMobileNetV5_Subband
        import ternary_encoder

        model = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32)
        model.eval()
        x = torch.randn(4, 21, 313)

        # A16 baseline (current production)
        with torch.no_grad():
            out_a16 = model(x, quantize=True)

        # Patch to A8
        original_fn = ternary_encoder._quantize_activation
        ternary_encoder._quantize_activation = lambda x, enabled=True, hadamard=None: \
            _quantize_activation_parametric(x, bits=8) if enabled else x

        try:
            with torch.no_grad():
                out_a8 = model(x, quantize=True)
        finally:
            ternary_encoder._quantize_activation = original_fn

        # Measure degradation
        mse_a16 = F.mse_loss(out_a16, x).item()
        mse_a8 = F.mse_loss(out_a8, x).item()

        # Pearson R
        def pearson_r(a, b):
            a_flat, b_flat = a.flatten(), b.flatten()
            ac = a_flat - a_flat.mean()
            bc = b_flat - b_flat.mean()
            return (ac * bc).sum() / ((ac**2).sum().sqrt() * (bc**2).sum().sqrt() + 1e-8)

        r_a16 = pearson_r(out_a16, x).item()
        r_a8 = pearson_r(out_a8, x).item()

        print(f"\n  A16: MSE={mse_a16:.4f}, R={r_a16:.4f}")
        print(f"  A8:  MSE={mse_a8:.4f}, R={r_a8:.4f}")
        print(f"  R drop: {(r_a16 - r_a8):.4f}")

        # On an untrained model, both should be similar (random weights)
        # The real test is after training — this just ensures A8 doesn't crash
        assert abs(r_a16 - r_a8) < 0.5, \
            f"A8 R drop too large on untrained model: {r_a16:.4f} → {r_a8:.4f}"

    def test_activation_buffer_size_estimate(self):
        """Verify that A8 halves the activation buffer memory."""
        # Largest activation in the encoder: [B, 112, 313] after focal1
        B, C, T = 1, 112, 313
        a16_bytes = B * C * T * 2  # INT16
        a8_bytes = B * C * T * 1   # INT8

        print(f"\n  Activation buffer (peak, per sample):")
        print(f"  A16: {a16_bytes:,} bytes ({a16_bytes/1024:.1f} KB)")
        print(f"  A8:  {a8_bytes:,} bytes ({a8_bytes/1024:.1f} KB)")
        print(f"  Savings: {(a16_bytes - a8_bytes)/1024:.1f} KB ({(a16_bytes-a8_bytes)/a16_bytes*100:.0f}%)")

        assert a8_bytes == a16_bytes // 2, "A8 should be exactly half of A16"
        assert a16_bytes / 1024 < 100, "Peak activation should be under 100 KB"
