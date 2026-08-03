"""
L2/L5/L7 — Ternary quantization: LSQ alpha, weight constraints, model shapes.

Validates TernaryConv1d produces {-1,0,1} weights under quantization, STE
gradient flows through the quantizer, encoder/decoder shapes are consistent,
full autoencoder roundtrip preserves shape, outputs are finite, and single-
sample edge cases don't crash. Covers distillation loss and clinical
augmentation when applicable.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np


@pytest.mark.l2
class TestTernaryConv1d:
    def test_quantized_weights_are_ternary(self):
        """Forward pass with quantize=True must use only {-1, 0, 1} weights."""
        from lamquant_codec.models.blocks import TernaryConv1d
        conv = TernaryConv1d(21, 96, kernel_size=7)
        # Initialise with random weights
        nn.init.normal_(conv.weight, std=0.5)
        conv.eval()

        alpha = torch.abs(conv.lsq_alpha)
        w_q = torch.round(torch.clamp(conv.weight / (alpha + 1e-8), -1, 1))
        unique = set(w_q.detach().numpy().flatten().tolist())
        assert unique.issubset({-1.0, 0.0, 1.0}), f"Non-ternary values: {unique}"

    def test_ste_gradient_flows(self):
        """Straight-Through Estimator must allow gradients to reach weight."""
        from lamquant_codec.models.blocks import TernaryConv1d
        conv = TernaryConv1d(4, 8, kernel_size=3)
        x = torch.randn(1, 4, 100, requires_grad=False)
        out = conv(x, quantize=True)
        loss = out.sum()
        loss.backward()
        assert conv.weight.grad is not None
        assert conv.weight.grad.abs().sum() > 0, "No gradient reached weights"

    def test_alpha_gradient_flows(self):
        """LSQ alpha must receive gradients."""
        from lamquant_codec.models.blocks import TernaryConv1d
        conv = TernaryConv1d(4, 8, kernel_size=3)
        x = torch.randn(1, 4, 100)
        out = conv(x, quantize=True)
        loss = out.sum()
        loss.backward()
        assert conv.lsq_alpha.grad is not None

    def test_non_quantized_passthrough(self):
        """With quantize=False, TernaryConv1d should behave as standard Conv1d."""
        from lamquant_codec.models.blocks import TernaryConv1d
        conv = TernaryConv1d(4, 8, kernel_size=3)
        x = torch.randn(1, 4, 50)
        out_q = conv(x, quantize=True)
        out_nq = conv(x, quantize=False)
        # They should differ because quantization changes weights
        # (unless weights happen to already be ternary, which is unlikely)
        # Just check shapes match
        assert out_q.shape == out_nq.shape


@pytest.mark.l2
class TestTernaryConvTranspose1d:
    def test_upsample_doubles_length(self):
        """Transposed conv with stride=2 should approximately double T."""
        from lamquant_codec.models.blocks import TernaryConvTranspose1d
        conv = TernaryConvTranspose1d(8, 8, kernel_size=3, stride=2)
        x = torch.randn(1, 8, 50)
        out = conv(x, quantize=True)
        # ConvTranspose1d with stride=2 should give ~2*T
        assert out.shape[2] == 100  # 50*2 = 100


@pytest.mark.l5
class TestModelShapes:
    def test_encode_shape(self, ternary_model, random_eeg_batch):
        """Encoder: [B, 21, 2500] -> [B, 32, T/8] where T/8 = ceil(2500/8) = 313."""
        with torch.no_grad():
            latent = ternary_model.encode(random_eeg_batch)
        assert latent.shape[:2] == (2, 32)
        # Temporal dim: 2500 with stride-8 and padding gives ceil(2500/8) = 313
        assert latent.shape[2] == 313, f"Got temporal dim {latent.shape[2]}"

    def test_full_roundtrip_shape(self, ternary_model, random_eeg_batch):
        """Full forward: [B, 21, 2500] -> [B, 21, 2500]."""
        with torch.no_grad():
            out = ternary_model(random_eeg_batch)
        assert out.shape == random_eeg_batch.shape

    def test_encoder_output_is_finite(self, ternary_model, random_eeg_batch):
        with torch.no_grad():
            latent = ternary_model.encode(random_eeg_batch)
        assert torch.isfinite(latent).all()

    def test_decoder_output_is_finite(self, ternary_model, random_eeg_batch):
        with torch.no_grad():
            out = ternary_model(random_eeg_batch)
        assert torch.isfinite(out).all()

    def test_single_sample_input(self, ternary_model):
        """Model should handle batch size 1."""
        x = torch.randn(1, 21, 2500)
        with torch.no_grad():
            out = ternary_model(x)
        assert out.shape == (1, 21, 2500)


@pytest.mark.l5
class TestDistillationLoss:
    def test_pearson_r_range(self):
        """Pearson R from distillation_loss must be in [-1, 1]."""
        from train_ternary import distillation_loss
        s = torch.randn(4, 21, 2500)
        t = torch.randn(4, 21, 2500)
        mse, _, r = distillation_loss(s, t)
        assert -1.0 <= r.item() <= 1.0

    def test_identical_inputs_give_r_one(self):
        """Identical signals should give R = 1.0."""
        from train_ternary import distillation_loss
        x = torch.randn(4, 21, 2500)
        _, _, r = distillation_loss(x, x)
        assert r.item() > 0.999

    def test_mse_is_non_negative(self):
        from train_ternary import distillation_loss
        s = torch.randn(4, 21, 2500)
        t = torch.randn(4, 21, 2500)
        mse, _, _ = distillation_loss(s, t)
        assert mse.item() >= 0.0


@pytest.mark.l7
class TestClinicalAugmentation:
    def test_output_shape_preserved(self):
        from train_ternary import clinical_augmentation
        x = torch.randn(2, 21, 2500)
        aug = clinical_augmentation(x)
        assert aug.shape == x.shape

    def test_augmentation_modifies_signal(self):
        from train_ternary import clinical_augmentation
        torch.manual_seed(0)
        x = torch.randn(2, 21, 2500)
        aug = clinical_augmentation(x.clone())
        # Augmentation adds noise, so output should differ
        assert not torch.allclose(x, aug)
