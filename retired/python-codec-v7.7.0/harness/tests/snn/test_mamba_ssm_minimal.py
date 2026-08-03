"""Unit tests for ai_models/snn/mamba_ssm_minimal.py — coverage gap supplement."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from ai_models.snn.mamba_ssm_minimal import (
    BidirectionalSSM,
    HomeostaticThresholdAdapter,
    MambaSNN,
    SelectiveSSM,
)
# Private quantizer helpers live in the canonical module; the
# ai_models shim's `from ... import *` only re-exports public names.
from lamquant_codec.models.mamba_ssm_minimal import (
    _int3_quantize_decay,
    _int4_quantize_ste,
)

pytestmark = pytest.mark.l2


# ---------------------------------------------------------------------------
# HomeostaticThresholdAdapter
# ---------------------------------------------------------------------------
class TestHomeostaticThresholdAdapter:
    def test_construction(self):
        h = HomeostaticThresholdAdapter(n_channels=40)
        assert h.ema_rate.shape == (40,)
        assert h.threshold_adj.shape == (40,)
        assert h.target_rate == 0.1

    def test_custom_params(self):
        h = HomeostaticThresholdAdapter(n_channels=8, target_rate=0.2,
                                          tau=0.5)
        assert h.target_rate == 0.2
        assert h.tau == 0.5

    def test_update_changes_ema(self):
        h = HomeostaticThresholdAdapter(n_channels=4)
        rates = np.array([0.5, 0.5, 0.5, 0.5])
        h.update(rates)
        # ema = 0.99*0 + 0.01*0.5 = 0.005
        assert h.ema_rate[0] == pytest.approx(0.005, abs=1e-6)

    def test_update_decreases_threshold_when_too_active(self):
        h = HomeostaticThresholdAdapter(n_channels=4, target_rate=0.1,
                                          tau=0.5)
        rates = np.array([1.0, 1.0, 1.0, 1.0])  # 10x target
        h.update(rates)
        # ema = 0.5; error = 0.4 → threshold_adj -= 0.004 → negative
        assert (h.threshold_adj < 0).all()

    def test_get_adjusted_thresholds(self):
        h = HomeostaticThresholdAdapter(n_channels=4)
        h.threshold_adj = np.array([0.1, -0.1, 0.2, 0.0])
        out = h.get_adjusted_thresholds(base_threshold=1.0)
        expected = np.array([1.1, 0.9, 1.2, 1.0])
        assert np.allclose(out, expected)


# ---------------------------------------------------------------------------
# _int4_quantize_ste
# ---------------------------------------------------------------------------
class TestInt4Quantize:
    def test_returns_torch_tensor(self):
        w = torch.randn(8, 4)
        out = _int4_quantize_ste(w)
        assert out.shape == w.shape
        assert isinstance(out, torch.Tensor)

    def test_zero_weight_no_div_by_zero(self):
        w = torch.zeros(4)
        out = _int4_quantize_ste(w)
        assert torch.isfinite(out).all()

    def test_gradient_flows_via_ste(self):
        w = torch.randn(4, requires_grad=True)
        out = _int4_quantize_ste(w)
        out.sum().backward()
        assert w.grad is not None


# ---------------------------------------------------------------------------
# _int3_quantize_decay
# ---------------------------------------------------------------------------
class TestInt3QuantizeDecay:
    def test_snaps_to_levels(self):
        A_log = torch.tensor([-2.0, -1.0, 0.0, 0.5])
        out = _int3_quantize_decay(A_log)
        # Levels include exactly -2, -1, 0, 0.5 → exact match
        assert torch.allclose(out, A_log, atol=1e-5)

    def test_intermediate_values_snap(self):
        # -2.4 is between -3 (idx 1) and -2 (idx 2). |-2.4-(-3)|=0.6,
        # |-2.4-(-2)|=0.4 → snaps to -2.
        A_log = torch.tensor([-2.4])
        out = _int3_quantize_decay(A_log)
        assert out.item() == pytest.approx(-2.0, abs=1e-5)

    def test_gradient_flows(self):
        A_log = torch.randn(4, requires_grad=True)
        out = _int3_quantize_decay(A_log)
        out.sum().backward()
        assert A_log.grad is not None


# ---------------------------------------------------------------------------
# SelectiveSSM forward — exercise the CPU fallback
# ---------------------------------------------------------------------------
class TestSelectiveSSM:
    def test_forward_shape(self):
        m = SelectiveSSM(d_model=8, d_state=4, d_conv=4, expand=2)
        x = torch.randn(2, 16, 8)
        out = m(x)
        assert out.shape == (2, 16, 8)

    def test_forward_quantize_path(self):
        m = SelectiveSSM(d_model=8, d_state=4, d_conv=4, expand=2)
        x = torch.randn(2, 16, 8)
        out = m(x, quantize=True)
        assert out.shape == (2, 16, 8)


# ---------------------------------------------------------------------------
# BidirectionalSSM smoke
# ---------------------------------------------------------------------------
class TestBidirectionalSSM:
    def test_forward_shape(self):
        m = BidirectionalSSM(d_model=8, d_state=4)
        x = torch.randn(2, 16, 8)
        out = m(x)
        assert out.shape == (2, 16, 8)


# ---------------------------------------------------------------------------
# MambaSNN smoke
# ---------------------------------------------------------------------------
class TestMambaSNNSmoke:
    def test_forward_shape(self):
        m = MambaSNN(in_channels=21, d_model=8, d_state=4, n_layers=1)
        x = torch.randn(1, 21, 313)  # [B, C, T]
        logits, rate = m(x)
        # Output should be activity logits
        assert logits.ndim == 3
        assert isinstance(rate, (float, torch.Tensor))
