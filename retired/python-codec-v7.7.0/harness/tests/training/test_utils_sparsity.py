"""
L2 — Utility function invariants: percent_zero_weights().

Validates weight sparsity calculation for all-zero, no-zero, half-zero,
empty models, and ternary layers with LSQ alpha thresholding.
"""
import pytest
import torch
import torch.nn as nn


@pytest.mark.l2
class TestPercentZeroWeights:
    def test_all_zero_weights(self):
        from utils import percent_zero_weights
        model = nn.Linear(10, 10, bias=False)
        nn.init.zeros_(model.weight)
        assert percent_zero_weights(model) == 100.0

    def test_no_zero_weights(self):
        from utils import percent_zero_weights
        model = nn.Linear(10, 10, bias=False)
        nn.init.ones_(model.weight)
        assert percent_zero_weights(model) == 0.0

    def test_half_zero(self):
        from utils import percent_zero_weights
        model = nn.Linear(10, 10, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.0)
            model.weight[:5, :] = 1.0
        result = percent_zero_weights(model)
        assert abs(result - 50.0) < 0.01

    def test_empty_model_returns_zero(self):
        """Model with no weight parameters should return 0.0, not crash."""
        from utils import percent_zero_weights
        model = nn.Sequential()  # no parameters
        assert percent_zero_weights(model) == 0.0

    def test_with_lsq_alpha(self):
        """Ternary layer: conv weights below alpha count as zero.

        SubLN adds GroupNorm weight (=1) and bias (=0) params that
        shift the total — we check >85% instead of ==100%.
        """
        from utils import percent_zero_weights
        from lamquant_codec.models.blocks import TernaryConv1d

        conv = TernaryConv1d(4, 8, kernel_size=3)
        with torch.no_grad():
            conv.weight.fill_(0.01)
            conv.lsq_alpha.fill_(1.0)
        result = percent_zero_weights(conv)
        assert result > 85.0, f"Expected >85% zero, got {result:.1f}%"

    def test_with_lsq_alpha_none_zero(self):
        """Ternary layer where all weights exceed alpha."""
        from utils import percent_zero_weights
        from lamquant_codec.models.blocks import TernaryConv1d

        conv = TernaryConv1d(4, 8, kernel_size=3)
        with torch.no_grad():
            conv.weight.fill_(5.0)
            conv.lsq_alpha.fill_(0.01)
        # All |w| = 5.0 > alpha = 0.01
        result = percent_zero_weights(conv)
        assert result == 0.0
