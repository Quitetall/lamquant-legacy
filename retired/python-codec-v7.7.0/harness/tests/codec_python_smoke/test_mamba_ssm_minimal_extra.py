"""Extra coverage for ``lamquant_codec.models.mamba_ssm_minimal``.

Pins the public-API edges not already covered by tests/snn/test_mamba_ssm_minimal.py:

  - MambaSNN.classify_per_timestep returns int FSQ levels {2, 3, 5}
  - MambaSNN.param_size_kb returns positive float; scales with bits
  - HomeostaticThresholdAdapter mid-training default args
  - HAS_MAMBA_CUDA constant is the import-time fallback flag
  - SelectiveSSM forward at small d_state values
  - MambaSNN with use_subband=True (stride=1 path)

CPU-only; the CUDA branch (lines 191-200) is unreachable here by design
— that branch needs an actual mamba-ssm CUDA install.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from lamquant_codec.models.mamba_ssm_minimal import (
    HAS_MAMBA_CUDA,
    BidirectionalSSM,
    HomeostaticThresholdAdapter,
    MambaSNN,
    SelectiveSSM,
    _int3_quantize_decay,
    _int4_quantize_ste,
)


pytestmark = pytest.mark.l2


# ---------------------------------------------------------------------------
# Module-level constant
# ---------------------------------------------------------------------------


class TestHasMambaCudaConstant:
    def test_has_mamba_cuda_is_bool(self):
        assert isinstance(HAS_MAMBA_CUDA, bool)


# ---------------------------------------------------------------------------
# HomeostaticThresholdAdapter — already covered, add the documented defaults
# ---------------------------------------------------------------------------


class TestHomeostaticThresholdAdapterDefaults:
    def test_default_n_channels(self):
        h = HomeostaticThresholdAdapter()
        assert h.ema_rate.shape == (40,)
        assert h.threshold_adj.shape == (40,)
        assert h.target_rate == 0.1
        assert h.tau == 0.99

    def test_multiple_updates_accumulate(self):
        h = HomeostaticThresholdAdapter(n_channels=4)
        rates = np.array([0.5, 0.5, 0.5, 0.5])
        prev = h.ema_rate.copy()
        for _ in range(3):
            h.update(rates)
        # EMA must have moved away from zero
        assert (h.ema_rate > prev).all()


# ---------------------------------------------------------------------------
# _int4_quantize_ste edge cases
# ---------------------------------------------------------------------------


class TestInt4QuantizeSTEEdgeCases:
    def test_large_input_clamped(self):
        w = torch.tensor([1e6, -1e6, 0.0, 1.0])
        out = _int4_quantize_ste(w)
        assert torch.isfinite(out).all()


class TestInt3QuantizeDecayEdgeCases:
    def test_out_of_range_picks_nearest(self):
        # -10 is outside the level set → picks nearest level (-4)
        A_log = torch.tensor([-10.0, 10.0])
        out = _int3_quantize_decay(A_log)
        # Closest levels are -4 and 0.5
        assert out[0].item() == pytest.approx(-4.0)
        assert out[1].item() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# SelectiveSSM — push small d_state for the chunked-scan path
# ---------------------------------------------------------------------------


class TestSelectiveSSMSmall:
    def test_d_state_1(self):
        m = SelectiveSSM(d_model=4, d_state=1, d_conv=2, expand=1)
        m.eval()
        x = torch.randn(2, 8, 4)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (2, 8, 4)

    def test_short_sequence(self):
        m = SelectiveSSM(d_model=4, d_state=2, d_conv=2, expand=1)
        m.eval()
        x = torch.randn(1, 4, 4)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (1, 4, 4)

    def test_long_sequence_multiple_chunks(self):
        # T=80 > CHUNK=32 → multiple chunk iterations
        m = SelectiveSSM(d_model=4, d_state=2, d_conv=2, expand=1)
        m.eval()
        x = torch.randn(1, 80, 4)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (1, 80, 4)


# ---------------------------------------------------------------------------
# BidirectionalSSM — small smoke
# ---------------------------------------------------------------------------


class TestBidirectionalSSMSmall:
    def test_residual_preserved(self):
        m = BidirectionalSSM(d_model=4, d_state=2)
        m.eval()
        x = torch.randn(1, 8, 4)
        with torch.no_grad():
            y = m(x)
        # Residual connection means output must remain finite even
        # when SSM forward is freshly initialized.
        assert torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# MambaSNN — classify_per_timestep + param_size_kb
# ---------------------------------------------------------------------------


class TestMambaSNNClassify:
    def test_classify_per_timestep_shape(self):
        m = MambaSNN(in_channels=21, d_model=8, d_state=2, n_layers=1)
        m.eval()
        x = torch.randn(2, 21, 2500)
        levels = m.classify_per_timestep(x, target_T=79)
        # [B, target_T]
        assert levels.shape == (2, 79)

    def test_classify_per_timestep_values_in_allowed_set(self):
        m = MambaSNN(in_channels=21, d_model=8, d_state=2, n_layers=1)
        m.eval()
        x = torch.randn(1, 21, 2500)
        levels = m.classify_per_timestep(x, target_T=79)
        # FSQ levels are 2, 3, or 5
        unique = set(levels.unique().tolist())
        assert unique.issubset({2, 3, 5})

    def test_classify_skips_pooling_when_matched(self):
        """When logits already match target_T (no pooling), the no-pool branch
        is taken. Use a model with stride=1 so logits.shape[2] == T_in."""
        m = MambaSNN(in_channels=21, d_model=8, d_state=2, n_layers=1,
                      use_subband=True)
        m.eval()
        x = torch.randn(1, 21, 79)
        levels = m.classify_per_timestep(x, target_T=79)
        assert levels.shape == (1, 79)


class TestMambaSNNParamSize:
    def test_param_size_kb_default_bits(self):
        m = MambaSNN(in_channels=21, d_model=8, d_state=2, n_layers=1)
        kb = m.param_size_kb()
        assert kb > 0
        assert isinstance(kb, float)

    def test_param_size_kb_8_vs_16(self):
        """Doubling bit width must double the storage estimate."""
        m = MambaSNN(in_channels=21, d_model=8, d_state=2, n_layers=1)
        kb8 = m.param_size_kb(bits=8)
        kb16 = m.param_size_kb(bits=16)
        assert pytest.approx(kb16, rel=1e-6) == kb8 * 2

    def test_param_size_kb_int4(self):
        m = MambaSNN(in_channels=21, d_model=8, d_state=2, n_layers=1)
        kb4 = m.param_size_kb(bits=4)
        kb8 = m.param_size_kb(bits=8)
        # 4 bits = half the storage of 8 bits
        assert pytest.approx(kb4, rel=1e-6) == kb8 / 2


class TestMambaSNNUseSubband:
    def test_use_subband_keeps_stride_1(self):
        m = MambaSNN(in_channels=21, d_model=8, d_state=2, n_layers=1,
                      use_subband=True)
        assert m.stride == 1
        m.eval()
        x = torch.randn(1, 21, 79)
        with torch.no_grad():
            logits, rate = m(x)
        # stride=1 → no pooling → T_out matches T_in
        assert logits.shape[-1] == 79

    def test_raw_eeg_strides_by_8(self):
        m = MambaSNN(in_channels=21, d_model=8, d_state=2, n_layers=1)
        assert m.stride == 8
        m.eval()
        x = torch.randn(1, 21, 256)
        with torch.no_grad():
            logits, rate = m(x)
        # 256 / 8 = 32
        assert logits.shape[-1] == 32
