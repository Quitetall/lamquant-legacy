"""Extra coverage tests for ``lamquant_codec.models.blocks``.

Pinned contracts (shape / dtype / finite, not exact numerics):

  - TernaryConv1d / TernaryConvTranspose1d / INT8Conv1d forward shapes
    in both ``quantize=True`` and ``quantize=False`` branches
  - LSQ and SEQ ternary modes both round-trip
  - Binary mode + INT-N bridge mode in TernaryConv1d.forward
  - Alpha init / clamp_alpha / ensure_initialized lifecycle
  - WHT smoothing and the Hadamard cache warmup
  - _quantize_activation 2D fallback + remainder branch
  - BitShiftNorm / EfficientChannelAttention / CSPBlock / ReGLUBottleneck
  - TernaryFocalBlock / TernaryDWSepFocalBlock / TernaryUpsampleBlock

No real EEG used — these are pure linear-algebra layers and we just push
random tensors of the documented shape through.
"""
from __future__ import annotations

import math

import pytest
import torch

from lamquant_codec.models.blocks import (
    ACTIVATION_BITS,
    BitShiftNorm,
    CSPBlock,
    EfficientChannelAttention,
    INT8Conv1d,
    QUANTIZATION_MODE,
    QUANTIZER_TYPE,
    ReGLUBottleneck,
    TernaryConv1d,
    TernaryConvTranspose1d,
    TernaryDWSepFocalBlock,
    TernaryFocalBlock,
    TernaryUpsampleBlock,
    _ActivationQuantFunction,
    _build_hadamard_32_cpu,
    _get_hadamard_32,
    _lsq_binary,
    _lsq_ternary,
    _quantize_activation,
    _seq_ternary,
    _wht_smooth,
    set_activation_bits,
    set_quantization_mode,
    set_quantizer_type,
    warmup_hadamard_cache,
)


pytestmark = pytest.mark.l2


# ---------------------------------------------------------------------------
# Module-level switches: cover the setter branches + restore defaults
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_module_state():
    """Reset module-level switches after each test so order doesn't matter."""
    import lamquant_codec.models.blocks as B
    saved = (B.QUANTIZATION_MODE, B.QUANTIZER_TYPE, B.ACTIVATION_BITS)
    yield
    B.QUANTIZATION_MODE = saved[0]
    B.QUANTIZER_TYPE = saved[1]
    B.ACTIVATION_BITS = saved[2]


class TestModuleSwitches:
    def test_set_quantization_mode_ternary(self):
        set_quantization_mode("ternary")
        import lamquant_codec.models.blocks as B
        assert B.QUANTIZATION_MODE == "ternary"

    def test_set_quantization_mode_binary(self):
        set_quantization_mode("binary")
        import lamquant_codec.models.blocks as B
        assert B.QUANTIZATION_MODE == "binary"

    def test_set_quantization_mode_invalid_rejected(self):
        with pytest.raises(AssertionError):
            set_quantization_mode("quaternary")

    def test_set_quantizer_type_lsq(self, capsys):
        set_quantizer_type("lsq")
        import lamquant_codec.models.blocks as B
        assert B.QUANTIZER_TYPE == "lsq"

    def test_set_quantizer_type_seq(self, capsys):
        set_quantizer_type("seq")
        import lamquant_codec.models.blocks as B
        assert B.QUANTIZER_TYPE == "seq"

    def test_set_quantizer_type_invalid(self):
        with pytest.raises(AssertionError):
            set_quantizer_type("seqq")

    def test_set_activation_bits_8(self):
        set_activation_bits(8)
        import lamquant_codec.models.blocks as B
        assert B.ACTIVATION_BITS == 8

    def test_set_activation_bits_16(self):
        set_activation_bits(16)
        import lamquant_codec.models.blocks as B
        assert B.ACTIVATION_BITS == 16

    def test_set_activation_bits_invalid(self):
        with pytest.raises(AssertionError):
            set_activation_bits(4)


# ---------------------------------------------------------------------------
# LSQ / SEQ / Binary ternary quantizer autograd functions
# ---------------------------------------------------------------------------


class TestLSQTernary:
    def test_forward_shape_preserved(self):
        w = torch.randn(8, 4, 3, requires_grad=True)
        alpha = torch.ones(8, 1, 1) * 0.1
        out = _lsq_ternary(w, alpha, grad_scale=0.1, deadzone_tau=0.1)
        assert out.shape == w.shape
        assert torch.isfinite(out).all()

    def test_backward_via_ste(self):
        w = torch.randn(8, 4, 3, requires_grad=True)
        alpha = (torch.ones(8, 1, 1) * 0.1).requires_grad_(True)
        out = _lsq_ternary(w, alpha, grad_scale=0.1, deadzone_tau=0.0)
        out.sum().backward()
        assert w.grad is not None
        assert alpha.grad is not None

    def test_zero_alpha_no_divbyzero(self):
        w = torch.randn(8, 4, 3)
        alpha = torch.zeros(8, 1, 1)
        out = _lsq_ternary(w, alpha, grad_scale=0.1, deadzone_tau=0.1)
        assert torch.isfinite(out).all()


class TestSEQTernary:
    def test_forward_shape_preserved(self):
        w = torch.randn(8, 4, 3, requires_grad=True)
        alpha = torch.ones(8, 1, 1) * 0.1
        out = _seq_ternary(w, alpha, grad_scale=0.1)
        assert out.shape == w.shape
        assert torch.isfinite(out).all()

    def test_backward(self):
        w = torch.randn(8, 4, 3, requires_grad=True)
        alpha = (torch.ones(8, 1, 1) * 0.1).requires_grad_(True)
        out = _seq_ternary(w, alpha, grad_scale=0.1)
        out.sum().backward()
        assert w.grad is not None
        assert alpha.grad is not None


class TestLSQBinary:
    def test_forward_shape_and_sign(self):
        w = torch.randn(8, 4, 3, requires_grad=True)
        alpha = torch.ones(8, 1, 1) * 0.1
        out = _lsq_binary(w, alpha, grad_scale=0.1)
        assert out.shape == w.shape
        # Binary should only produce ±alpha
        # (alpha may differ per channel)
        for c in range(out.shape[0]):
            unique = out[c].unique()
            assert len(unique) <= 2

    def test_backward(self):
        w = torch.randn(8, 4, 3, requires_grad=True)
        alpha = (torch.ones(8, 1, 1) * 0.1).requires_grad_(True)
        out = _lsq_binary(w, alpha, grad_scale=0.1)
        out.sum().backward()
        assert w.grad is not None

    def test_zero_input_handled(self):
        """Weight==0 path should not hit a div-by-zero or sign-of-zero issue."""
        w = torch.zeros(4, 4, 3, requires_grad=True)
        alpha = torch.ones(4, 1, 1) * 0.1
        out = _lsq_binary(w, alpha, grad_scale=0.1)
        # Zero weights -> sign(0) tie-break to +1 in the function
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Hadamard cache
# ---------------------------------------------------------------------------


class TestHadamardCache:
    def test_build_32x32_cpu(self):
        H = _build_hadamard_32_cpu()
        assert H.shape == (32, 32)
        # Normalized Hadamard: H @ H.T = I (within fp32 noise)
        identity = H @ H.T
        assert torch.allclose(identity, torch.eye(32), atol=1e-5)

    def test_get_hadamard_lazy_init(self):
        """First call to _get_hadamard_32 for an unseen (device, dtype) lazily
        populates the cache."""
        device = torch.device("cpu")
        H = _get_hadamard_32(device, torch.float32)
        assert H.shape == (32, 32)
        # Cached for next call
        H2 = _get_hadamard_32(device, torch.float32)
        assert H is H2 or torch.equal(H, H2)

    def test_warmup_cpu_string_arg(self, capsys):
        # Accept str device argument
        warmup_hadamard_cache("cpu")
        # Returns nothing meaningful — the side effect is cache population
        H = _get_hadamard_32(torch.device("cpu"), torch.float32)
        assert H is not None


# ---------------------------------------------------------------------------
# WHT smoothing branch coverage
# ---------------------------------------------------------------------------


class TestWHTSmooth:
    def test_power_of_two_length(self):
        x = torch.randn(2, 4, 64)
        out = _wht_smooth(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_non_power_of_two_pads(self):
        # T=50 → pads to 64 then trims
        x = torch.randn(1, 2, 50)
        out = _wht_smooth(x)
        assert out.shape == (1, 2, 50)

    def test_very_long_sequence_skipped(self):
        # T > 4096 short-circuits and returns input
        x = torch.randn(1, 2, 5000)
        out = _wht_smooth(x)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# _quantize_activation: 2D fallback + remainder + disabled path
# ---------------------------------------------------------------------------


class TestQuantizeActivation:
    def test_disabled_returns_input(self):
        x = torch.randn(2, 4, 8)
        out = _quantize_activation(x, enabled=False)
        # The function returns the input unchanged when disabled.
        assert out is x

    def test_3d_exact_block_multiple(self):
        # T=64 → 2 full blocks of 32, no remainder
        x = torch.randn(2, 4, 64)
        out = _quantize_activation(x, enabled=True)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_3d_with_remainder(self):
        # T=50 → 1 block of 32 + 18-sample remainder
        x = torch.randn(2, 4, 50)
        out = _quantize_activation(x, enabled=True)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_3d_less_than_one_block(self):
        # T=20 → 0 blocks + 20-sample remainder
        x = torch.randn(2, 4, 20)
        out = _quantize_activation(x, enabled=True)
        assert out.shape == x.shape

    def test_2d_fallback(self):
        # 2D input → falls through to scalar-scale path
        x = torch.randn(8, 16)
        out = _quantize_activation(x, enabled=True)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_3d_with_supplied_hadamard(self):
        H = _build_hadamard_32_cpu()
        x = torch.randn(2, 4, 64)
        out = _quantize_activation(x, enabled=True, hadamard=H)
        assert out.shape == x.shape

    def test_3d_supplied_hadamard_dtype_cast(self):
        # Pass in fp64 Hadamard against fp32 input — must cast safely
        H = _build_hadamard_32_cpu().double()
        x = torch.randn(2, 4, 64, dtype=torch.float32)
        out = _quantize_activation(x, enabled=True, hadamard=H)
        assert out.dtype == torch.float32


class TestActivationQuantFunction:
    def test_forward_clamps_to_range(self):
        x = torch.tensor([[[1000.0, -1000.0, 0.5]]])
        scale = torch.tensor(0.01)
        out = _ActivationQuantFunction.apply(x, scale, 8)
        assert torch.isfinite(out).all()

    def test_backward_straight_through(self):
        x = torch.randn(2, 4, requires_grad=True)
        scale = torch.tensor(0.1)
        out = _ActivationQuantFunction.apply(x, scale, 8)
        out.sum().backward()
        assert x.grad is not None
        # STE: gradient passes through
        assert torch.allclose(x.grad, torch.ones_like(x))


# ---------------------------------------------------------------------------
# TernaryConv1d — all forward branches
# ---------------------------------------------------------------------------


class TestTernaryConv1d:
    def test_forward_shape_quantized(self):
        conv = TernaryConv1d(in_ch=8, out_ch=16, kernel_size=3, stride=1)
        x = torch.randn(2, 8, 32)
        y = conv(x, quantize=True)
        assert y.shape == (2, 16, 32)
        assert torch.isfinite(y).all()

    def test_forward_shape_unquantized(self):
        conv = TernaryConv1d(in_ch=8, out_ch=16, kernel_size=3, stride=1)
        x = torch.randn(2, 8, 32)
        y = conv(x, quantize=False)
        # unquantized path uses the parent nn.Conv1d.forward directly
        assert y.shape == (2, 16, 32)

    def test_forward_strided(self):
        conv = TernaryConv1d(in_ch=8, out_ch=16, kernel_size=3, stride=2)
        x = torch.randn(2, 8, 32)
        y = conv(x, quantize=True)
        # stride 2 halves T (with padding=k//2 it's exactly T/stride for even T)
        assert y.shape[2] == 16

    def test_init_alpha_runs(self):
        conv = TernaryConv1d(in_ch=8, out_ch=16, kernel_size=3)
        # Flag starts False; ensure_initialized flips it.
        assert not conv._alpha_init_flag.item()
        conv.ensure_initialized()
        assert conv._alpha_init_flag.item()

    def test_ensure_initialized_idempotent(self):
        conv = TernaryConv1d(in_ch=8, out_ch=16, kernel_size=3)
        conv.ensure_initialized()
        alpha_before = conv.lsq_alpha.data.clone()
        conv.ensure_initialized()  # second call should not change alpha
        assert torch.equal(conv.lsq_alpha.data, alpha_before)

    def test_clamp_alpha(self):
        conv = TernaryConv1d(in_ch=8, out_ch=16, kernel_size=3)
        # Force alpha to an extreme value
        conv.lsq_alpha.data.fill_(1e6)
        conv.clamp_alpha()
        # After clamping, alpha must be within [0.5*std, 2.0*std]
        with torch.no_grad():
            w_std = conv.weight.std(dim=(1, 2), keepdim=True).clamp(min=1e-4)
            assert (conv.lsq_alpha.data <= 2.0 * w_std + 1e-5).all()

    def test_set_bits_bridge(self):
        """INT-N bridge path (set_bits(8)) takes the bridge branch."""
        conv = TernaryConv1d(in_ch=8, out_ch=16, kernel_size=3)
        conv.set_bits(8)
        x = torch.randn(2, 8, 32)
        y = conv(x, quantize=True)
        assert y.shape == (2, 16, 32)

    def test_forward_binary_mode(self):
        set_quantization_mode("binary")
        conv = TernaryConv1d(in_ch=8, out_ch=16, kernel_size=3)
        x = torch.randn(2, 8, 32)
        y = conv(x, quantize=True)
        assert y.shape == (2, 16, 32)

    def test_forward_seq_mode(self, capsys):
        set_quantizer_type("seq")
        conv = TernaryConv1d(in_ch=8, out_ch=16, kernel_size=3)
        x = torch.randn(2, 8, 32)
        y = conv(x, quantize=True)
        assert y.shape == (2, 16, 32)

    def test_train_vs_eval_deadzone(self):
        conv = TernaryConv1d(in_ch=8, out_ch=16, kernel_size=3)
        conv.train()
        x = torch.randn(2, 8, 32)
        y_train = conv(x, quantize=True)
        conv.eval()
        y_eval = conv(x, quantize=True)
        # Both branches should produce a valid output
        assert y_train.shape == y_eval.shape

    def test_grouped(self):
        conv = TernaryConv1d(in_ch=16, out_ch=16, kernel_size=3, groups=16)
        x = torch.randn(2, 16, 32)
        y = conv(x, quantize=True)
        assert y.shape == (2, 16, 32)


# ---------------------------------------------------------------------------
# INT8Conv1d
# ---------------------------------------------------------------------------


class TestINT8Conv1d:
    def test_forward_shape(self):
        conv = INT8Conv1d(in_ch=8, out_ch=4, kernel_size=1, bias=True)
        x = torch.randn(2, 8, 16)
        y = conv(x, quantize=True)
        assert y.shape == (2, 4, 16)
        assert torch.isfinite(y).all()

    def test_forward_unquantized(self):
        conv = INT8Conv1d(in_ch=8, out_ch=4, kernel_size=1, bias=True)
        x = torch.randn(2, 8, 16)
        y = conv(x, quantize=False)
        assert y.shape == (2, 4, 16)

    def test_ensure_initialized(self):
        conv = INT8Conv1d(in_ch=8, out_ch=4, kernel_size=1)
        assert not conv._scale_init_flag.item()
        conv.ensure_initialized()
        assert conv._scale_init_flag.item()
        # Idempotent
        conv.ensure_initialized()
        assert conv._scale_init_flag.item()


# ---------------------------------------------------------------------------
# TernaryConvTranspose1d
# ---------------------------------------------------------------------------


class TestTernaryConvTranspose1d:
    def test_forward_upsample(self):
        conv = TernaryConvTranspose1d(in_ch=4, out_ch=4, kernel_size=3, stride=2)
        x = torch.randn(2, 4, 16)
        y = conv(x, quantize=True)
        assert y.dim() == 3
        assert y.shape[0] == 2
        assert y.shape[1] == 4
        assert torch.isfinite(y).all()

    def test_forward_unquantized(self):
        conv = TernaryConvTranspose1d(in_ch=4, out_ch=4, kernel_size=3, stride=2)
        x = torch.randn(2, 4, 16)
        y = conv(x, quantize=False)
        assert y.shape[0] == 2

    def test_ensure_initialized(self):
        conv = TernaryConvTranspose1d(in_ch=4, out_ch=4, kernel_size=3, stride=2)
        assert not conv._alpha_init_flag.item()
        conv.ensure_initialized()
        assert conv._alpha_init_flag.item()

    def test_clamp_alpha(self):
        conv = TernaryConvTranspose1d(in_ch=4, out_ch=4, kernel_size=3, stride=2)
        conv.lsq_alpha.data.fill_(1e6)
        conv.clamp_alpha()
        with torch.no_grad():
            w_std = conv.weight.std(dim=(1, 2), keepdim=True).clamp(min=1e-4)
            assert (conv.lsq_alpha.data <= 2.0 * w_std + 1e-5).all()

    def test_binary_mode(self):
        set_quantization_mode("binary")
        conv = TernaryConvTranspose1d(in_ch=4, out_ch=4, kernel_size=3, stride=2)
        x = torch.randn(2, 4, 16)
        y = conv(x, quantize=True)
        assert y.shape[0] == 2

    def test_seq_mode(self, capsys):
        set_quantizer_type("seq")
        conv = TernaryConvTranspose1d(in_ch=4, out_ch=4, kernel_size=3, stride=2)
        x = torch.randn(2, 4, 16)
        y = conv(x, quantize=True)
        assert y.shape[0] == 2


# ---------------------------------------------------------------------------
# Composite blocks
# ---------------------------------------------------------------------------


class TestBitShiftNorm:
    def test_forward_shape(self):
        bsn = BitShiftNorm(channels=8)
        x = torch.randn(2, 8, 32)
        y = bsn(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()


class TestEfficientChannelAttention:
    def test_forward_shape(self):
        eca = EfficientChannelAttention(channels=16, kernel_size=3)
        x = torch.randn(2, 16, 32)
        y = eca(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()


class TestCSPBlock:
    def test_forward_shape_stride1(self):
        # channels must be even (half-split)
        blk = CSPBlock(channels=16, kernel_size=3, stride=1)
        x = torch.randn(2, 16, 32)
        y = blk(x, quantize=True)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()

    def test_forward_shape_stride2(self):
        blk = CSPBlock(channels=16, kernel_size=3, stride=2)
        x = torch.randn(2, 16, 32)
        y = blk(x, quantize=True)
        # stride > 1 forces the TernaryConv1d shortcut branch
        assert y.dim() == 3
        assert y.shape[0] == 2
        assert y.shape[1] == 16


class TestReGLUBottleneck:
    def test_forward_shape(self):
        bn = ReGLUBottleneck(in_ch=8, hidden_ch=16, out_ch=4)
        x = torch.randn(2, 8, 32)
        y = bn(x, quantize=True)
        assert y.shape == (2, 4, 32)
        assert torch.isfinite(y).all()


class TestTernaryFocalBlock:
    def test_same_channels_identity_shortcut(self):
        blk = TernaryFocalBlock(in_ch=16, out_ch=16, kernel_size=3, stride=1)
        x = torch.randn(2, 16, 32)
        y = blk(x, quantize=True)
        assert y.shape == x.shape

    def test_channel_change_uses_ternary_shortcut(self):
        blk = TernaryFocalBlock(in_ch=8, out_ch=16, kernel_size=3, stride=1)
        x = torch.randn(2, 8, 32)
        y = blk(x, quantize=True)
        assert y.shape == (2, 16, 32)

    def test_stride2_uses_ternary_shortcut(self):
        blk = TernaryFocalBlock(in_ch=16, out_ch=16, kernel_size=3, stride=2)
        x = torch.randn(2, 16, 32)
        y = blk(x, quantize=True)
        assert y.shape[1] == 16
        assert y.shape[2] == 16

    def test_unquantized(self):
        blk = TernaryFocalBlock(in_ch=16, out_ch=16, kernel_size=3, stride=1)
        x = torch.randn(2, 16, 32)
        y = blk(x, quantize=False)
        assert y.shape == x.shape

    def test_non_groupnorm_compatible_channels(self):
        # out_ch=3 is not divisible by 4 → falls into the GroupNorm(1, 3) branch
        blk = TernaryFocalBlock(in_ch=8, out_ch=3, kernel_size=3, stride=1)
        x = torch.randn(2, 8, 32)
        y = blk(x, quantize=True)
        assert y.shape == (2, 3, 32)


class TestTernaryDWSepFocalBlock:
    def test_same_channels_identity_shortcut(self):
        blk = TernaryDWSepFocalBlock(in_ch=16, out_ch=16, kernel_size=5, stride=1)
        x = torch.randn(2, 16, 32)
        y = blk(x, quantize=True)
        assert y.shape == x.shape

    def test_channel_change_uses_ternary_shortcut(self):
        blk = TernaryDWSepFocalBlock(in_ch=8, out_ch=16, kernel_size=3, stride=1)
        x = torch.randn(2, 8, 32)
        y = blk(x, quantize=True)
        assert y.shape == (2, 16, 32)

    def test_stride2(self):
        blk = TernaryDWSepFocalBlock(in_ch=16, out_ch=16, kernel_size=3, stride=2)
        x = torch.randn(2, 16, 32)
        y = blk(x, quantize=True)
        assert y.shape[1] == 16
        assert y.shape[2] == 16

    def test_non_groupnorm8_compatible_channels(self):
        # out_ch=4 not divisible by 8 → falls to GroupNorm(4, 4) branch
        # in_ch == out_ch so the implicit identity shortcut is used.
        blk = TernaryDWSepFocalBlock(in_ch=4, out_ch=4, kernel_size=3, stride=1)
        x = torch.randn(2, 4, 32)
        y = blk(x, quantize=True)
        assert y.shape == x.shape


class TestTernaryUpsampleBlock:
    def test_same_channels_identity_shortcut_stride1(self):
        blk = TernaryUpsampleBlock(in_ch=16, out_ch=16, kernel_size=3, stride=1)
        x = torch.randn(2, 16, 32)
        y = blk(x, quantize=True)
        assert y.shape == x.shape

    def test_stride2_upsample(self):
        # Production usage: same in/out channels with stride=2 upsampling.
        blk = TernaryUpsampleBlock(in_ch=16, out_ch=16, kernel_size=3, stride=2)
        x = torch.randn(2, 16, 16)
        y = blk(x, quantize=True)
        assert y.shape[1] == 16
        assert y.shape[2] > x.shape[2]  # T expanded

    def test_unquantized(self):
        blk = TernaryUpsampleBlock(in_ch=16, out_ch=16, kernel_size=3, stride=2)
        x = torch.randn(2, 16, 16)
        y = blk(x, quantize=False)
        assert y.shape[1] == 16
