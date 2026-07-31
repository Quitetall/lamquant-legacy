"""Extra coverage tests for ``lamquant_codec.models.encoder``.

Pinned contracts (shape / dtype / finite, not exact numerics):

  - TernaryMobileNetV5 full-band encoder: [B, 21, 2500] → reconstructs same length
  - TernaryMobileNetV5_Subband: configurable n_blocks + kernel_sizes
  - encode_stage1/2/3, encode, decode, forward all produce documented shapes
  - _ZeroPadShortcut pads channels and strides correctly
  - _get_rotation produces an orthogonal matrix
  - _cdf_forward / _cdf_inverse round-trip within numerical noise
  - from_checkpoint loads a freshly-instantiated state_dict (V1 + V2 paths)
  - ensure_initialized + _refresh_hadamard_refs flip the corresponding flags
  - TernaryMobileNetV5_Subband_V2: width 144, 4 DW-sep blocks

No real EEG used — these are pure model definitions and we just push
torch.randn through with the documented [B, C, T] shape.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from lamquant_codec.models.encoder import (
    TernaryMobileNetV5,
    TernaryMobileNetV5_Subband,
    TernaryMobileNetV5_Subband_V2,
    _ZeroPadShortcut,
)
from lamquant_codec.models.blocks import (
    TernaryConv1d,
    TernaryConvTranspose1d,
    INT8Conv1d,
    TernaryFocalBlock,
    TernaryUpsampleBlock,
)


pytestmark = pytest.mark.l2


# ---------------------------------------------------------------------------
# _ZeroPadShortcut
# ---------------------------------------------------------------------------


class TestZeroPadShortcut:
    def test_pads_channels(self):
        sc = _ZeroPadShortcut(in_ch=21, out_ch=128, stride=1)
        x = torch.randn(2, 21, 313)
        y = sc(x)
        assert y.shape == (2, 128, 313)
        # Original channels preserved exactly
        assert torch.equal(y[:, :21], x)
        # Padded channels are zero
        assert torch.all(y[:, 21:] == 0)

    def test_strided_downsample(self):
        sc = _ZeroPadShortcut(in_ch=21, out_ch=128, stride=2)
        x = torch.randn(2, 21, 313)
        y = sc(x)
        # stride=2 → ceil(313/2) = 157
        assert y.shape == (2, 128, 157)
        # Take every other sample
        assert torch.equal(y[:, :21], x[:, :, ::2])

    def test_no_pad_when_dims_match(self):
        sc = _ZeroPadShortcut(in_ch=16, out_ch=16, stride=1)
        x = torch.randn(2, 16, 32)
        y = sc(x)
        assert y.shape == x.shape
        assert torch.equal(y, x)


# ---------------------------------------------------------------------------
# TernaryMobileNetV5 (full-band Gen 7.0 encoder)
# ---------------------------------------------------------------------------


class TestTernaryMobileNetV5:
    def test_forward_shape_quantized(self):
        torch.manual_seed(0)
        m = TernaryMobileNetV5(in_ch=21, latent_dim=32)
        m.eval()
        x = torch.randn(1, 21, 2500)
        with torch.no_grad():
            y = m(x, quantize=True)
        assert y.shape == (1, 21, 2500)
        assert torch.isfinite(y).all()

    def test_forward_shape_unquantized(self):
        torch.manual_seed(0)
        m = TernaryMobileNetV5(in_ch=21, latent_dim=32)
        m.eval()
        x = torch.randn(1, 21, 2500)
        with torch.no_grad():
            y = m(x, quantize=False)
        assert y.shape == (1, 21, 2500)

    def test_encode_only_latent_shape(self):
        torch.manual_seed(0)
        m = TernaryMobileNetV5(in_ch=21, latent_dim=32)
        m.eval()
        x = torch.randn(1, 21, 2500)
        with torch.no_grad():
            lat = m.encode(x, quantize=True)
        # Latent: stride 8 → 2500/8 ≈ 312
        assert lat.shape[0] == 1
        assert lat.shape[1] == 32
        assert lat.shape[2] >= 312
        assert lat.shape[2] <= 314

    def test_encode_unquantized_path(self):
        m = TernaryMobileNetV5(in_ch=21, latent_dim=32)
        m.eval()
        x = torch.randn(1, 21, 2500)
        with torch.no_grad():
            lat = m.encode(x, quantize=False)
        assert lat.shape[1] == 32

    def test_smaller_in_channels(self):
        # Custom in_ch — pin shape contract not specific channel count
        m = TernaryMobileNetV5(in_ch=8, latent_dim=16)
        m.eval()
        x = torch.randn(2, 8, 2500)
        with torch.no_grad():
            y = m(x, quantize=True)
        assert y.shape == (2, 8, 2500)


# ---------------------------------------------------------------------------
# TernaryMobileNetV5_Subband (Gen 7.1)
# ---------------------------------------------------------------------------


class TestTernaryMobileNetV5Subband:
    def test_default_construction(self):
        m = TernaryMobileNetV5_Subband()
        assert m.n_blocks == 3
        assert m.kernel_sizes == (3, 5, 7)
        # bneck dims
        assert isinstance(m.bneck_v, INT8Conv1d)
        assert isinstance(m.bneck_g, TernaryConv1d)
        # CDF buffer shape
        assert m.cdf_breakpoints.shape == (32, 32)

    def test_kernel_size_count_assertion(self):
        with pytest.raises(AssertionError, match="kernel sizes"):
            TernaryMobileNetV5_Subband(n_blocks=4, kernel_sizes=(3, 5, 7))

    def test_min_block_count_assertion(self):
        with pytest.raises(AssertionError, match="at least 2 blocks"):
            TernaryMobileNetV5_Subband(n_blocks=1, kernel_sizes=(3,))

    def test_forward_default_shape(self):
        torch.manual_seed(0)
        m = TernaryMobileNetV5_Subband()
        m.eval()
        x = torch.randn(1, 21, 313)
        with torch.no_grad():
            y = m(x, quantize=True)
        # Output reconstructs to original time length
        assert y.shape == (1, 21, 313)
        assert torch.isfinite(y).all()

    def test_forward_unquantized(self):
        torch.manual_seed(0)
        m = TernaryMobileNetV5_Subband()
        m.eval()
        x = torch.randn(1, 21, 313)
        with torch.no_grad():
            y = m(x, quantize=False)
        assert y.shape == (1, 21, 313)

    def test_encode_only_latent_shape(self):
        torch.manual_seed(0)
        m = TernaryMobileNetV5_Subband()
        m.eval()
        x = torch.randn(1, 21, 313)
        with torch.no_grad():
            lat = m.encode(x, quantize=True)
        assert lat.shape == (1, 32, 79)
        assert torch.isfinite(lat).all()

    def test_encode_stages(self):
        torch.manual_seed(0)
        m = TernaryMobileNetV5_Subband()
        m.eval()
        x = torch.randn(1, 21, 313)
        with torch.no_grad():
            s1 = m.encode_stage1(x, quantize=True)
            s2 = m.encode_stage2(s1, quantize=True)
            s3 = m.encode_stage3(s2, quantize=True)
        # encode_stage3 produces latent shape
        assert s3.shape == (1, 32, 79)
        # Each stage's output is finite
        assert torch.isfinite(s1).all()
        assert torch.isfinite(s2).all()
        assert torch.isfinite(s3).all()

    def test_encode_stage3_training_branch(self):
        """In training mode, encode_stage3 takes the dithered-FSQ branch."""
        torch.manual_seed(0)
        m = TernaryMobileNetV5_Subband()
        m.train()
        x = torch.randn(1, 21, 313)
        s1 = m.encode_stage1(x, quantize=True)
        s2 = m.encode_stage2(s1, quantize=True)
        s3 = m.encode_stage3(s2, quantize=True)
        assert s3.shape == (1, 32, 79)
        assert torch.isfinite(s3).all()

    def test_encode_training_branch(self):
        """In training mode encode() takes the StableCodec dither path."""
        torch.manual_seed(0)
        m = TernaryMobileNetV5_Subband()
        m.train()
        x = torch.randn(1, 21, 313)
        lat = m.encode(x, quantize=True)
        assert lat.shape == (1, 32, 79)
        assert torch.isfinite(lat).all()

    def test_decode_shape(self):
        torch.manual_seed(0)
        m = TernaryMobileNetV5_Subband()
        m.eval()
        lat = torch.randn(1, 32, 79).clamp(-0.99, 0.99)
        with torch.no_grad():
            out = m.decode(lat, target_len=313, quantize=True)
        assert out.shape == (1, 21, 313)
        assert torch.isfinite(out).all()

    def test_decode_unquantized(self):
        m = TernaryMobileNetV5_Subband()
        m.eval()
        lat = torch.randn(1, 32, 79).clamp(-0.99, 0.99)
        with torch.no_grad():
            out = m.decode(lat, target_len=313, quantize=False)
        assert out.shape == (1, 21, 313)

    def test_get_rotation_orthogonal(self):
        m = TernaryMobileNetV5_Subband()
        Q = m._get_rotation()
        assert Q.shape == (32, 32)
        # Q is orthogonal: Q @ Q.T = I
        identity = Q @ Q.T
        assert torch.allclose(identity, torch.eye(32), atol=1e-4)

    def test_cdf_forward_inverse_roundtrip(self):
        torch.manual_seed(42)
        m = TernaryMobileNetV5_Subband()
        # latent values inside the breakpoint range
        latent = torch.randn(2, 32, 8).clamp(-2.5, 2.5)
        uniform = m._cdf_forward(latent)
        # Maps to [-1, 1]
        assert uniform.max().item() <= 1.0 + 1e-5
        assert uniform.min().item() >= -1.0 - 1e-5
        # Round-trip via _cdf_inverse approximately recovers latent
        recovered = m._cdf_inverse(uniform)
        # Quantile interpolation: within one breakpoint spacing
        assert torch.allclose(recovered, latent, atol=0.5)

    def test_cdf_forward_clamps_outliers(self):
        m = TernaryMobileNetV5_Subband()
        latent = torch.tensor([[[100.0, -100.0, 0.0]]] * 32).reshape(1, 32, 3)
        uniform = m._cdf_forward(latent)
        # Saturated to [-1, 1]
        assert uniform.max().item() <= 1.0 + 1e-5
        assert uniform.min().item() >= -1.0 - 1e-5

    def test_cdf_inverse_full_range(self):
        m = TernaryMobileNetV5_Subband()
        # Endpoints of uniform [-1, 1] should map to breakpoint edges
        u = torch.linspace(-1, 1, 5).reshape(1, 1, 5).expand(1, 32, 5).contiguous()
        latent = m._cdf_inverse(u)
        assert latent.shape == u.shape
        assert torch.isfinite(latent).all()

    def test_ensure_initialized(self):
        m = TernaryMobileNetV5_Subband()
        m.ensure_initialized()
        # All TernaryConv1d / INT8Conv1d submodules should have their flags set
        for sub in m.modules():
            if isinstance(sub, (TernaryConv1d, INT8Conv1d)):
                if hasattr(sub, "_alpha_init_flag"):
                    assert sub._alpha_init_flag.item()
                if hasattr(sub, "_scale_init_flag"):
                    assert sub._scale_init_flag.item()

    def test_refresh_hadamard_refs(self):
        m = TernaryMobileNetV5_Subband()
        # All target submodules should pick up the Hadamard reference
        for sub in m.modules():
            if isinstance(sub, (TernaryConv1d, INT8Conv1d, TernaryConvTranspose1d,
                                  TernaryFocalBlock, TernaryUpsampleBlock)):
                assert sub._hadamard_ref is m._hadamard_32

    def test_apply_re_refreshes_hadamard(self):
        """After .to(device) / .float() the Hadamard refs must be rebound."""
        m = TernaryMobileNetV5_Subband()
        # cpu→cpu is enough to exercise _apply override
        m = m.cpu()
        for sub in m.modules():
            if isinstance(sub, (TernaryConv1d, INT8Conv1d)):
                assert sub._hadamard_ref is m._hadamard_32


class TestTernaryMobileNetV5SubbandConfigurable:
    def test_n_blocks_2(self):
        m = TernaryMobileNetV5_Subband(n_blocks=2, kernel_sizes=(3, 5))
        m.eval()
        assert m.n_blocks == 2
        # No focal_mid (just focal1 + focal_last)
        assert len(m.focal_mid) == 0
        x = torch.randn(1, 21, 313)
        with torch.no_grad():
            y = m(x, quantize=True)
        assert y.shape == (1, 21, 313)

    def test_n_blocks_4(self):
        m = TernaryMobileNetV5_Subband(n_blocks=4, kernel_sizes=(3, 5, 5, 7))
        m.eval()
        assert m.n_blocks == 4
        # 2 middle blocks
        assert len(m.focal_mid) == 2
        x = torch.randn(1, 21, 313)
        with torch.no_grad():
            y = m(x, quantize=True)
        assert y.shape == (1, 21, 313)

    def test_custom_width(self):
        m = TernaryMobileNetV5_Subband(width=64)
        m.eval()
        x = torch.randn(1, 21, 313)
        with torch.no_grad():
            lat = m.encode(x, quantize=True)
        assert lat.shape[1] == 32

    def test_custom_cdf_entries(self):
        m = TernaryMobileNetV5_Subband(cdf_entries=16)
        assert m.cdf_breakpoints.shape == (32, 16)


class TestFromCheckpoint:
    def test_from_checkpoint_state_dict(self, tmp_path):
        """from_checkpoint loads a freshly-saved state dict (V1 default)."""
        src = TernaryMobileNetV5_Subband(width=128)
        sd_path = tmp_path / "v1.ckpt"
        torch.save(src.state_dict(), str(sd_path))
        loaded = TernaryMobileNetV5_Subband.from_checkpoint(
            str(sd_path), device="cpu"
        )
        # Width detected from focal_mid.0 conv weight shape
        assert loaded.focal_mid[0].conv.weight.shape[0] == 128

    def test_from_checkpoint_wrapped_in_model_state_dict(self, tmp_path):
        """from_checkpoint unwraps {'model_state_dict': sd}."""
        src = TernaryMobileNetV5_Subband(width=128)
        sd_path = tmp_path / "wrapped.ckpt"
        torch.save({"model_state_dict": src.state_dict(),
                    "step": 0, "loss": 0.1}, str(sd_path))
        loaded = TernaryMobileNetV5_Subband.from_checkpoint(
            str(sd_path), device="cpu"
        )
        assert isinstance(loaded, TernaryMobileNetV5_Subband)

    def test_from_checkpoint_v2_via_dwsep_marker(self, tmp_path):
        """When focal2.dw.weight is present, from_checkpoint constructs V2."""
        src = TernaryMobileNetV5_Subband_V2(width=144)
        sd_path = tmp_path / "v2.ckpt"
        torch.save(src.state_dict(), str(sd_path))
        loaded = TernaryMobileNetV5_Subband.from_checkpoint(
            str(sd_path), device="cpu"
        )
        assert isinstance(loaded, TernaryMobileNetV5_Subband_V2)

    def test_from_checkpoint_unsafe_load_fallback(self, tmp_path, capsys):
        """If weights_only=True fails, from_checkpoint falls back to
        weights_only=False. We trigger that by saving a dict that
        weights_only mode refuses to load (functions, numpy structured types,
        etc.). Here we use a top-level numpy structured-type that the safe
        loader rejects but the unsafe one accepts."""
        src = TernaryMobileNetV5_Subband(width=128)
        sd = src.state_dict()
        # The straightforward path: include a slice object (function-like).
        # weights_only=True only accepts tensors + a small allowlist; a
        # slice falls outside that allowlist and triggers the fallback.
        wrapped = {"model_state_dict": sd, "config_slice": slice(0, 10)}
        sd_path = tmp_path / "unsafe.ckpt"
        torch.save(wrapped, str(sd_path))
        # The fallback must succeed (line 466 path runs)
        loaded = TernaryMobileNetV5_Subband.from_checkpoint(
            str(sd_path), device="cpu"
        )
        assert isinstance(loaded, TernaryMobileNetV5_Subband)

    def test_from_checkpoint_legacy_focal2_conv_weight(self, tmp_path, capsys):
        """An old checkpoint with focal2.conv.weight (n_blocks==3 was the only
        layout) infers width from focal2.conv.weight and n_blocks=3."""
        src = TernaryMobileNetV5_Subband(width=128)
        sd = src.state_dict()
        # The current code already saves focal_mid.0 and focal2 aliases via
        # backward-compat. Rename focal_mid.0.* to focal2.* and strip
        # focal_mid.* / focal_last.* to simulate an old checkpoint.
        legacy = {}
        for k, v in sd.items():
            # Skip aliased focal_mid / focal_last; keep focal2/focal3 only
            if k.startswith("focal_mid.") or k.startswith("focal_last."):
                continue
            legacy[k] = v
        # Now legacy has focal2 / focal3 from the backward-compat aliases.
        # Confirm we got both before saving.
        assert any(k.startswith("focal2.conv.weight") for k in legacy), \
            "legacy state_dict missing focal2.conv.weight alias"
        sd_path = tmp_path / "legacy.ckpt"
        torch.save(legacy, str(sd_path))
        loaded = TernaryMobileNetV5_Subband.from_checkpoint(
            str(sd_path), device="cpu"
        )
        assert isinstance(loaded, TernaryMobileNetV5_Subband)
        assert loaded.n_blocks == 3

    def test_from_checkpoint_default_width_no_markers(self, tmp_path, capsys):
        """When the checkpoint has neither focal2.conv.weight nor
        focal_mid.0.conv.weight, width falls back to 128 default."""
        # Empty state dict → all keys missing → default width=128, n_blocks=3
        sd_path = tmp_path / "empty.ckpt"
        torch.save({}, str(sd_path))
        loaded = TernaryMobileNetV5_Subband.from_checkpoint(
            str(sd_path), device="cpu"
        )
        # focal_mid.0 conv weight: width=128
        assert loaded.focal_mid[0].conv.weight.shape[0] == 128
        assert loaded.n_blocks == 3

    def test_from_checkpoint_unexpected_keys_logged(self, tmp_path, capsys):
        """When the state dict carries extra keys not in the model, the
        loader prints a 'Skipped' message."""
        src = TernaryMobileNetV5_Subband(width=128)
        sd = src.state_dict()
        # Sprinkle in an unexpected key
        sd["this_key_does_not_exist.weight"] = torch.zeros(1)
        sd_path = tmp_path / "extra.ckpt"
        torch.save(sd, str(sd_path))
        TernaryMobileNetV5_Subband.from_checkpoint(str(sd_path), device="cpu")
        # The 'Skipped N old params' print branch runs.
        out = capsys.readouterr().out
        assert "Skipped" in out


# ---------------------------------------------------------------------------
# TernaryMobileNetV5_Subband_V2
# ---------------------------------------------------------------------------


class TestTernaryMobileNetV5SubbandV2:
    def test_construction(self):
        m = TernaryMobileNetV5_Subband_V2()
        # focal2/3/4 are DW-sep
        assert hasattr(m, "focal2")
        assert hasattr(m, "focal3")
        assert hasattr(m, "focal4")
        assert isinstance(m.bneck_v, INT8Conv1d)
        assert isinstance(m.bneck_g, TernaryConv1d)

    def test_forward_shape(self):
        torch.manual_seed(0)
        m = TernaryMobileNetV5_Subband_V2()
        m.eval()
        x = torch.randn(1, 21, 313)
        with torch.no_grad():
            y = m(x, quantize=True)
        assert y.shape == (1, 21, 313)
        assert torch.isfinite(y).all()

    def test_encode_latent_shape(self):
        torch.manual_seed(0)
        m = TernaryMobileNetV5_Subband_V2()
        m.eval()
        x = torch.randn(1, 21, 313)
        with torch.no_grad():
            lat = m.encode(x, quantize=True)
        assert lat.shape == (1, 32, 79)
        assert torch.isfinite(lat).all()

    def test_encode_stages(self):
        torch.manual_seed(0)
        m = TernaryMobileNetV5_Subband_V2()
        m.eval()
        x = torch.randn(1, 21, 313)
        with torch.no_grad():
            s1 = m.encode_stage1(x, quantize=True)
            s2 = m.encode_stage2(s1, quantize=True)
            s3 = m.encode_stage3(s2, quantize=True)
            bn = m._encode_bottleneck(s3, quantize=True)
        assert bn.shape == (1, 32, 79)
        assert torch.isfinite(bn).all()

    def test_encode_training_branch(self):
        torch.manual_seed(0)
        m = TernaryMobileNetV5_Subband_V2()
        m.train()
        x = torch.randn(1, 21, 313)
        lat = m.encode(x, quantize=True)
        assert lat.shape == (1, 32, 79)
        assert torch.isfinite(lat).all()

    def test_decode_inherits_from_v1(self):
        torch.manual_seed(0)
        m = TernaryMobileNetV5_Subband_V2()
        m.eval()
        lat = torch.randn(1, 32, 79).clamp(-0.99, 0.99)
        with torch.no_grad():
            out = m.decode(lat, target_len=313, quantize=True)
        assert out.shape == (1, 21, 313)

    def test_non_multiple_of_8_width_falls_to_groupnorm4(self):
        # width=42 is divisible by 6/7 but not by 8 → GroupNorm(4, 42)
        m = TernaryMobileNetV5_Subband_V2(width=44)
        # 44 % 8 = 4, so N_GROUPS=4
        assert m.focal1_norm.num_groups == 4
