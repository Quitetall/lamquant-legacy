"""Functional tests for ``lamquant_codec.models.siren``.

Pinned contracts:
  - SIREN constructs with documented architecture (4 layers, 21 outputs)
  - forward(coords) returns [T, n_channels] tensor
  - quantize_int4 -> dequantize_int4 produces a vector within scale*7 of input
  - flatten_weights / load_flat_weights round-trips through the same shape
  - make_coords yields the right shape + range

Numeric values are NOT pinned — exact weights, exact quantized values, and
exact forward pass results depend on random init + scale and would drift
under any benign refactor.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from lamquant_codec.models.siren import (
    SIREN,
    SIREN_HIDDEN_DIM,
    SIREN_N_CHANNELS,
    SIREN_N_LAYERS,
    SIREN_N_PARAMS,
    SirenLayer,
    dequantize_int4,
    make_coords,
    quantize_int4,
)


class TestSirenLayer:
    def test_constructs(self) -> None:
        layer = SirenLayer(in_features=8, out_features=16)
        assert layer.linear.in_features == 8
        assert layer.linear.out_features == 16

    def test_forward_shape(self) -> None:
        layer = SirenLayer(in_features=4, out_features=12)
        x = torch.randn(7, 4)
        y = layer(x)
        assert y.shape == (7, 12)

    def test_forward_output_bounded(self) -> None:
        """SIREN output passes through sin → bounded in [-1, 1]."""
        layer = SirenLayer(in_features=4, out_features=8)
        y = layer(torch.randn(10, 4))
        assert y.abs().max().item() <= 1.0 + 1e-6


class TestSIREN:
    def test_default_architecture(self) -> None:
        net = SIREN()
        assert net.hidden_dim == SIREN_HIDDEN_DIM
        assert net.n_layers == SIREN_N_LAYERS
        assert net.n_channels == SIREN_N_CHANNELS
        assert net.param_count() == SIREN_N_PARAMS

    def test_forward_shape(self) -> None:
        net = SIREN()
        coords = make_coords(T=128)
        out = net(coords)
        assert out.shape == (128, SIREN_N_CHANNELS)

    def test_flatten_weights_shape(self) -> None:
        net = SIREN()
        flat = net.flatten_weights()
        assert flat.shape == (SIREN_N_PARAMS,)
        assert flat.dtype == np.float32

    def test_load_flat_weights_roundtrip(self) -> None:
        net = SIREN()
        original = net.flatten_weights()
        rng = np.random.RandomState(0)
        new_weights = rng.randn(SIREN_N_PARAMS).astype(np.float32) * 0.01
        net.load_flat_weights(new_weights)
        recovered = net.flatten_weights()
        assert recovered.shape == new_weights.shape
        np.testing.assert_allclose(recovered, new_weights, rtol=1e-5)


class TestMakeCoords:
    def test_shape(self) -> None:
        c = make_coords(T=64)
        assert c.shape == (64, 1)

    def test_range(self) -> None:
        c = make_coords(T=100)
        assert c.min().item() == pytest.approx(-1.0)
        assert c.max().item() == pytest.approx(1.0)

    def test_monotonic(self) -> None:
        c = make_coords(T=50).squeeze()
        assert torch.all(c[1:] > c[:-1])


class TestQuantizeInt4:
    def test_returns_int8_and_scale(self) -> None:
        w = np.random.RandomState(7).randn(64).astype(np.float32)
        q, scale = quantize_int4(w)
        assert q.dtype == np.int8
        assert isinstance(scale, float)
        assert scale > 0

    def test_quantized_values_in_int4_range(self) -> None:
        w = np.random.RandomState(0).randn(100).astype(np.float32)
        q, _ = quantize_int4(w)
        assert q.min() >= -7
        assert q.max() <= 7

    def test_zero_input_handled(self) -> None:
        w = np.zeros(32, dtype=np.float32)
        q, scale = quantize_int4(w)
        assert (q == 0).all()
        assert scale > 0  # guard against div-by-zero

    def test_roundtrip_within_scale(self) -> None:
        """|dequant(quant(x)) - x| <= scale (one quantization step)."""
        w = np.random.RandomState(2).randn(128).astype(np.float32) * 5
        q, scale = quantize_int4(w)
        recovered = dequantize_int4(q, scale)
        assert np.abs(recovered - w).max() <= scale + 1e-6


class TestDequantizeInt4:
    def test_dtype_is_float32(self) -> None:
        q = np.array([-7, -3, 0, 3, 7], dtype=np.int8)
        out = dequantize_int4(q, scale=1.5)
        assert out.dtype == np.float32

    def test_linearity(self) -> None:
        q = np.array([-7, 0, 7], dtype=np.int8)
        out = dequantize_int4(q, scale=2.0)
        np.testing.assert_allclose(out, [-14.0, 0.0, 14.0])
