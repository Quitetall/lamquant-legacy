"""Coverage tests for ``lamquant_codec.ops.wht`` (Walsh-Hadamard Transform).

Pins:
  - shape preservation
  - lossless round-trip: inverse_32(forward_32(x)) == x
  - linearity: WHT(a + b) == WHT(a) + WHT(b)
  - input-length validation (raises on len != 32)
  - torch + numpy parity within float tolerance
  - torch shape rejection for dim 1 != 32

Math fixtures via ``np.random`` and ``torch.randn`` — not synthetic EEG.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from lamquant_codec.ops.wht import (
    forward_32,
    forward_32_torch,
    inverse_32,
    inverse_32_torch,
)


class TestForward32:
    def test_shape_preserved(self) -> None:
        x = np.random.RandomState(0).randn(32)
        y = forward_32(x)
        assert y.shape == x.shape

    def test_rejects_wrong_length(self) -> None:
        for n in (1, 16, 31, 33, 64):
            with pytest.raises(ValueError, match="WHT-32"):
                forward_32(np.zeros(n))

    def test_linearity(self) -> None:
        rng = np.random.RandomState(1)
        a = rng.randn(32)
        b = rng.randn(32)
        np.testing.assert_allclose(
            forward_32(a + b),
            forward_32(a) + forward_32(b),
            rtol=1e-12,
        )

    def test_zero_input_zero_output(self) -> None:
        x = np.zeros(32)
        np.testing.assert_array_equal(forward_32(x), 0)

    def test_does_not_mutate_input(self) -> None:
        x = np.random.RandomState(2).randn(32)
        x_copy = x.copy()
        _ = forward_32(x)
        np.testing.assert_array_equal(x, x_copy)


class TestInverse32:
    def test_lossless_roundtrip(self) -> None:
        x = np.random.RandomState(3).randn(32)
        recon = inverse_32(forward_32(x))
        np.testing.assert_allclose(recon, x, rtol=1e-12, atol=1e-12)

    def test_self_inverse_property(self) -> None:
        """WHT * WHT = N * I -> forward then inverse recovers x exactly."""
        x = np.arange(32, dtype=np.float64)
        np.testing.assert_allclose(
            inverse_32(forward_32(x)), x, rtol=1e-12,
        )


class TestForward32Torch:
    def test_shape_preserved(self) -> None:
        x = torch.randn(4, 32, 313, generator=torch.Generator().manual_seed(0))
        y = forward_32_torch(x)
        assert y.shape == x.shape

    def test_rejects_wrong_channel_dim(self) -> None:
        for c in (1, 8, 16, 64):
            with pytest.raises(ValueError, match="WHT-32"):
                forward_32_torch(torch.zeros(1, c, 100))

    def test_linearity(self) -> None:
        a = torch.randn(2, 32, 50)
        b = torch.randn(2, 32, 50)
        torch.testing.assert_close(
            forward_32_torch(a + b),
            forward_32_torch(a) + forward_32_torch(b),
            rtol=1e-5, atol=1e-5,
        )

    def test_zero_input(self) -> None:
        x = torch.zeros(3, 32, 100)
        y = forward_32_torch(x)
        assert (y == 0).all()


class TestInverse32Torch:
    def test_lossless_roundtrip(self) -> None:
        x = torch.randn(2, 32, 100)
        recon = inverse_32_torch(forward_32_torch(x))
        torch.testing.assert_close(recon, x, rtol=1e-5, atol=1e-5)


class TestParityNumpyVsTorch:
    def test_forward_matches(self) -> None:
        """numpy and torch implementations must agree on the same input."""
        rng = np.random.RandomState(7)
        x = rng.randn(32).astype(np.float64)
        y_np = forward_32(x)
        y_torch = forward_32_torch(
            torch.tensor(x, dtype=torch.float64).reshape(1, 32, 1)
        ).reshape(32).numpy()
        np.testing.assert_allclose(y_np, y_torch, rtol=1e-10, atol=1e-10)
