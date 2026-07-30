"""Coverage tests for `lamquant_codec.ops.wht`.

The Walsh-Hadamard transform is pure math. Tests pin:

  - 32-point forward + inverse round-trip (numpy)
  - 32-point forward + inverse round-trip (torch)
  - Forward of zero stays zero
  - Forward of one-hot impulse spreads to all 32 bins
  - Length-error on wrong input
  - Numpy and torch implementations agree
  - WHT is its own inverse up to scale (forward(forward(x)) == 32*x)
  - WHT is linear (forward(a*x + b*y) == a*forward(x) + b*forward(y))

Uses np.random / torch.randn — no synthetic EEG semantics.
"""
from __future__ import annotations

import numpy as np
import pytest

from lamquant_codec.ops.wht import (
    forward_32,
    forward_32_torch,
    inverse_32,
    inverse_32_torch,
)

pytestmark = [pytest.mark.l3]


# ============================================================
# Numpy variant
# ============================================================


class TestForward32:

    def test_zero_in_zero_out(self):
        x = np.zeros(32, dtype=np.int64)
        y = forward_32(x)
        assert y.shape == (32,)
        assert np.array_equal(y, np.zeros(32, dtype=np.int64))

    def test_dc_input_concentrates_at_bin_zero(self):
        # All ones → first row of Hadamard matrix dots to 32, others to 0.
        x = np.ones(32, dtype=np.int64)
        y = forward_32(x)
        assert y[0] == 32
        assert (y[1:] == 0).all()

    def test_self_inverse_up_to_scale(self):
        rng = np.random.default_rng(42)
        x = rng.integers(-1000, 1000, size=32).astype(np.int64)
        y = forward_32(forward_32(x))
        # H @ H = N * I  →  forward(forward(x)) = 32*x
        assert np.array_equal(y, 32 * x)

    def test_inverse_recovers_input(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(32).astype(np.float64) * 100
        y = forward_32(x)
        recovered = inverse_32(y)
        np.testing.assert_allclose(recovered, x, atol=1e-10)

    def test_wrong_length_raises(self):
        x = np.zeros(31, dtype=np.float64)
        with pytest.raises(ValueError, match="32"):
            forward_32(x)

    def test_input_is_not_mutated(self):
        rng = np.random.default_rng(1)
        x = rng.integers(-100, 100, size=32).astype(np.int64)
        before = x.copy()
        forward_32(x)
        assert np.array_equal(x, before)

    def test_linearity(self):
        rng = np.random.default_rng(2)
        x = rng.integers(-50, 50, size=32).astype(np.int64)
        y = rng.integers(-50, 50, size=32).astype(np.int64)
        a, b = 2, 3
        lhs = forward_32(a * x + b * y)
        rhs = a * forward_32(x) + b * forward_32(y)
        assert np.array_equal(lhs, rhs)


# ============================================================
# Torch variant — only if torch is importable
# ============================================================


class TestForward32Torch:

    @pytest.fixture(autouse=True)
    def _torch_or_skip(self):
        torch = pytest.importorskip("torch")
        self.torch = torch

    def test_zero_in_zero_out(self):
        x = self.torch.zeros((1, 32, 4), dtype=self.torch.float32)
        y = forward_32_torch(x)
        assert tuple(y.shape) == (1, 32, 4)
        assert bool(self.torch.all(y == 0))

    def test_self_inverse_up_to_scale(self):
        self.torch.manual_seed(0)
        x = self.torch.randn(2, 32, 8)
        y = forward_32_torch(forward_32_torch(x))
        # Allow small float-roundoff.
        self.torch.testing.assert_close(y, 32 * x, atol=1e-3, rtol=1e-3)

    def test_inverse_recovers_input(self):
        self.torch.manual_seed(1)
        x = self.torch.randn(1, 32, 6)
        recon = inverse_32_torch(forward_32_torch(x))
        self.torch.testing.assert_close(recon, x, atol=1e-3, rtol=1e-3)

    def test_wrong_dim_raises(self):
        x = self.torch.zeros((1, 31, 4))
        with pytest.raises(ValueError, match="32"):
            forward_32_torch(x)

    def test_matches_numpy_implementation(self):
        # Forward of [B,32,T] should be column-wise WHT of each time slice.
        rng = np.random.default_rng(3)
        arr = rng.standard_normal((1, 32, 4)).astype(np.float64)
        np_results = np.stack(
            [forward_32(arr[0, :, t]) for t in range(arr.shape[2])],
            axis=-1,
        )[None]
        t_results = forward_32_torch(self.torch.from_numpy(arr)).numpy()
        np.testing.assert_allclose(t_results, np_results, atol=1e-8)
