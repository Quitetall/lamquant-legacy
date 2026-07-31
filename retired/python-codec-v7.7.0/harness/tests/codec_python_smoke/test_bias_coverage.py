"""Coverage tests for ``lamquant_codec.ops.bias``.

Pins behavioural contracts, not numeric magic:
  - cancel_jit(restore_jit) and restore_jit(cancel_jit) round-trip exactly
  - cancel_jit on zero input returns zero
  - cancel_jit reduces magnitude on a DC-biased input
  - ctx_len < 1 raises ValueError
  - Pure shape preservation

Math fixtures only (np.random). No EEG.
"""
from __future__ import annotations

import numpy as np
import pytest

from lamquant_codec.ops.bias import BIAS_CTX_LEN, cancel_jit, restore_jit


def _math_residual(n: int = 512, scale: int = 100, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return (rng.randn(n) * scale).astype(np.int64)


class TestCancelRestoreRoundtrip:
    @pytest.mark.parametrize("ctx_len", [1, 4, 32, 64])
    def test_roundtrip_default_inputs(self, ctx_len: int) -> None:
        sig = _math_residual(n=512, scale=50)
        cancelled = cancel_jit(sig, ctx_len)
        restored = restore_jit(cancelled, ctx_len)
        np.testing.assert_array_equal(restored, sig)

    def test_dtype_int64_preserved(self) -> None:
        sig = _math_residual(n=128)
        out = cancel_jit(sig, BIAS_CTX_LEN)
        assert out.dtype == np.int64

    def test_shape_preserved(self) -> None:
        for n in (1, 10, 32, 33, 1024):
            sig = _math_residual(n=n, seed=n)
            out = cancel_jit(sig, BIAS_CTX_LEN)
            assert out.shape == (n,)


class TestZeroInput:
    def test_zero_residual_cancel_is_zero(self) -> None:
        zeros = np.zeros(128, dtype=np.int64)
        out = cancel_jit(zeros, BIAS_CTX_LEN)
        np.testing.assert_array_equal(out, zeros)

    def test_zero_residual_restore_is_zero(self) -> None:
        zeros = np.zeros(128, dtype=np.int64)
        out = restore_jit(zeros, BIAS_CTX_LEN)
        np.testing.assert_array_equal(out, zeros)


class TestDcBiasReduction:
    def test_constant_input_first_sample_unchanged(self) -> None:
        """On the first sample the running buf is empty, so cancel is a no-op
        and restore is also a no-op."""
        sig = np.full(256, 1000, dtype=np.int64)
        cancelled = cancel_jit(sig, BIAS_CTX_LEN)
        # Sample 0: bias = 0/ctx_len = 0; out[0] == sig[0]
        assert cancelled[0] == sig[0]
        # Roundtrip still exact.
        restored = restore_jit(cancelled, BIAS_CTX_LEN)
        np.testing.assert_array_equal(restored, sig)

    def test_dc_bias_reduces_sum_magnitude(self) -> None:
        """After the warm-up window, a constant input becomes nearly zero."""
        sig = np.full(512, 500, dtype=np.int64)
        cancelled = cancel_jit(sig, BIAS_CTX_LEN)
        # Tail of cancelled signal should be near zero (predictable bias).
        # Specifically magnitudes much smaller than the original DC offset.
        assert np.abs(cancelled[BIAS_CTX_LEN * 2:]).max() < np.abs(sig).max()


class TestValidation:
    @pytest.mark.parametrize("bad", [0, -1, -100])
    def test_cancel_ctx_len_below_one_raises(self, bad: int) -> None:
        with pytest.raises(ValueError):
            cancel_jit(np.zeros(8, dtype=np.int64), bad)

    @pytest.mark.parametrize("bad", [0, -5])
    def test_restore_ctx_len_below_one_raises(self, bad: int) -> None:
        with pytest.raises(ValueError):
            restore_jit(np.zeros(8, dtype=np.int64), bad)


class TestNegativeValues:
    def test_negative_input_roundtrip(self) -> None:
        sig = -_math_residual(n=256, scale=200)
        out = cancel_jit(sig, BIAS_CTX_LEN)
        restored = restore_jit(out, BIAS_CTX_LEN)
        np.testing.assert_array_equal(restored, sig)


class TestBiasCtxLenConstant:
    def test_default_pinned(self) -> None:
        assert BIAS_CTX_LEN == 32
