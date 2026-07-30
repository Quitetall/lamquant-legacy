"""Deep coverage tests for ``lamquant_codec.ops.bias``.

Complements ``test_bias_coverage.py``. Targets:
  - cancel_jit on random + constant + sinusoidal inputs: roundtrip exact
  - Magnitude reduction: cancelled signal max should not blow up
  - Validation: ctx_len < 1 raises ValueError
  - Inner _floor_div parity with Python //  (when numba is present)
  - Pure-Python fallback path equivalence (not directly executable when
    numba is installed — covered by behaviour pinning instead)

Math fixtures only. No real EDF / EEG required.
"""
from __future__ import annotations

import numpy as np
import pytest

from lamquant_codec.ops.bias import BIAS_CTX_LEN, cancel_jit, restore_jit


def _math_signal(n: int, kind: str = "random", scale: int = 100,
                 seed: int = 0) -> np.ndarray:
    if kind == "random":
        rng = np.random.RandomState(seed)
        return (rng.randn(n) * scale).astype(np.int64)
    if kind == "constant":
        return np.full(n, scale, dtype=np.int64)
    if kind == "sinusoid":
        t = np.arange(n)
        return (np.sin(2 * np.pi * t / 64.0) * scale).astype(np.int64)
    if kind == "ramp":
        return np.arange(n, dtype=np.int64) - n // 2
    raise ValueError(f"unknown kind: {kind}")


class TestRoundtripAllKinds:
    @pytest.mark.parametrize("kind", ["random", "constant", "sinusoid", "ramp"])
    @pytest.mark.parametrize("ctx_len", [1, 4, 16, 32, 64, 128])
    def test_roundtrip_exact(self, kind: str, ctx_len: int) -> None:
        sig = _math_signal(512, kind=kind, scale=200, seed=ctx_len)
        cancelled = cancel_jit(sig, ctx_len)
        restored = restore_jit(cancelled, ctx_len)
        np.testing.assert_array_equal(restored, sig)


class TestEdgeCases:
    def test_length_1_signal(self) -> None:
        sig = np.array([42], dtype=np.int64)
        cancelled = cancel_jit(sig, BIAS_CTX_LEN)
        # First sample with empty buf: bias = 0/ctx = 0 -> identity
        assert cancelled[0] == sig[0]
        np.testing.assert_array_equal(restore_jit(cancelled, BIAS_CTX_LEN), sig)

    def test_length_0_signal(self) -> None:
        sig = np.zeros(0, dtype=np.int64)
        cancelled = cancel_jit(sig, BIAS_CTX_LEN)
        assert cancelled.shape == (0,)

    def test_large_negative_bias(self) -> None:
        """Constant strongly-negative input: roundtrip must remain exact."""
        sig = np.full(256, -1000, dtype=np.int64)
        cancelled = cancel_jit(sig, BIAS_CTX_LEN)
        restored = restore_jit(cancelled, BIAS_CTX_LEN)
        np.testing.assert_array_equal(restored, sig)


class TestMagnitudeReduction:
    """After warm-up, cancellation reduces magnitude on DC-biased signals."""

    def test_sinusoid_plus_dc(self) -> None:
        """Sinusoid+DC: tail of cancelled should oscillate around 0, not DC."""
        dc = 500
        sig = _math_signal(1024, kind="sinusoid", scale=100) + dc
        sig = sig.astype(np.int64)
        cancelled = cancel_jit(sig, 64)
        # Tail mean magnitude is much smaller than DC bias.
        tail = cancelled[200:]
        assert np.abs(np.mean(tail)) < dc

    def test_ramp_signal_handled(self) -> None:
        """Linear ramp: bias cancellation should track the running mean.
        Roundtrip must still be exact."""
        sig = _math_signal(1024, kind="ramp")
        cancelled = cancel_jit(sig, 32)
        restored = restore_jit(cancelled, 32)
        np.testing.assert_array_equal(restored, sig)


class TestValidationDeep:
    @pytest.mark.parametrize("bad_ctx", [0, -1, -1000])
    def test_cancel_validation(self, bad_ctx: int) -> None:
        sig = np.zeros(32, dtype=np.int64)
        with pytest.raises(ValueError, match="ctx_len"):
            cancel_jit(sig, bad_ctx)

    @pytest.mark.parametrize("bad_ctx", [0, -1, -1000])
    def test_restore_validation(self, bad_ctx: int) -> None:
        sig = np.zeros(32, dtype=np.int64)
        with pytest.raises(ValueError, match="ctx_len"):
            restore_jit(sig, bad_ctx)

    def test_float_ctx_len_coerced_to_int(self) -> None:
        """int(ctx_len) coerces float — 4.0 -> 4, 4.7 -> 4 (int truncation)."""
        sig = _math_signal(32, kind="random")
        # ctx_len=4.0 -> coerced to 4, should work.
        out = cancel_jit(sig, 4.0)
        restored = restore_jit(out, 4.0)
        np.testing.assert_array_equal(restored, sig)


class TestSelfInverse:
    """cancel_jit(restore_jit(x)) for the same ctx_len.

    cancel & restore are MIRROR operations -- restore is the inverse of cancel.
    Restore is NOT idempotent, but a single cancel→restore must give back the
    original signal.
    """

    def test_cancel_then_restore_is_identity(self) -> None:
        sig = _math_signal(256, kind="random", scale=300)
        out = restore_jit(cancel_jit(sig, BIAS_CTX_LEN), BIAS_CTX_LEN)
        np.testing.assert_array_equal(out, sig)


class TestFloorDivCorrectness:
    """The bias module exposes a fused jit; sanity-check the bias arithmetic
    via roundtrip on signals that previously hit the -1/32 = -0 vs -1 bug."""

    @pytest.mark.parametrize("scale", [-100, -3, -1, 1, 3, 100])
    def test_small_constant_inputs_roundtrip(self, scale: int) -> None:
        sig = np.full(128, scale, dtype=np.int64)
        out = restore_jit(cancel_jit(sig, 32), 32)
        np.testing.assert_array_equal(out, sig)


class TestBiasCtxLenExport:
    def test_constant_exported(self) -> None:
        # Imported via constants module.
        assert isinstance(BIAS_CTX_LEN, int)
        assert BIAS_CTX_LEN >= 1
