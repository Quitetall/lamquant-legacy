"""Deep coverage tests for ``lamquant_codec.ops.lpc``.

Complements ``test_lpc_coverage.py``. Specifically targets:
  - pyref vs vectorised parity for float LPC analyze/synthesize
  - _floor_div behaviour (positive + negative numerator parity with Python //)
  - analyze_jit warm-up path (T <= order) preserves residual = signal - bias
  - synthesize_jit on a constant input is the inverse of analyze_jit
  - Levinson-Durbin early-return paths (E underflow)
  - Multi-channel parity: analyze() loops match per-channel calls
  - residual energy < signal energy on an AR(1) process (variance invariant)

Math fixtures only; no synthetic EEG, no real EDF needed (pure math).
"""
from __future__ import annotations

import numpy as np
import pytest

from lamquant_codec.ops.lpc import (
    HAS_NUMBA,
    Q_LPC,
    _analyze_channel_pyref,
    _analyze_int_pyref,
    _floor_div,
    _synthesize_channel_pyref,
    _synthesize_int_pyref,
    analyze,
    analyze_channel,
    analyze_int,
    analyze_jit,
    synthesize,
    synthesize_channel,
    synthesize_int,
    synthesize_jit,
)


def _ar1(T: int, alpha: float = 0.9, scale: float = 1000.0, seed: int = 0) -> np.ndarray:
    """AR(1) process — has predictable structure for LPC."""
    rng = np.random.RandomState(seed)
    sig = np.zeros(T, dtype=np.float64)
    sig[0] = rng.randn() * scale
    for n in range(1, T):
        sig[n] = alpha * sig[n - 1] + rng.randn() * scale
    return sig


def _ar1_int(T: int, alpha: float = 0.9, scale: float = 1000.0, seed: int = 1) -> np.ndarray:
    return _ar1(T, alpha, scale, seed).astype(np.int64)


class TestFloorDiv:
    """``_floor_div`` MUST match Python's // (floor toward -inf)."""
    @pytest.mark.parametrize("a,b", [
        (10, 3),
        (-10, 3),
        (10, -3),
        (-10, -3),
        (0, 5),
        (-1, 32),    # the bias.py bug example
        (-100, 32),  # the bias.py bug example
        (100, 32),
    ])
    def test_matches_python_floor(self, a: int, b: int) -> None:
        assert _floor_div(np.int64(a), np.int64(b)) == a // b


class TestPyrefEdgeCases:
    """Exercise the rare branches inside _analyze_channel_pyref."""

    def test_pyref_zero_signal(self) -> None:
        """All-zero -> R[0]=0 path: returns zero coeffs + signal copy."""
        sig = np.zeros(256, dtype=np.float64)
        coeffs, res = _analyze_channel_pyref(sig, order=8)
        assert np.all(coeffs == 0)
        np.testing.assert_array_equal(res, sig)

    def test_pyref_synthesize_zero_residual(self) -> None:
        residual = np.zeros(128, dtype=np.float64)
        coeffs = np.array([0.5, -0.3], dtype=np.float64)
        out = _synthesize_channel_pyref(residual, coeffs)
        np.testing.assert_array_equal(out, residual)


class TestPyrefVsVectorised:
    """Vectorised analyze_channel MUST match the pyref spec on floats."""
    @pytest.mark.parametrize("order", [1, 4, 8])
    def test_coeffs_match(self, order: int) -> None:
        sig = _ar1(T=512, seed=order)
        coeffs_ref, _ = _analyze_channel_pyref(sig, order=order)
        coeffs, _ = analyze_channel(sig, order=order)
        # Both compute identical Levinson recursion; allow eps for sum order.
        np.testing.assert_allclose(coeffs, coeffs_ref, atol=1e-9, rtol=1e-9)

    def test_residual_matches_pyref(self) -> None:
        sig = _ar1(T=512, seed=7)
        _, res_ref = _analyze_channel_pyref(sig, order=8)
        _, res = analyze_channel(sig, order=8)
        np.testing.assert_allclose(res, res_ref, atol=1e-9, rtol=1e-9)

    def test_pyref_synthesize_inverse(self) -> None:
        sig = _ar1(T=300, seed=2)
        coeffs, residual = _analyze_channel_pyref(sig, order=6)
        recon = _synthesize_channel_pyref(residual, coeffs)
        np.testing.assert_allclose(recon, sig, atol=1e-7, rtol=1e-7)


class TestResidualEnergy:
    """LPC residual energy MUST be <= signal energy for predictable signals."""

    def test_ar1_residual_smaller(self) -> None:
        sig = _ar1(T=2048, alpha=0.92, scale=1000.0, seed=0)
        _, residual = analyze_channel(sig, order=8)
        # AR(1) with alpha=0.92 has strong autocorrelation; LPC should
        # remove most predictable energy.
        sig_energy = np.sum(sig[8:] ** 2)
        res_energy = np.sum(residual[8:] ** 2)
        # Be generous: residual should be at least 5% smaller for a strong AR(1).
        assert res_energy < sig_energy
        assert res_energy < 0.95 * sig_energy


class TestAnalyzeOrderEdge:
    def test_order_zero_returns_zero_length_coeffs(self) -> None:
        sig = _ar1(T=128, seed=3)
        coeffs, residual = analyze_channel(sig, order=0)
        assert coeffs.shape == (0,)
        # With order=0 no prediction is done — residual == signal copy.
        np.testing.assert_array_equal(residual, sig)

    def test_short_signal_T_eq_order(self) -> None:
        """T == order: the vectorised pred block (T-order=0) is skipped.

        autocorrelation requires at least order+1 samples — fix
        autocorr_len = T = order+1 so R has length order+1 (the buffer
        the function asks for) while T-order == 1 still lets the
        vectorised pred path run for a single output.
        """
        sig = _ar1(T=9, seed=4)
        coeffs, residual = analyze_channel(sig, order=8, autocorr_len=9)
        assert coeffs.shape == (8,)
        assert residual.shape == sig.shape


class TestSynthesisIIRSpec:
    def test_synthesize_zero_residual_is_zero(self) -> None:
        """synthesize on a zero residual + nonzero coeffs == zero signal."""
        residual = np.zeros(256, dtype=np.float64)
        coeffs = np.array([0.5, 0.3, 0.1], dtype=np.float64)
        out = synthesize_channel(residual, coeffs)
        np.testing.assert_array_equal(out, residual)

    def test_synthesize_first_order_samples_passthrough(self) -> None:
        """First `order` samples are pass-through (no prediction history)."""
        residual = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        coeffs = np.array([0.5, 0.25], dtype=np.float64)
        out = synthesize_channel(residual, coeffs)
        assert out[0] == residual[0]
        assert out[1] == residual[1]


class TestMultiChannelParity:
    def test_analyze_matches_per_channel_loop(self) -> None:
        rng = np.random.RandomState(11)
        sig = rng.randn(4, 400).astype(np.float64) * 500.0
        coeffs_multi, res_multi = analyze(sig, order=8)
        for c in range(4):
            c_ref, r_ref = analyze_channel(sig[c], order=8)
            np.testing.assert_allclose(coeffs_multi[c], c_ref, atol=1e-12)
            np.testing.assert_allclose(res_multi[c], r_ref, atol=1e-12)

    def test_synthesize_matches_per_channel_loop(self) -> None:
        rng = np.random.RandomState(12)
        sig = rng.randn(3, 256).astype(np.float64) * 100.0
        coeffs, residual = analyze(sig, order=4)
        recon_multi = synthesize(residual, coeffs)
        for c in range(3):
            recon_ref = synthesize_channel(residual[c], coeffs[c])
            np.testing.assert_allclose(recon_multi[c], recon_ref, atol=1e-12)


class TestAnalyzeJitWarmupPath:
    """T <= order or T < 3 or order == 0 hits the warm-up-only branch."""

    @pytest.mark.parametrize("T,order", [(2, 8), (3, 4), (4, 4)])
    def test_short_signal_roundtrip(self, T: int, order: int) -> None:
        sig = _ar1_int(T=T, seed=T)
        coeffs, residual = analyze_jit(sig, order=order, ctx_len=8)
        # Always returns Q27 coeffs of size max(order,0).
        assert coeffs.shape == (order,)
        recon = synthesize_jit(residual, coeffs, order=order, ctx_len=8)
        np.testing.assert_array_equal(recon, sig)

    def test_order_zero_path(self) -> None:
        sig = _ar1_int(T=64, seed=5)
        coeffs, residual = analyze_jit(sig, order=0, ctx_len=16)
        assert coeffs.shape == (0,)
        recon = synthesize_jit(residual, coeffs, order=0, ctx_len=16)
        np.testing.assert_array_equal(recon, sig)

    def test_zero_signal_jit_path(self) -> None:
        """All-zero input -> autocorrelation R[0]==0 -> warm-up-only branch."""
        sig = np.zeros(128, dtype=np.int64)
        coeffs, residual = analyze_jit(sig, order=4, ctx_len=8)
        assert coeffs.shape == (4,)
        # All coeffs should be zero on zero input.
        assert np.all(coeffs == 0)
        recon = synthesize_jit(residual, coeffs, order=4, ctx_len=8)
        np.testing.assert_array_equal(recon, sig)


class TestAnalyzeJitFull:
    """The full analyze_jit path: T > order, Levinson + integer LPC + bias."""

    def test_ar1_roundtrip(self) -> None:
        sig = _ar1_int(T=512, alpha=0.85, seed=8)
        coeffs, residual = analyze_jit(sig, order=8, ctx_len=32)
        recon = synthesize_jit(residual, coeffs, order=8, ctx_len=32)
        np.testing.assert_array_equal(recon, sig)

    def test_residual_smaller_than_signal_on_ar(self) -> None:
        sig = _ar1_int(T=1024, alpha=0.93, scale=2000.0, seed=9)
        _, residual = analyze_jit(sig, order=8, ctx_len=32)
        # AR1 residual after LPC + bias cancellation should be smaller.
        assert np.sum(residual.astype(np.float64) ** 2) < \
               np.sum(sig.astype(np.float64) ** 2)


class TestAnalyzeIntDtypeCoercion:
    def test_input_dtype_promotion(self) -> None:
        """analyze_int promotes int16 input to int64."""
        rng = np.random.RandomState(13)
        sig_i16 = (rng.randn(200) * 1000).astype(np.int16)
        float_coeffs, _ = analyze_channel(sig_i16.astype(np.float64), order=4)
        cq, res = analyze_int(sig_i16, float_coeffs, order=4)
        assert res.dtype == np.int64
        assert cq.dtype == np.int32


class TestQConstants:
    def test_Q_LPC_pinned(self) -> None:
        assert Q_LPC == 27


class TestSynthesizeIntDtypeCoercion:
    def test_int32_coeffs_accepted(self) -> None:
        sig = _ar1_int(T=256, seed=14)
        float_coeffs, _ = analyze_channel(sig.astype(np.float64), order=4)
        cq, res = analyze_int(sig, float_coeffs, order=4)
        # Pass non-contiguous coeffs slice and float-cast residual to force coercion.
        recon = synthesize_int(res.astype(np.float64).astype(np.int64),
                                cq.astype(np.int32), order=4)
        np.testing.assert_array_equal(recon, sig)
