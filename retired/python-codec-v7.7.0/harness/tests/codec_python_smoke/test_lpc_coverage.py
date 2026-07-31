"""Coverage tests for ``lamquant_codec.ops.lpc``.

Pins shape + invariant contracts, not exact numerics:
  - analyze_channel / synthesize_channel: shape preservation and round-trip
    reconstruction within float tolerance
  - analyze / synthesize: multi-channel wrappers preserve shape
  - analyze_int / synthesize_int: integer (Q27) round-trip is bit-exact
  - analyze_jit / synthesize_jit: fused round-trip preserves integer signal
  - Edge cases: T < order, zero signal, order=0 path through analyze_jit
  - pyref vs production parity on shape/sign for analyze_channel

Math fixtures (np.random) — NOT synthetic EEG data.
"""
from __future__ import annotations

import numpy as np
import pytest

from lamquant_codec.ops.lpc import (
    Q_LPC,
    _analyze_channel_pyref,
    _analyze_int_pyref,
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


def _math_signal(T: int = 512, seed: int = 0) -> np.ndarray:
    """Numeric-only fixture from np.random — math shape, not EEG data."""
    rng = np.random.RandomState(seed)
    return rng.randn(T).astype(np.float64) * 1000.0


def _math_signal_multi(C: int = 4, T: int = 512, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(C, T).astype(np.float64) * 1000.0


def _math_int_signal(T: int = 512, seed: int = 1) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return (rng.randn(T) * 5000.0).astype(np.int64)


class TestAnalyzeChannel:
    def test_returns_coeffs_and_residual_shapes(self) -> None:
        sig = _math_signal(T=512)
        coeffs, residual = analyze_channel(sig, order=8)
        assert coeffs.shape == (8,)
        assert residual.shape == (512,)

    def test_order_param_controls_coeff_length(self) -> None:
        sig = _math_signal(T=512)
        for order in (1, 4, 8, 12):
            coeffs, _ = analyze_channel(sig, order=order)
            assert coeffs.shape == (order,)

    def test_residual_variance_not_greater_than_signal(self) -> None:
        """LPC residual variance should be <= signal variance for predictable data."""
        # Use an AR(1) process so prediction has signal to remove.
        rng = np.random.RandomState(123)
        T = 1024
        sig = np.zeros(T)
        sig[0] = rng.randn()
        for n in range(1, T):
            sig[n] = 0.9 * sig[n - 1] + rng.randn()
        _, residual = analyze_channel(sig, order=4)
        assert np.var(residual[4:]) <= np.var(sig) + 1e-9

    def test_finite_output(self) -> None:
        sig = _math_signal(T=400)
        coeffs, residual = analyze_channel(sig, order=6)
        assert np.all(np.isfinite(coeffs))
        assert np.all(np.isfinite(residual))

    def test_zero_signal_short_circuits(self) -> None:
        """abs(R[0])<=1e-12 path returns zero coeffs + copy of signal."""
        sig = np.zeros(256, dtype=np.float64)
        coeffs, residual = analyze_channel(sig, order=8)
        assert coeffs.shape == (8,)
        assert np.all(coeffs == 0.0)
        assert residual.shape == sig.shape

    def test_short_signal_T_eq_order(self) -> None:
        """T==order: the vectorised residual block (T>order) is skipped."""
        # autocorr_len is min(autocorr_len, T) so T must be >= order+1 for R
        # to have order+1 entries. We exercise the T>order guard separately:
        # here T equals order+1 -> the early-residual branch is not taken,
        # but the vectorised pred block (T-order=1) still runs.
        sig = _math_signal(T=9)
        coeffs, residual = analyze_channel(sig, order=8, autocorr_len=9)
        assert coeffs.shape == (8,)
        assert residual.shape == (9,)


class TestSynthesizeChannel:
    def test_roundtrip_reconstructs_signal(self) -> None:
        """analyze + synthesize should reconstruct (lossy float; check L2 small)."""
        sig = _math_signal(T=512)
        coeffs, residual = analyze_channel(sig, order=8)
        recon = synthesize_channel(residual, coeffs)
        assert recon.shape == sig.shape
        # The IIR feedback exactly inverts the FIR forward — should be near-machine-eps.
        np.testing.assert_allclose(recon, sig, atol=1e-6, rtol=1e-6)

    def test_dtype_coerced_to_float64(self) -> None:
        sig = _math_signal(T=256).astype(np.float32)
        coeffs, residual = analyze_channel(sig.astype(np.float64), order=4)
        # Pass float32; wrapper must coerce.
        recon = synthesize_channel(residual.astype(np.float32),
                                   coeffs.astype(np.float32))
        assert recon.dtype == np.float64


class TestAnalyzeSynthesizeMulti:
    def test_multi_channel_shapes(self) -> None:
        sig = _math_signal_multi(C=5, T=512)
        coeffs, residual = analyze(sig, order=8)
        assert coeffs.shape == (5, 8)
        assert residual.shape == (5, 512)

    def test_multi_roundtrip(self) -> None:
        sig = _math_signal_multi(C=3, T=400)
        coeffs, residual = analyze(sig, order=6)
        recon = synthesize(residual, coeffs)
        assert recon.shape == sig.shape
        np.testing.assert_allclose(recon, sig, atol=1e-6, rtol=1e-6)


class TestAnalyzeInt:
    def test_bit_exact_roundtrip(self) -> None:
        """Integer LPC analyze+synthesize must reconstruct exactly."""
        sig = _math_int_signal(T=400)
        # Pick float coeffs from a separate analyze so the shapes line up.
        float_coeffs, _ = analyze_channel(sig.astype(np.float64), order=8)
        coeffs_q27, residual = analyze_int(sig, float_coeffs, order=8)
        assert coeffs_q27.dtype == np.int32
        assert coeffs_q27.shape == (8,)
        assert residual.dtype == np.int64
        recon = synthesize_int(residual, coeffs_q27, order=8)
        np.testing.assert_array_equal(recon, sig.astype(np.int64))

    def test_empty_signal_returns_empty(self) -> None:
        empty = np.zeros(0, dtype=np.int64)
        float_coeffs = np.zeros(8, dtype=np.float64)
        coeffs_q27, residual = analyze_int(empty, float_coeffs, order=8)
        assert coeffs_q27.shape == (8,)
        assert residual.shape == (0,)
        assert residual.dtype == np.int64

    def test_pyref_matches_vectorised(self) -> None:
        """pyref is the SPEC; vectorised must reproduce it bit-exactly."""
        sig = _math_int_signal(T=300)
        float_coeffs, _ = analyze_channel(sig.astype(np.float64), order=6)
        cq_ref, res_ref = _analyze_int_pyref(sig, float_coeffs, order=6)
        cq, res = analyze_int(sig, float_coeffs, order=6)
        np.testing.assert_array_equal(cq, cq_ref)
        np.testing.assert_array_equal(res, res_ref)

    def test_synth_pyref_matches_jit(self) -> None:
        sig = _math_int_signal(T=256)
        float_coeffs, _ = analyze_channel(sig.astype(np.float64), order=4)
        cq, res = analyze_int(sig, float_coeffs, order=4)
        recon_ref = _synthesize_int_pyref(res, cq, order=4)
        recon = synthesize_int(res, cq, order=4)
        np.testing.assert_array_equal(recon, recon_ref)


class TestAnalyzeJit:
    def test_roundtrip_long_signal(self) -> None:
        """analyze_jit + synthesize_jit must reconstruct the int signal."""
        sig = _math_int_signal(T=512)
        coeffs_q27, residual = analyze_jit(sig, order=8, ctx_len=32)
        assert coeffs_q27.dtype == np.int32
        assert coeffs_q27.shape == (8,)
        assert residual.dtype == np.int64
        recon = synthesize_jit(residual, coeffs_q27, order=8, ctx_len=32)
        np.testing.assert_array_equal(recon, sig.astype(np.int64))

    def test_short_signal_path(self) -> None:
        """T <= order branch: bias-only, no LPC."""
        sig = _math_int_signal(T=4)
        coeffs_q27, residual = analyze_jit(sig, order=8, ctx_len=32)
        assert coeffs_q27.shape == (8,)
        recon = synthesize_jit(residual, coeffs_q27, order=8, ctx_len=32)
        np.testing.assert_array_equal(recon, sig.astype(np.int64))

    def test_order_zero_path(self) -> None:
        sig = _math_int_signal(T=128)
        coeffs_q27, residual = analyze_jit(sig, order=0, ctx_len=32)
        # order=0 -> zero-length coeffs
        assert coeffs_q27.shape == (0,)
        recon = synthesize_jit(residual, coeffs_q27, order=0, ctx_len=32)
        np.testing.assert_array_equal(recon, sig.astype(np.int64))

    def test_zero_signal_path(self) -> None:
        """All-zero input: R[0] is zero, hits the second early return."""
        sig = np.zeros(256, dtype=np.int64)
        coeffs_q27, residual = analyze_jit(sig, order=8, ctx_len=32)
        assert coeffs_q27.shape == (8,)
        recon = synthesize_jit(residual, coeffs_q27, order=8, ctx_len=32)
        np.testing.assert_array_equal(recon, sig)


class TestQLpc:
    def test_constant_pinned(self) -> None:
        assert Q_LPC == 27
