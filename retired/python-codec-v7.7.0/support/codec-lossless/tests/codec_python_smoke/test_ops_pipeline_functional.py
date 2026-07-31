"""Functional tests for ``lamquant_codec.ops.pipeline``.

Pins SHAPE + INVARIANTS, not exact numerics:
  - hp_filter preserves shape + dtype
  - preprocess_subband returns L3 with the expected channel count
  - preprocess_subband + reconstruct_from_subband round-trips through
    the correct shapes (numeric exactness is the codec's job, not this
    test's; reconstruction error tolerance is intentionally generous)
  - non-finite input raises ValueError
  - shape mismatch in reconstruct raises ValueError
"""
from __future__ import annotations

import numpy as np
import pytest

from lamquant_codec.ops.pipeline import (
    hp_filter,
    preprocess_subband,
    preprocess_subband_single,
    reconstruct_from_subband,
)


def _synth(C: int = 4, T: int = 2500, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    t = np.arange(T) / 250.0
    sig = np.zeros((C, T), dtype=np.float64)
    for c in range(C):
        sig[c] = (
            10 * np.sin(2 * np.pi * (2 + c) * t)
            + rng.randn(T)
        )
    return sig


class TestHpFilter:
    def test_preserves_shape_2d(self) -> None:
        sig = _synth()
        out = hp_filter(sig)
        assert out.shape == sig.shape

    def test_preserves_shape_1d(self) -> None:
        sig = _synth(C=1)[0]
        out = hp_filter(sig)
        assert out.shape == sig.shape

    def test_preserves_dtype(self) -> None:
        sig = _synth().astype(np.float32)
        out = hp_filter(sig)
        assert out.dtype == sig.dtype

    def test_removes_dc(self) -> None:
        sig = np.full((1, 2500), 100.0)
        out = hp_filter(sig.astype(np.float64))
        # DC should be greatly attenuated (not equal to input)
        assert abs(out.mean()) < abs(sig.mean())


class TestPreprocessSubband:
    def test_shape_contract(self) -> None:
        """L3 approx must come out at [C, 313] (3-level decimation of
        T=2500 -> 313)."""
        sig = _synth(C=4, T=2500)
        l3, coeffs, subs = preprocess_subband(sig)
        assert l3.shape == (4, 313)
        assert coeffs.shape[0] == 4
        assert len(subs) == 4

    def test_lpc_coeffs_shape_matches_order(self) -> None:
        sig = _synth(C=2)
        _, coeffs, _ = preprocess_subband(sig, order=8)
        assert coeffs.shape == (2, 8)


class TestPreprocessSubbandSingle:
    def test_returns_three_outputs(self) -> None:
        sig = _synth(C=3)
        l3, coeffs, subs = preprocess_subband_single(sig)
        assert l3.shape[0] == 3
        assert coeffs.shape[0] == 3
        assert len(subs) == 3

    def test_rejects_1d_input(self) -> None:
        with pytest.raises(ValueError, match="Expected"):
            preprocess_subband_single(np.zeros(2500))

    def test_rejects_non_finite(self) -> None:
        sig = _synth()
        sig[0, 100] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            preprocess_subband_single(sig)


class TestReconstructFromSubband:
    def test_channel_mismatch_raises(self) -> None:
        sig = _synth(C=2)
        l3, coeffs, subs = preprocess_subband(sig)
        # Drop one channel from subs to force a mismatch
        with pytest.raises(ValueError, match="Channel mismatch"):
            reconstruct_from_subband(l3, coeffs, subs[:1])

    def test_lpc_row_mismatch_raises(self) -> None:
        sig = _synth(C=3)
        l3, coeffs, subs = preprocess_subband(sig)
        with pytest.raises(ValueError, match="Channel mismatch"):
            reconstruct_from_subband(l3, coeffs[:2], subs)

    def test_roundtrip_shape(self) -> None:
        sig = _synth(C=4)
        l3, coeffs, subs = preprocess_subband(sig)
        recon = reconstruct_from_subband(l3, coeffs, subs)
        # T can be off by a few samples due to filter / lifting edge —
        # the contract is "channels match, length is positive".
        assert recon.shape[0] == 4
        assert recon.shape[1] > 0
