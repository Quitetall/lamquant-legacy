"""Subband decomposition — RawEEG → SubbandDecomposition.

Pins lamquant_codec.decompose.decompose():
  - Input: RawEEG with [C, T] signal
  - Output: SubbandDecomposition with l3_approx [C, T/8], detail bands,
    LPC coefficients, original source preserved
  - Shape invariants per lifting level
  - Source-signal pass-through (for downstream lossless codec)
"""
from __future__ import annotations

import numpy as np
import pytest

from lamquant_codec.codec_types import RawEEG, SubbandDecomposition
from lamquant_codec.decompose import decompose

pytestmark = pytest.mark.l3


def _raw(n_ch: int = 21, T: int = 2500, *, seed: int = 0) -> RawEEG:
    rng = np.random.default_rng(seed)
    sig = (rng.standard_normal((n_ch, T)).astype(np.float32) * 50).astype(np.float32)
    return RawEEG(signal=sig, sample_rate=250.0)


# ============================================================
# 1. Output type + populated fields
# ============================================================


class TestOutputType:

    def test_returns_subband_decomposition(self):
        out = decompose(_raw())
        assert isinstance(out, SubbandDecomposition)

    def test_l3_approx_populated(self):
        out = decompose(_raw())
        assert out.l3_approx is not None
        assert isinstance(out.l3_approx, np.ndarray)

    def test_lpc_coeffs_populated(self):
        out = decompose(_raw(), lpc_order=8)
        assert out.lpc_coeffs is not None
        assert isinstance(out.lpc_coeffs, np.ndarray)
        assert out.lpc_order == 8


# ============================================================
# 2. Shape invariants
# ============================================================


class TestShapes:

    def test_l3_approx_has_correct_channel_count(self):
        out = decompose(_raw(n_ch=21))
        assert out.l3_approx.shape[0] == 21

    @pytest.mark.parametrize("n_ch", [1, 4, 21, 32])
    def test_l3_approx_channel_count_parametrized(self, n_ch):
        out = decompose(_raw(n_ch=n_ch))
        assert out.l3_approx.shape[0] == n_ch

    def test_l3_approx_time_dim_is_eighth_of_input(self):
        # 3 levels of dyadic downsampling → length / 8.
        out = decompose(_raw(T=2500))
        # Tolerate ±2 samples for boundary handling.
        assert abs(out.l3_approx.shape[1] - 312) <= 2

    def test_detail_bands_progressively_shrink(self):
        out = decompose(_raw(n_ch=4, T=2500))
        # l1 = T/2, l2 = T/4, l3 = T/8 (within ±2)
        assert abs(out.l1_detail.shape[1] - 1250) <= 2
        assert abs(out.l2_detail.shape[1] - 625) <= 2
        assert abs(out.l3_detail.shape[1] - 312) <= 2

    def test_detail_band_channel_count_matches_input(self):
        out = decompose(_raw(n_ch=21, T=2500))
        assert out.l1_detail.shape[0] == 21
        assert out.l2_detail.shape[0] == 21
        assert out.l3_detail.shape[0] == 21


# ============================================================
# 3. Source signal pass-through (for downstream lossless codec)
# ============================================================


class TestSourcePassThrough:

    def test_source_signal_preserved_byte_exact(self):
        raw = _raw(seed=42)
        out = decompose(raw)
        np.testing.assert_array_equal(out.source_signal, raw.signal)

    def test_source_dtype_preserved(self):
        raw = _raw(seed=43)
        out = decompose(raw)
        assert out.source_signal.dtype == raw.signal.dtype


# ============================================================
# 4. Input shape normalization
# ============================================================


class TestInputShape:

    def test_1d_signal_promoted_to_single_channel(self):
        """RawEEG.__post_init__ reshapes 1D → (1, T); decompose then runs
        as a single-channel pipeline."""
        raw = RawEEG(signal=np.zeros(2500, dtype=np.float32),
                     sample_rate=250.0)
        assert raw.signal.ndim == 2 and raw.signal.shape[0] == 1
        out = decompose(raw)
        assert out.l3_approx.shape[0] == 1

    def test_rejects_3d_signal(self):
        """3D signals pass through RawEEG unchanged but decompose must
        refuse them with a clear ValueError."""
        raw = RawEEG(signal=np.zeros((2, 4, 100), dtype=np.float32),
                     sample_rate=250.0)
        with pytest.raises(ValueError, match=r"\[C, T\]"):
            decompose(raw)


# ============================================================
# 5. Determinism — same input twice gives same decomposition
# ============================================================


class TestDeterminism:

    def test_identical_input_gives_identical_output(self):
        raw1 = _raw(seed=99)
        raw2 = _raw(seed=99)
        out1 = decompose(raw1)
        out2 = decompose(raw2)
        np.testing.assert_array_equal(out1.l3_approx, out2.l3_approx)
        np.testing.assert_array_equal(out1.l1_detail, out2.l1_detail)
        np.testing.assert_array_equal(out1.l2_detail, out2.l2_detail)
        np.testing.assert_array_equal(out1.l3_detail, out2.l3_detail)
