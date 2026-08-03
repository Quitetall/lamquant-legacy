"""Fused pipeline tests.

Verifies fused_compress/fused_decompress produce byte-identical output
to the reference path, and that canonical numba functions round-trip.
"""
import pytest
import numpy as np


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestFusedCompress:
    """fused_compress must be byte-identical to _compress_bytes."""

    def test_byte_identical(self, rng):
        from lamquant_codec.lossless import _compress_bytes_ref
        from lamquant_codec.ops.fused_lml import fused_compress, HAS_NUMBA
        if not HAS_NUMBA:
            pytest.skip("numba not available")

        for n_ch, T in [(1, 250), (4, 1250), (21, 2500)]:
            sig = rng.integers(-5000, 5000, (n_ch, T)).astype(np.float64)
            ref = _compress_bytes_ref(sig)
            fused = fused_compress(sig)
            assert ref == fused, f"Mismatch at {n_ch}ch x {T}"


class TestCanonicalFunctions:
    """lifting_1d_forward/inverse_int_jit and lpc_analyze/synthesize_jit."""

    def test_lifting_roundtrip(self, rng):
        import sys, os
        sys.path.insert(0, os.path.join(os.getcwd(), 'ai_models', 'student'))
        from subband_preprocess import (
            lifting_1d_forward_int_jit, lifting_1d_inverse_int_jit,
        )
        for T in [10, 63, 128, 625, 2500]:
            sig = rng.integers(-5000, 5000, T, dtype=np.int64)
            a, d = lifting_1d_forward_int_jit(sig)
            rec = lifting_1d_inverse_int_jit(a, d)
            assert np.array_equal(sig, rec), f"Fail at T={T}"

    def test_lpc_roundtrip(self, rng):
        import sys, os
        sys.path.insert(0, os.path.join(os.getcwd(), 'ai_models', 'student'))
        from subband_preprocess import lpc_analyze_jit, lpc_synthesize_jit

        for T in [50, 313, 625]:
            for order in [1, 2, 3]:
                sig = rng.integers(-3000, 3000, T, dtype=np.int64)
                coeffs, res = lpc_analyze_jit(sig, np.int64(order), np.int64(16))
                rec = lpc_synthesize_jit(res, coeffs, np.int64(order), np.int64(16))
                assert np.array_equal(sig, rec), \
                    f"Fail at T={T} order={order}"
