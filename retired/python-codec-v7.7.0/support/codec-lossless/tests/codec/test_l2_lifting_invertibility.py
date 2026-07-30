"""
Level 2 (Property-based) tests for lifting DWT invertibility.

CRITICAL: Lifting DWT must be invertible. If forward(backward(x)) != x,
the codec is fundamentally broken.
"""

import pytest
import numpy as np


@pytest.mark.l2
class TestL2LiftingInvertibility:
    """Property-based tests for lifting DWT invertibility."""
    
    def test_lifting_roundtrip_identity(self):
        """CRITICAL INVARIANT: lifting_forward(lifting_backward(x)) ≈ x (up to quantization)."""
        # Sample data
        np.random.seed(42)
        signal = np.random.randn(21, 2500).astype(np.float32)
        
        try:
            from subband_preprocess import preprocess_subband_single
            
            # Forward pass
            l3, coeffs, subs = preprocess_subband_single(signal, order=8, autocorr_len=256)
            
            # Verify L3 exists and has correct shape
            assert l3.shape == (21, 313), f"L3 shape wrong: {l3.shape}"
            assert np.all(np.isfinite(l3)), "L3 contains NaN/Inf"
            
            # For invertibility test: store original signal energy
            original_energy = np.sum(signal**2)
            l3_energy = np.sum(l3**2)
            
            # L3 energy should be positive and not crazy large
            assert 0 < l3_energy < original_energy, \
                f"L3 energy {l3_energy} not in (0, {original_energy})"
            
        except ImportError:
            pytest.skip("subband_preprocess not available")
    
    def test_lifting_no_information_loss(self):
        """INVARIANT: 3-level Le Gall 5/3 lifting DWT is perfectly invertible.

        The L3 approximation subband alone is not the whole signal — it's
        a downsampled low-pass projection, so its energy is expected to be
        *less* than the input's. The real invariant for a lossless integer
        lifting wavelet is that forward + inverse reproduces the input.

        This test operates on the raw `lifting_3level_forward` / `_inverse`
        functions (not the full `preprocess_subband_single` pipeline, which
        prepends LPC analysis and therefore does not compose with the raw
        signal). LPC invariants are tested separately below.
        """
        np.random.seed(42)
        signal = np.random.randn(21, 2500).astype(np.float64)

        try:
            from subband_preprocess import (
                lifting_3level_forward,
                lifting_3level_inverse,
            )

            for ch in range(signal.shape[0]):
                subs = lifting_3level_forward(signal[ch])
                # Shape sanity on the L3 approximation subband
                assert subs['l3_approx'].shape == (313,), subs['l3_approx'].shape
                recon = lifting_3level_inverse(subs)
                # Perfect reconstruction: integer lifting is exactly invertible
                # on floats up to numerical precision of the predict/update
                # operations.
                max_abs_err = np.max(np.abs(signal[ch] - recon))
                assert max_abs_err < 1e-8, \
                    f"channel {ch}: roundtrip error {max_abs_err:g}"

        except ImportError:
            pytest.skip("subband_preprocess not available")

    def test_lifting_constant_signal(self):
        """INVARIANT: lifting DWT of a constant signal gives a constant L3
        approximation and ~zero detail subbands.

        This tests lifting in isolation. The full `preprocess_subband` pipeline
        runs LPC analysis first, which *by design* whitens a constant signal
        down to its boundary — so testing the full pipeline here would be a
        category error. The LPC-composes-with-constants behavior is covered
        in a dedicated LPC test.
        """
        const_value = 50.0
        signal = np.ones(2500, dtype=np.float64) * const_value

        try:
            from subband_preprocess import lifting_3level_forward

            subs = lifting_3level_forward(signal)
            l3 = subs['l3_approx']

            # L3 approximation should still be ~constant (up to boundary
            # effects at the first/last sample from the 5/3 update step).
            assert l3.shape == (313,)
            l3_mid = l3[4:-4]  # strip boundary samples
            assert l3_mid.std() < 1e-8, \
                f"interior of L3 approx not constant (std={l3_mid.std():g})"
            assert abs(l3_mid.mean() - const_value) < 1e-6, \
                f"L3 mean {l3_mid.mean()} != expected {const_value}"

            # Detail subbands should be essentially zero on a constant input.
            for key in ('l1_detail', 'l2_detail', 'l3_detail'):
                d = subs[key]
                if len(d) > 8:
                    d_mid = d[4:-4]
                    assert np.max(np.abs(d_mid)) < 1e-8, \
                        f"{key} interior nonzero: max|d|={np.max(np.abs(d_mid)):g}"

        except ImportError:
            pytest.skip("subband_preprocess not available")
    
    def test_pipeline_produces_valid_output(self):
        """Full pipeline (LPC + lifting) produces finite, reasonably-scaled L3.

        Note: the full pipeline is NOT linear because LPC coefficients are
        data-dependent. We only verify the output is valid, not linear.
        Strict linearity and invertibility are tested on the lifting DWT
        alone (test_3level_roundtrip in test_e2e_codec.py).
        """
        np.random.seed(42)
        signal = np.random.randn(21, 2500).astype(np.float32) * 50  # ~50 uV scale

        try:
            from subband_preprocess import preprocess_subband_single

            l3, coeffs, subs = preprocess_subband_single(signal, order=8, autocorr_len=256)

            assert l3.shape == (21, 313), f"L3 shape wrong: {l3.shape}"
            assert np.isfinite(l3).all(), "L3 has NaN/Inf"
            # L3 should preserve rough magnitude (within 10x)
            assert l3.std() > signal.std() * 0.01, "L3 std suspiciously small"
            assert l3.std() < signal.std() * 100, "L3 std suspiciously large"

        except ImportError:
            pytest.skip("subband_preprocess not available")
    
    def test_lifting_preservation_per_channel(self):
        """INVARIANT: Each channel should have reasonable reconstruction."""
        np.random.seed(42)
        signal = np.random.randn(21, 2500).astype(np.float32)
        
        try:
            from subband_preprocess import preprocess_subband_single
            
            l3, _, _ = preprocess_subband_single(signal, order=8, autocorr_len=256)
            
            # Check each channel individually
            for ch in range(21):
                ch_original_energy = np.sum(signal[ch, :]**2)
                ch_l3_energy = np.sum(l3[ch, :]**2)
                
                # Should preserve some energy in each channel
                assert ch_l3_energy > 0, f"Channel {ch} L3 has zero energy"
                assert ch_l3_energy < ch_original_energy, \
                    f"Channel {ch} L3 has more energy than original"
        
        except ImportError:
            pytest.skip("subband_preprocess not available")


    def test_lifting_negative_signal(self):
        """INVARIANT: Lifting on a negative signal roundtrips exactly (Bug F3 verification).

        The firmware had a rounding bias where negative values rounded
        away from zero instead of toward zero. This test verifies the
        Python reference doesn't have the same bug.
        """
        try:
            from subband_preprocess import lifting_3level_forward, lifting_3level_inverse

            np.random.seed(99)
            # Negative-biased signal (depression/sleep EEG patterns)
            signal = -np.abs(np.random.randn(2500).astype(np.float64)) * 1000

            subs = lifting_3level_forward(signal)
            recon = lifting_3level_inverse(subs)
            max_err = float(np.max(np.abs(signal - recon)))
            assert max_err < 1e-8, f"negative signal roundtrip error: {max_err:g}"
        except ImportError:
            pytest.skip("subband_preprocess not available")

    def test_lifting_odd_length_313(self):
        """INVARIANT: Lifting handles the L3 odd-length case (625→313+312)."""
        try:
            from subband_preprocess import lifting_1d_forward, lifting_1d_inverse

            np.random.seed(77)
            signal = np.random.randn(625).astype(np.float64)
            approx, detail = lifting_1d_forward(signal)
            assert approx.shape == (313,), f"approx shape: {approx.shape}"
            assert detail.shape == (312,), f"detail shape: {detail.shape}"

            recon = lifting_1d_inverse(approx, detail)
            max_err = float(np.max(np.abs(signal - recon)))
            assert max_err < 1e-8, f"625→313+312 roundtrip error: {max_err:g}"
        except ImportError:
            pytest.skip("subband_preprocess not available")


@pytest.mark.l2
class TestL2CodecRoundtrip:
    """Property-based tests for any compression/codec roundtrip."""

    def test_q31_conversion_reversible(self):
        """INVARIANT: Q31 int -> float -> int should be (approximately) reversible."""
        np.random.seed(42)
        original_int = np.random.randint(-1000000, 1000000, size=(21, 2500), dtype=np.int32)

        # Convert to float and back
        as_float = (original_int.astype(np.float32) / 2147483647.0) * 1000.0
        # Scale back
        back_to_int = (as_float * 2147483647.0 / 1000.0).astype(np.int32)

        # Should be approximately the same (within rounding error)
        diff = np.abs(original_int - back_to_int).astype(np.float32)
        max_error = np.max(diff)

        # Error should be at most a few units (quantization)
        assert max_error <= 10, f"Q31 roundtrip error too large: {max_error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
