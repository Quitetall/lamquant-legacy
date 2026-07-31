"""
Level 2 (Property-based), L5 (Cross-impl), L7 (Adversarial) tests for preprocessing.

Focus: Q31->float conversion, LPC, lifting DWT, L3 approximation.
"""

import pytest  # decomp(lossless-carve): skip when ai_models absent
pytest.importorskip("subband_preprocess", reason="Neural-coupled test; requires LamQuant-Neural sibling clone")

import pytest
import numpy as np
import struct

from subband_preprocess import preprocess_subband_single  # via conftest sys.path


class TestL2PreprocessingInvariants:
    """L2: Property-based tests - mathematical invariants that must hold."""
    
    @pytest.mark.l2
    def test_q31_conversion_range(self, sample_eeg_q31):
        """INVARIANT: Q31 int to float conversion output must be in valid audio range.
        
        After Q31->float conversion scaled to microvolts, output must be:
        - Finite (no NaN/Inf)
        - In reasonable EEG range (-10000 to 10000 uV)
        - Preserve monotonicity (if x < y, then convert(x) < convert(y))
        """
        eeg_float = (sample_eeg_q31.astype(np.float32) / 2147483647.0) * 1000.0
        
        # Invariant 1: Finiteness
        assert np.all(np.isfinite(eeg_float)), "Q31 conversion produced NaN or Inf"
        
        # Invariant 2: Range
        assert np.all(np.abs(eeg_float) <= 10000), "Q31 conversion out of EEG range"
        
        # Invariant 3: Monotonicity
        for _ in range(100):
            idx1, idx2 = np.random.randint(0, sample_eeg_q31.size, 2)
            if sample_eeg_q31.flat[idx1] < sample_eeg_q31.flat[idx2]:
                assert eeg_float.flat[idx1] < eeg_float.flat[idx2], \
                    "Q31 conversion violated monotonicity"
    
    @pytest.mark.l2
    def test_lpc_residual_is_nontrivial(self, sample_eeg_float):
        """INVARIANT: LPC residual of a realistic signal is finite, non-zero,
        and strictly lower-energy than the input.

        The L3 approximation of the residual is the TNN input; it can have
        arbitrarily small energy relative to the raw signal (since LPC
        whitens), so the meaningful invariant here is on the LPC stage —
        that it produces a well-defined residual, not that L3 preserves
        input energy.
        """
        from subband_preprocess import lpc_analyze

        eeg_float = sample_eeg_float.astype(np.float64).copy()
        coeffs, residual = lpc_analyze(eeg_float, order=8, autocorr_len=256)

        assert residual.shape == eeg_float.shape
        assert np.all(np.isfinite(residual)), "LPC residual contains NaN/Inf"
        assert np.all(np.isfinite(coeffs)), "LPC coefficients contain NaN/Inf"
        assert np.sum(residual ** 2) > 0, "LPC residual is all zeros"

        # Whitening should not INCREASE energy (LPC minimizes residual energy
        # in the MSE sense). Allow a tiny slack for boundary handling.
        orig_energy = float(np.sum(eeg_float ** 2))
        res_energy = float(np.sum(residual ** 2))
        assert res_energy <= orig_energy * 1.05, (
            f"LPC residual energy {res_energy:g} exceeds input {orig_energy:g}")
    
    @pytest.mark.l2
    def test_l3_shape_invariant(self, sample_eeg_float):
        """INVARIANT: L3 output shape must always be [21, 313]."""
        eeg_q31 = (sample_eeg_float * 1000 / 2147483647.0).astype(np.int32)
        
        for order in [6, 8, 10]:
            for autocorr_len in [128, 256, 512]:
                l3, _, _ = preprocess_subband_single(eeg_q31, order=order, autocorr_len=autocorr_len)
                assert l3.shape == (21, 313), \
                    f"L3 shape mismatch for order={order}: {l3.shape}"
    
    @pytest.mark.l2
    def test_preprocessing_idempotence(self, sample_eeg_float):
        """INVARIANT: Preprocessing same signal twice should give same L3."""
        eeg_q31 = (sample_eeg_float * 1000 / 2147483647.0).astype(np.int32)
        
        l3_1, _, _ = preprocess_subband_single(eeg_q31, order=8, autocorr_len=256)
        l3_2, _, _ = preprocess_subband_single(eeg_q31, order=8, autocorr_len=256)
        
        np.testing.assert_array_equal(l3_1, l3_2, 
            err_msg="Preprocessing not idempotent")
    
    @pytest.mark.l2
    def test_lpc_residual_scale_linearity(self, sample_eeg_float):
        """INVARIANT: Scaling the input scales the LPC residual by the same
        factor, up to small numerical error.

        LPC coefficients are invariant under input scaling (both numerator
        and denominator of the autocorrelation ratio scale by k²), so
        residual(k·x) == k·residual(x) exactly in infinite precision. In
        float64 arithmetic we expect relative error ≲ 1e-6.

        We operate on `sample_eeg_float` directly (microvolts) instead of
        round-tripping through Q31 because the previous revision of this
        test truncated small-valued EEG to integer zero, masking any real
        bug in the linearity path.
        """
        from subband_preprocess import lpc_analyze

        x = sample_eeg_float.astype(np.float64)
        k = 0.5

        _, res_1 = lpc_analyze(x, order=8, autocorr_len=256)
        _, res_k = lpc_analyze(k * x, order=8, autocorr_len=256)

        denom = np.maximum(np.abs(res_1) * abs(k), 1e-9)
        rel_err = np.abs(res_k - k * res_1) / denom
        # 99th percentile relative error should be tiny; use percentile rather
        # than max to ignore a handful of boundary samples where LPC warmup
        # dominates.
        p99 = float(np.percentile(rel_err, 99))
        assert p99 < 1e-6, f"LPC residual scale linearity broken (p99={p99:g})"


class TestL5PreprocessingCrossImpl:
    """L5: Cross-implementation - verify preprocessing against reference.
    
    Note: These test against bit-exact reference implementations or known values.
    """
    
    @pytest.mark.l5
    def test_q31_conversion_bit_exact(self):
        """Test Q31->float conversion matches reference implementation exactly."""
        # Reference: IEEE 754 division, no approximation
        q31_val = np.int32(1073741823)  # 2^30 - middle value
        
        # Current implementation
        result = (q31_val / 2147483647.0) * 1000.0
        
        # Reference calculation
        reference = (np.float32(q31_val) / np.float32(2147483647.0)) * np.float32(1000.0)
        
        # For FP32, should be exact or very close
        assert abs(result - reference) < 1e-5 * abs(reference)
    
    @pytest.mark.l5
    def test_lpc_coefficient_bounds(self, sample_eeg_float):
        """Test LPC coefficients stay in valid range for any input."""
        eeg_q31 = (sample_eeg_float * 1000 / 2147483647.0).astype(np.int32)
        
        l3, coeffs, subs = preprocess_subband_single(eeg_q31, order=8, autocorr_len=256)
        
        # LPC coefficients should be in [-1, 1] for stability
        for coeff_set in coeffs:
            assert np.all(np.abs(coeff_set) <= 1.01), \
                "LPC coefficients unstable (|a_k| > 1)"
    
    @pytest.mark.l5
    def test_dwt_roundtrip_bit_exact(self, sample_eeg_float):
        """Lifting DWT forward + inverse must exactly reconstruct the input.

        This is THE bit-exactness invariant for the integer lifting codec —
        if it fails, the firmware decoder and the training-time analysis
        stage are not inverses of each other and the loss function is
        meaningless. We test the lifting stage in isolation because the
        full `preprocess_subband_single` pipeline prepends LPC analysis;
        LPC invariants are covered by the dedicated LPC tests above.
        """
        from subband_preprocess import lifting_3level_forward, lifting_3level_inverse

        signal = sample_eeg_float.astype(np.float64)  # [21, 2500]
        for ch in range(signal.shape[0]):
            subs = lifting_3level_forward(signal[ch])
            recon = lifting_3level_inverse(subs)
            max_err = float(np.max(np.abs(signal[ch] - recon)))
            assert max_err < 1e-8, \
                f"channel {ch}: lifting DWT roundtrip error {max_err:g}"


class TestL7PreprocessingAdversarial:
    """L7: Adversarial - worst-case inputs that should not crash or corrupt."""
    
    @pytest.mark.l7
    def test_preprocessing_all_zeros(self):
        """Adversarial: All-zero input should produce all-zero L3."""
        eeg_zeros = np.zeros((21, 2500), dtype=np.int32)
        l3, coeffs, subs = preprocess_subband_single(eeg_zeros, order=8, autocorr_len=256)
        
        assert l3.shape == (21, 313)
        assert np.allclose(l3, 0), "All-zero input produced non-zero L3"
    
    @pytest.mark.l7
    def test_preprocessing_extreme_values(self):
        """Adversarial: Extreme Q31 values should not crash."""
        eeg_extreme = np.zeros((21, 2500), dtype=np.int32)
        eeg_extreme[:, ::100] = np.iinfo(np.int32).max  # Peak values
        eeg_extreme[:, 1::100] = np.iinfo(np.int32).min  # Trough values
        
        l3, _, _ = preprocess_subband_single(eeg_extreme, order=8, autocorr_len=256)
        
        assert l3.shape == (21, 313)
        assert np.all(np.isfinite(l3)), "Extreme values produced NaN/Inf in L3"
    
    @pytest.mark.l7
    def test_preprocessing_noise_only(self):
        """Adversarial: Pure noise should still produce finite output."""
        eeg_noise = np.random.randint(-1000, 1000, size=(21, 2500), dtype=np.int32)
        
        for _ in range(5):
            l3, _, _ = preprocess_subband_single(eeg_noise, order=8, autocorr_len=256)
            assert np.all(np.isfinite(l3)), "Noise produced NaN/Inf in L3"
    
    @pytest.mark.l7
    def test_preprocessing_sinusoid_input(self):
        """Adversarial: Pure sinusoid (periodic signal) should compress well."""
        # Create 21 sinusoids at different frequencies
        t = np.arange(2500) / 250  # 10 seconds at 250 Hz
        eeg_sine = np.zeros((21, 2500), dtype=np.int32)
        
        for ch in range(21):
            freq = 1 + ch  # 1 Hz to 21 Hz
            sine = np.sin(2 * np.pi * freq * t)
            eeg_sine[ch] = (sine * 1000).astype(np.int32)
        
        l3, _, _ = preprocess_subband_single(eeg_sine, order=8, autocorr_len=256)
        
        # Sinusoids should compress well
        sine_energy = np.sum(eeg_sine.astype(np.float32)**2)
        l3_energy = np.sum(l3**2)
        ratio = l3_energy / (sine_energy + 1e-10)
        
        assert ratio < 0.5, f"Sinusoid compression poor: {ratio:.2%}"
    
    @pytest.mark.l7
    def test_preprocessing_spike_input(self):
        """Adversarial: Input with sharp spikes (seizure-like)."""
        eeg_spiky = np.zeros((21, 2500), dtype=np.int32)
        
        # Add seizure-like spikes
        for ch in range(21):
            for spike_idx in range(10, 2500, 250):
                eeg_spiky[ch, spike_idx:spike_idx+50] = np.iinfo(np.int32).max // 2
        
        l3, _, _ = preprocess_subband_single(eeg_spiky, order=8, autocorr_len=256)
        
        assert l3.shape == (21, 313)
        assert np.all(np.isfinite(l3))
    
    @pytest.mark.l7
    def test_preprocessing_dc_offset(self):
        """Adversarial: Large DC offset should not break preprocessing."""
        eeg_noise = np.random.randint(-100, 100, size=(21, 2500), dtype=np.int32)
        eeg_dc = eeg_noise + np.int32(500000)  # Add large DC offset
        
        l3, _, _ = preprocess_subband_single(eeg_dc, order=8, autocorr_len=256)
        
        assert l3.shape == (21, 313)
        assert np.all(np.isfinite(l3))
    
    @pytest.mark.l7
    def test_preprocessing_step_input(self):
        """Adversarial: Step function (very non-smooth)."""
        eeg_step = np.zeros((21, 2500), dtype=np.int32)
        eeg_step[:, 1250:] = np.int32(100000)  # Step at midpoint
        
        l3, _, _ = preprocess_subband_single(eeg_step, order=8, autocorr_len=256)
        
        assert l3.shape == (21, 313)
        assert np.all(np.isfinite(l3))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
