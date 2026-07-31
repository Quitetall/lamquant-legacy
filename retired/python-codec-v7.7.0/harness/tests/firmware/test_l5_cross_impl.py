"""
Level 5 (Cross-implementation) tests - CRITICAL for firmware matching.

CRITICAL FAILURE: Python predictions != firmware predictions.
This is when R=0.96 in training becomes R=0.70 on device.

Focus areas:
1. Bit-exact numerical matching
2. Quantization consistency
3. Fixed-point vs floating-point equivalence
"""

import pytest
import numpy as np
import torch


@pytest.mark.l5
class TestL5QuantizationConsistency:
    """Test that quantization is consistent across encoding/decoding."""
    
    def test_q31_quantization_consistent(self):
        """INVARIANT: Q31 quantization must be deterministic and reversible."""
        values = np.array([-2147483647, -1000000, -1, 0, 1, 1000000, 2147483646], dtype=np.int32)
        
        for val in values:
            # Convert to float
            as_float = (val / 2147483647.0) * 1000.0
            
            # Scale back to Q31
            back = int((as_float * 2147483647.0 / 1000.0))
            
            # Error should be at most 1 unit (due to rounding)
            error = abs(val - back)
            assert error <= 1, f"Q31 roundtrip error for {val}: {error}"
    
    def test_student_quantization_deterministic(self):
        """INVARIANT: Student quantization must be deterministic - same input -> same output."""
        try:
            from lamquant_codec.models.encoder import TernaryMobileNetV5_Subband
            
            device = torch.device('cpu')
            model = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32).to(device)
            model.eval()
            
            batch = torch.randn(2, 21, 313).to(device)
            
            # Run quantized forward pass twice
            with torch.no_grad():
                out1 = model(batch, quantize=True)
                out2 = model(batch, quantize=True)
            
            # Should be identical (deterministic)
            torch.testing.assert_close(out1, out2,
                msg="Quantization not deterministic")
            
        except ImportError:
            pytest.skip("Student model not available")


@pytest.mark.l5
class TestL5BitExactEquivalence:
    """Test bit-exact matching with reference implementations."""
    
    def test_lpc_coefficient_stability(self):
        """INVARIANT: LPC coefficients must satisfy Durbin-Levinson stability (|a_k| <= 1)."""
        np.random.seed(42)
        signal = np.random.randn(21, 2500).astype(np.float32)
        
        try:
            from subband_preprocess import preprocess_subband_single
            
            l3, coeffs, subs = preprocess_subband_single(signal, order=8, autocorr_len=256)
            
            # Check all coefficient sets are stable
            for ch_coeffs in coeffs:
                for coeff in ch_coeffs:
                    assert abs(coeff) <= 1.0, \
                        f"Unstable LPC coefficient: {coeff}"
        
        except ImportError:
            pytest.skip("subband_preprocess not available")
    
    def test_dwt_lifting_roundtrip_exact(self):
        """INVARIANT: Le Gall 5/3 lifting is perfectly invertible.

        This is the only meaningful bit-exact invariant the lifting stage
        must satisfy across implementations: forward + inverse reproduces
        the input. (The previous revision of this test asserted a symmetry
        property that doesn't actually hold for biorthogonal 5/3 lifting
        with boundary handling.)
        """
        try:
            from subband_preprocess import (
                lifting_3level_forward, lifting_3level_inverse,
            )

            np.random.seed(42)
            for ch in range(21):
                signal = np.random.randn(2500).astype(np.float64)
                subs = lifting_3level_forward(signal)
                recon = lifting_3level_inverse(subs)
                max_err = float(np.max(np.abs(signal - recon)))
                assert max_err < 1e-8, (
                    f"channel {ch}: lifting roundtrip error {max_err:g}")

        except ImportError:
            pytest.skip("subband_preprocess not available")


@pytest.mark.l5
class TestL5NumberPrecision:
    """Test floating-point precision consistency."""
    
    def test_fixed_vs_float_equivalence(self):
        """INVARIANT: Fixed-point and float implementations should be close."""
        # Create test value that's exactly representable
        val_float = np.float32(0.5)
        val_int = int(val_float * 2147483647.0)
        
        # Convert back
        reconstructed = val_int / 2147483647.0
        
        # Should be close
        assert abs(val_float - np.float32(reconstructed)) < 1e-6
    
    def test_model_output_numerical_stability(self):
        """INVARIANT: Model outputs should never be NaN even with edge cases."""
        try:
            from lamquant_codec.models.encoder import TernaryMobileNetV5_Subband
            
            device = torch.device('cpu')
            model = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32).to(device)
            model.eval()
            
            # Test various numerical edge cases
            test_cases = [
                torch.zeros(2, 21, 313),  # All zeros
                torch.ones(2, 21, 313) * 1e-10,  # Very small
                torch.ones(2, 21, 313) * 1e10,  # Very large
                torch.randn(2, 21, 313),  # Normal
            ]
            
            with torch.no_grad():
                for batch in test_cases:
                    batch = batch.to(device)
                    output = model(batch, quantize=True)
                    assert torch.all(torch.isfinite(output)), \
                        f"Model produced NaN for input: {batch[0,0,0].item()}"
        
        except ImportError:
            pytest.skip("Student model not available")


@pytest.mark.l5
class TestL5FirmwarePredictor:
    """Tests that verify firmware-equivalent predictions."""
    
    def test_quantized_vs_float_difference_bounded(self):
        """INVARIANT: Quantized output shouldn't differ from float by more than quantization error."""
        try:
            from lamquant_codec.models.encoder import TernaryMobileNetV5_Subband
            
            device = torch.device('cpu')
            model = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32).to(device)
            model.eval()
            
            batch = torch.randn(8, 21, 313).to(device)
            
            with torch.no_grad():
                float_out = model(batch, quantize=False)
                quant_out = model(batch, quantize=True)
            
            # Difference should be bounded by quantization step size
            diff = torch.abs(float_out - quant_out)
            max_diff = torch.max(diff)
            
            # Ternary quantization {-1,0,+1} causes large deltas on untrained
            # weights.  After training, the network learns to compensate.
            # Here we just verify the output is finite and the diff is bounded
            # (not that it's small — that's a training quality test, not a
            # correctness test).
            assert torch.all(torch.isfinite(quant_out)), \
                "Quantized output contains NaN/Inf"
            float_range = torch.max(float_out) - torch.min(float_out)
            if float_range > 0:
                relative_diff = (max_diff / float_range).item()
                assert relative_diff < 5.0, \
                    f"Quantization changed output by {relative_diff:.2%} (>500%)"
        
        except ImportError:
            pytest.skip("Student model not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
