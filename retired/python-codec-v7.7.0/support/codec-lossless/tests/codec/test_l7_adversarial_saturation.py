"""
Level 7 (Adversarial) tests - Edge cases that cause NaN in production.

CRITICAL EDGE CASES:
1. Saturated ADC input (high-amplitude seizure)
2. All-zeros input (dead channel)
3. Mixed normal + saturated (some channels good, some bad)

These edge cases happen in real patients and must not crash/produce NaN.
"""

import pytest  # decomp(lossless-carve): skip when ai_models absent
pytest.importorskip("subband_preprocess", reason="Neural-coupled test; requires LamQuant-Neural sibling clone")

import pytest
import numpy as np
import torch


@pytest.mark.l7
class TestL7AdversarialSaturation:
    """Test handling of saturated/clipped ADC input."""
    
    def test_saturated_adc_q31_max(self):
        """Adversarial: Input saturated at Q31 max (high seizure amplitude)."""
        # Create partially saturated signal (seizure with high spikes)
        eeg = np.ones((21, 2500), dtype=np.int32) * 100000  # Normal level
        
        # Add seizure spikes that saturate
        eeg[:, 1000:1100] = np.iinfo(np.int32).max // 2  # Saturate some samples
        
        try:
            from subband_preprocess import preprocess_subband_single
            
            l3, coeffs, subs = preprocess_subband_single(eeg, order=8, autocorr_len=256)
            
            assert l3.shape == (21, 313), f"Shape changed: {l3.shape}"
            assert np.all(np.isfinite(l3)), "Saturated input produced NaN in L3"
            assert not np.allclose(l3, 0), "Saturated input produced all-zeros L3"
        
        except ImportError:
            pytest.skip("subband_preprocess not available")
    
    def test_alternating_saturation(self):
        """Adversarial: Signal alternates between min/max (extreme clipping)."""
        eeg = np.zeros((21, 2500), dtype=np.int32)
        eeg[:, ::2] = np.iinfo(np.int32).max // 2
        eeg[:, 1::2] = np.iinfo(np.int32).min // 2
        
        try:
            from subband_preprocess import preprocess_subband_single
            
            l3, _, _ = preprocess_subband_single(eeg, order=8, autocorr_len=256)
            
            assert np.all(np.isfinite(l3)), "Alternating saturation produced NaN"
            assert not np.allclose(l3, 0), "Extreme clipping produced all-zeros"
        
        except ImportError:
            pytest.skip("subband_preprocess not available")
    
    def test_single_saturated_channel(self):
        """Adversarial: One channel saturated, others normal."""
        eeg = np.random.randint(-100000, 100000, size=(21, 2500), dtype=np.int32)
        eeg[5, :] = np.iinfo(np.int32).max // 2  # Channel 5 saturated
        
        try:
            from subband_preprocess import preprocess_subband_single
            
            l3, _, _ = preprocess_subband_single(eeg, order=8, autocorr_len=256)
            
            assert np.all(np.isfinite(l3)), "Single saturated channel caused NaN"
            
            # Other channels should still have energy
            for ch in range(21):
                if ch != 5:
                    assert not np.allclose(l3[ch, :], 0), \
                        f"Channel {ch} lost all energy due to ch 5 saturation"
        
        except ImportError:
            pytest.skip("subband_preprocess not available")


@pytest.mark.l7
class TestL7AdversarialAllZeros:
    """Test handling of all-zeros or near-zero input."""
    
    def test_all_zeros_signal(self):
        """Adversarial: Completely silent input (flatline EEG - electrode disconnected)."""
        eeg = np.zeros((21, 2500), dtype=np.int32)
        
        try:
            from subband_preprocess import preprocess_subband_single
            
            l3, coeffs, subs = preprocess_subband_single(eeg, order=8, autocorr_len=256)
            
            assert l3.shape == (21, 313)
            assert np.all(np.isfinite(l3)), "All-zeros produced NaN"
            assert np.allclose(l3, 0), "All-zeros should produce all-zero L3"
        
        except ImportError:
            pytest.skip("subband_preprocess not available")
    
    def test_very_small_signal(self):
        """Adversarial: Signal below noise floor (10 bits resolution)."""
        eeg = np.random.randint(-10, 10, size=(21, 2500), dtype=np.int32)
        
        try:
            from subband_preprocess import preprocess_subband_single
            
            l3, _, _ = preprocess_subband_single(eeg, order=8, autocorr_len=256)
            
            assert l3.shape == (21, 313)
            assert np.all(np.isfinite(l3)), "Noise-floor input produced NaN"
        
        except ImportError:
            pytest.skip("subband_preprocess not available")
    
    def test_single_zero_channel(self):
        """Adversarial: One channel all zeros (electrode disconnected)."""
        eeg = np.random.randint(-100000, 100000, size=(21, 2500), dtype=np.int32)
        eeg[10, :] = 0  # Channel 10 dead
        
        try:
            from subband_preprocess import preprocess_subband_single
            
            l3, _, _ = preprocess_subband_single(eeg, order=8, autocorr_len=256)
            
            assert np.all(np.isfinite(l3)), "Dead channel caused NaN in other channels"
            # Dead channel's L3 should be zero
            assert np.allclose(l3[10, :], 0), "Dead channel didn't produce zero L3"
        
        except ImportError:
            pytest.skip("subband_preprocess not available")


@pytest.mark.l7
class TestL7AdversarialModelInput:
    """Test model robustness to adversarial preprocessing outputs."""
    
    def test_model_saturated_l3_input(self):
        """Adversarial: Feed saturated L3 values to model."""
        try:
            from lamquant_neural.models.encoder import TernaryMobileNetV5_Subband
            
            device = torch.device('cpu')
            model = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32).to(device)
            model.eval()
            
            # Create L3 with extreme values
            batch = torch.ones(2, 21, 313).to(device) * 1e6
            
            with torch.no_grad():
                output = model(batch, quantize=True)
            
            assert torch.all(torch.isfinite(output)), \
                "Model failed on saturated L3 input"
        
        except ImportError:
            pytest.skip("Student model not available")
    
    def test_model_all_zeros_l3(self):
        """Adversarial: Feed all-zero L3 (silent/dead electrode)."""
        try:
            from lamquant_neural.models.encoder import TernaryMobileNetV5_Subband
            
            device = torch.device('cpu')
            model = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32).to(device)
            model.eval()
            
            batch = torch.zeros(2, 21, 313).to(device)
            
            with torch.no_grad():
                output = model(batch, quantize=True)
            
            assert torch.all(torch.isfinite(output)), \
                "Model failed on all-zero L3 input"
            # With biases, random model won't output zero; just verify bounded
            assert torch.max(torch.abs(output)) < 100, \
                "Model output exploded on zero input"
        
        except ImportError:
            pytest.skip("Student model not available")
    
    def test_model_mixed_valid_saturated(self):
        """Adversarial: Batch with mix of normal and saturated channels."""
        try:
            from lamquant_neural.models.encoder import TernaryMobileNetV5_Subband
            
            device = torch.device('cpu')
            model = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32).to(device)
            model.eval()
            
            batch = torch.randn(4, 21, 313).to(device)
            # Make some channels saturated
            batch[:, [3, 7, 12], :] = torch.ones(4, 3, 313).to(device) * 1e6
            # Make some channels zero
            batch[:, [1, 5], :] = 0
            
            with torch.no_grad():
                output = model(batch, quantize=True)
            
            assert output.shape == batch.shape
            assert torch.all(torch.isfinite(output)), \
                "Model failed on mixed valid/saturated/zero batch"
        
        except ImportError:
            pytest.skip("Student model not available")


@pytest.mark.l7
class TestL7AdversarialCornerCases:
    """Other corner cases that might cause issues."""
    
    def test_barely_nonzero(self):
        """Adversarial: Input with minimal magnitude (underflow risk)."""
        eeg = np.random.uniform(-1e-10, 1e-10, size=(21, 2500)).astype(np.float32)
        eeg_int = (eeg * 2147483647.0).astype(np.int32)
        
        try:
            from subband_preprocess import preprocess_subband_single
            
            l3, _, _ = preprocess_subband_single(eeg_int, order=8, autocorr_len=256)
            
            assert np.all(np.isfinite(l3)), "Barely-nonzero input produced NaN"
        
        except ImportError:
            pytest.skip("subband_preprocess not available")
    
    def test_dc_offset_removal(self):
        """Adversarial: Very large DC offset (signal offset to one rail)."""
        eeg = np.ones((21, 2500), dtype=np.int32) * (np.iinfo(np.int32).max // 2)
        eeg += np.random.randint(-1000, 1000, size=(21, 2500), dtype=np.int32)
        
        try:
            from subband_preprocess import preprocess_subband_single
            
            l3, _, _ = preprocess_subband_single(eeg, order=8, autocorr_len=256)
            
            assert np.all(np.isfinite(l3)), "Large DC offset caused NaN"
        
        except ImportError:
            pytest.skip("subband_preprocess not available")
    
    def test_high_frequency_input(self):
        """Adversarial: Nyquist-rate oscillation (fs/2 Hz)."""
        t = np.arange(2500)
        # Create signal oscillating at Nyquist (alternating +/-)
        eeg = np.zeros((21, 2500), dtype=np.int32)
        for ch in range(21):
            eeg[ch, ::2] = np.int32(500000)
            eeg[ch, 1::2] = np.int32(-500000)
        
        try:
            from subband_preprocess import preprocess_subband_single
            
            l3, _, _ = preprocess_subband_single(eeg, order=8, autocorr_len=256)
            
            assert np.all(np.isfinite(l3)), "Nyquist-rate input produced NaN"
        
        except ImportError:
            pytest.skip("subband_preprocess not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
