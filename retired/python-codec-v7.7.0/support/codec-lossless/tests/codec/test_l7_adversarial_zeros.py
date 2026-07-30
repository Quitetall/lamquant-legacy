"""
Level 7 Adversarial tests - specifically for all-zeros and dead channel cases.
(This test content was partially included in test_l7_adversarial_saturation.py)
"""

import pytest  # decomp(lossless-carve): skip when ai_models absent
pytest.importorskip("subband_preprocess", reason="Neural-coupled test; requires LamQuant-Neural sibling clone")

import pytest
import numpy as np
import torch


@pytest.mark.l7
class TestL7DeadChannels:
    """Adversarial tests for dead/silent channels (common in clinical settings)."""
    
    def test_all_channels_zero(self):
        """Adversarial: All 21 channels silent (complete electrode failure)."""
        eeg = np.zeros((21, 2500), dtype=np.int32)
        
        try:
            from subband_preprocess import preprocess_subband_single
            
            l3, coeffs, subs = preprocess_subband_single(eeg, order=8, autocorr_len=256)
            
            assert l3.shape == (21, 313)
            assert np.all(np.isfinite(l3)), "All-zero input produced NaN"
            assert np.allclose(l3, 0, atol=1e-10), "All-zero output should be zero"
        
        except ImportError:
            pytest.skip("subband_preprocess not available")
    
    def test_multiple_dead_channels(self):
        """Adversarial: Multiple (but not all) channels dead."""
        eeg = np.random.randint(-100000, 100000, size=(21, 2500), dtype=np.int32)
        # Kill channels 0, 5, 10, 15, 20
        for ch in [0, 5, 10, 15, 20]:
            eeg[ch, :] = 0
        
        try:
            from subband_preprocess import preprocess_subband_single
            
            l3, _, _ = preprocess_subband_single(eeg, order=8, autocorr_len=256)
            
            assert np.all(np.isfinite(l3)), "Multiple dead channels caused NaN"
            
            # Dead channels should have zero L3
            for ch in [0, 5, 10, 15, 20]:
                assert np.allclose(l3[ch, :], 0), f"Dead channel {ch} not zero in L3"
            
            # Alive channels should have energy
            for ch in [1, 2, 3]:
                assert not np.allclose(l3[ch, :], 0), f"Live channel {ch} lost energy"
        
        except ImportError:
            pytest.skip("subband_preprocess not available")
    
    def test_alternating_dead_alive_pattern(self):
        """Adversarial: Alternating pattern (odd channels dead, even alive)."""
        eeg = np.zeros((21, 2500), dtype=np.int32)
        for ch in range(21):
            if ch % 2 == 0:  # Even channels: alive
                eeg[ch, :] = np.random.randint(-100000, 100000, size=2500, dtype=np.int32)
            # Odd channels: dead (stay zero)
        
        try:
            from subband_preprocess import preprocess_subband_single
            
            l3, _, _ = preprocess_subband_single(eeg, order=8, autocorr_len=256)
            
            assert np.all(np.isfinite(l3)), "Alternating pattern caused NaN"
            
            # Check the pattern is preserved in L3
            for ch in range(21):
                if ch % 2 == 1:  # Should be dead
                    assert np.allclose(l3[ch, :], 0), f"Odd channel {ch} should be zero"
        
        except ImportError:
            pytest.skip("subband_preprocess not available")
    
    def test_dead_channel_temporal_pattern(self):
        """Adversarial: Channel silent during first half, active in second."""
        eeg = np.random.randint(-100000, 100000, size=(21, 2500), dtype=np.int32)
        eeg[:, :1250] = 0  # First half silent
        # Second half has signal (already has it from random)
        
        try:
            from subband_preprocess import preprocess_subband_single
            
            l3, _, _ = preprocess_subband_single(eeg, order=8, autocorr_len=256)
            
            assert np.all(np.isfinite(l3)), "Temporal dead pattern caused NaN"
        
        except ImportError:
            pytest.skip("subband_preprocess not available")


@pytest.mark.l7
class TestL7ModelWithDeadData:
    """Test model robustness when fed preprocessing output from dead channels."""
    
    def test_model_with_all_zero_l3(self):
        """Adversarial: Model processes all-zero L3 (preprocessed dead channel)."""
        try:
            from lamquant_neural.models.encoder import TernaryMobileNetV5_Subband
            
            device = torch.device('cpu')
            model = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32).to(device)
            model.eval()
            
            batch = torch.zeros(4, 21, 313).to(device)
            
            with torch.no_grad():
                output = model(batch, quantize=True)
            
            assert torch.all(torch.isfinite(output)), "All-zero L3 caused NaN"

            # With biases (GroupNorm, conv), a randomly initialized model won't
            # output zero for zero input.  The invariant is: no NaN/Inf and
            # output is bounded (not exploding).
            assert torch.max(torch.abs(output)) < 100, \
                "Model output exploded on zero input"
        
        except ImportError:
            pytest.skip("Student model not available")
    
    def test_model_batch_with_dead_channels_mixed(self):
        """Adversarial: Some samples in batch have dead channels."""
        try:
            from lamquant_neural.models.encoder import TernaryMobileNetV5_Subband
            
            device = torch.device('cpu')
            model = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32).to(device)
            model.eval()
            
            batch = torch.randn(4, 21, 313).to(device)
            
            # Sample 0: normal
            # Sample 1: channels 5, 10 dead
            batch[1, [5, 10], :] = 0
            # Sample 2: all dead
            batch[2, :, :] = 0
            # Sample 3: normal
            
            with torch.no_grad():
                output = model(batch, quantize=True)
            
            assert output.shape == batch.shape
            assert torch.all(torch.isfinite(output)), \
                "Mixed dead/alive batch caused NaN"
        
        except ImportError:
            pytest.skip("Student model not available")


@pytest.mark.l7
class TestL7VerySmallSignals:
    """Test handling of signals below typical noise floor."""
    
    def test_noise_floor_signal(self):
        """Adversarial: Signal at noise floor (1-2 units in Q31)."""
        eeg = np.random.randint(-5, 5, size=(21, 2500), dtype=np.int32)
        
        try:
            from subband_preprocess import preprocess_subband_single
            
            l3, _, _ = preprocess_subband_single(eeg, order=8, autocorr_len=256)
            
            assert np.all(np.isfinite(l3)), "Noise-floor signal produced NaN"
        
        except ImportError:
            pytest.skip("subband_preprocess not available")
    
    def test_single_spike_silence(self):
        """Adversarial: One sample at full scale, rest silent."""
        eeg = np.zeros((21, 2500), dtype=np.int32)
        eeg[:, 1250] = np.iinfo(np.int32).max // 2  # Single spike at center
        
        try:
            from subband_preprocess import preprocess_subband_single
            
            l3, _, _ = preprocess_subband_single(eeg, order=8, autocorr_len=256)
            
            assert np.all(np.isfinite(l3)), "Single spike in silence produced NaN"
        
        except ImportError:
            pytest.skip("subband_preprocess not available")
    
    def test_sub_quantization_signal(self):
        """Adversarial: Signal with magnitude < 1 in original units."""
        eeg = np.ones((21, 2500), dtype=np.float32) * 0.1  # 0.1 microvolts
        eeg_int = (eeg * 2147483647.0 / 1000.0).astype(np.int32)
        
        try:
            from subband_preprocess import preprocess_subband_single
            
            l3, _, _ = preprocess_subband_single(eeg_int, order=8, autocorr_len=256)
            
            assert np.all(np.isfinite(l3)), "Sub-quantization signal produced NaN"
        
        except ImportError:
            pytest.skip("subband_preprocess not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
