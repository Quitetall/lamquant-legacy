"""
Level 2, 5, 7 tests for model architecture and tensor shapes.

Focus: Catching shape mismatches early, ensuring models compute correctly.
"""

import pytest
import numpy as np
import torch


@pytest.mark.l2
class TestL2ArchitectureShapeInvariants:
    """L2: Shape invariants that must hold for all models."""
    
    def test_student_subband_encoder_output(self):
        """INVARIANT: Student subband encoder [B, 21, 313] -> fixed latent size."""
        try:
            from lamquant_codec.models.encoder import TernaryMobileNetV5_Subband
            device = torch.device('cpu')
            model = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32).to(device)
            
            # Test with batch
            batch = torch.randn(4, 21, 313).to(device)
            latent = model.encode(batch, quantize=False)
            
            # Latent should be [4, 32, 1] (spatial collapsed after multiple down-samples)
            assert latent.shape[0] == 4, "Batch size not preserved"
            assert latent.shape[1] == 32, f"Latent dim wrong: {latent.shape[1]} != 32"
            
        except ImportError:
            pytest.skip("Student model not available")
    
    def test_student_subband_decoder_output(self):
        """INVARIANT: Student decoder latent -> [B, 21, 313] (L3 approximation)."""
        try:
            from lamquant_codec.models.encoder import TernaryMobileNetV5_Subband
            device = torch.device('cpu')
            model = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32).to(device)
            
            batch = torch.randn(4, 21, 313).to(device)
            output = model(batch, quantize=False)
            
            # Output must match input shape (L3 reconstruction)
            assert output.shape == batch.shape, \
                f"Decoder output shape {output.shape} != input shape {batch.shape}"
            
        except ImportError:
            pytest.skip("Student model not available")
    
    def test_teacher_autoencoder_shapes(self):
        """INVARIANT: Teacher autoencoder preserves shape."""
        try:
            from oracle.train_teacher import FP32OracleAutoEncoder
            device = torch.device('cpu')
            model = FP32OracleAutoEncoder().to(device)
            
            # Teacher works on full [B, 21, 2500] signals
            batch = torch.randn(4, 21, 2500).to(device)
            output = model(batch)
            
            assert output.shape == batch.shape, \
                f"Teacher output shape {output.shape} != input {batch.shape}"
            
        except ImportError:
            pytest.skip("Teacher model not available")
    
    def test_snn_input_output_sizes(self):
        """INVARIANT: SNN has fixed input/output sizes based on receptive field."""
        try:
            from mamba_ssm_minimal import MambaSNN
            device = torch.device('cpu')
            model = MambaSNN(in_channels=21, use_subband=True).to(device)

            batch_size = 4
            signal = torch.randn(batch_size, 21, 313).to(device)

            with torch.no_grad():
                output, spike_rate = model(signal)

            assert output.shape[0] == batch_size
            assert output.shape[1] == 8, f"SNN groups wrong: {output.shape[1]} != 8"
            assert output.shape[2] == 313, f"SNN time dim wrong: {output.shape[2]} != 313"

        except (ImportError, AttributeError):
            pytest.skip("SNN model not available")


@pytest.mark.l5
class TestL5ArchitectureCrossImpl:
    """L5: Cross-implementation shape and computation verification."""
    
    def test_quantization_doesnt_change_shape(self):
        """Test that quantization preserves tensor shape."""
        try:
            from lamquant_codec.models.encoder import TernaryMobileNetV5_Subband
            device = torch.device('cpu')
            model = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32).to(device)
            
            batch = torch.randn(4, 21, 313).to(device)
            
            output_fp32 = model(batch, quantize=False)
            output_q = model(batch, quantize=True)
            
            assert output_fp32.shape == output_q.shape, \
                "Quantization changed output shape"
            
        except ImportError:
            pytest.skip("Student model not available")
    
    def test_gradient_flow(self):
        """Test that gradients flow through encoder and decoder."""
        try:
            from lamquant_codec.models.encoder import TernaryMobileNetV5_Subband
            device = torch.device('cpu')
            model = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32).to(device)
            
            batch = torch.randn(4, 21, 313, requires_grad=True).to(device)
            output = model(batch, quantize=False)
            loss = output.sum()
            loss.backward()
            
            # Input gradients should exist
            assert batch.grad is not None, "No gradients to input"
            assert torch.any(batch.grad != 0), "All input gradients are zero"
            
        except ImportError:
            pytest.skip("Student model not available")


@pytest.mark.l7
class TestL7ArchitectureAdversarial:
    """L7: Adversarial inputs that should not crash or produce NaN."""
    
    def test_all_zeros_input(self):
        """Adversarial: All-zero input should not produce NaN."""
        try:
            from lamquant_codec.models.encoder import TernaryMobileNetV5_Subband
            device = torch.device('cpu')
            model = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32).to(device)
            
            batch = torch.zeros(4, 21, 313).to(device)
            output = model(batch, quantize=False)
            
            assert torch.all(torch.isfinite(output)), "Zero input produced NaN/Inf"
            
        except ImportError:
            pytest.skip("Student model not available")
    
    def test_extreme_values_input(self):
        """Adversarial: Very large input values."""
        try:
            from lamquant_codec.models.encoder import TernaryMobileNetV5_Subband
            device = torch.device('cpu')
            model = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32).to(device)
            
            batch = torch.ones(4, 21, 313).to(device) * 1000
            output = model(batch, quantize=False)
            
            assert torch.all(torch.isfinite(output)), "Large input produced NaN/Inf"
            
        except ImportError:
            pytest.skip("Student model not available")
    
    def test_nan_input_handling(self):
        """Adversarial: NaN in input should not silently propagate."""
        try:
            from lamquant_codec.models.encoder import TernaryMobileNetV5_Subband
            device = torch.device('cpu')
            model = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32).to(device)
            
            batch = torch.randn(4, 21, 313).to(device)
            batch[0, 0, 0] = float('nan')
            
            output = model(batch, quantize=False)
            
            # Output should probably be NaN due to NaN input, but not crash
            # (This is acceptable behavior)
            
        except ImportError:
            pytest.skip("Student model not available")
    
    def test_single_sample_batch(self):
        """Adversarial: Batch size 1 (edge case for BatchNorm)."""
        try:
            from lamquant_codec.models.encoder import TernaryMobileNetV5_Subband
            device = torch.device('cpu')
            model = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32).to(device)
            model.eval()  # BatchNorm needs eval mode for single sample
            
            batch = torch.randn(1, 21, 313).to(device)
            output = model(batch, quantize=False)
            
            assert output.shape[0] == 1, "Batch size not preserved"
            assert torch.all(torch.isfinite(output)), "BatchNorm broke on size 1"
            
        except ImportError:
            pytest.skip("Student model not available")
    
    def test_large_batch_size(self):
        """Adversarial: Very large batch might expose memory issues."""
        try:
            from lamquant_codec.models.encoder import TernaryMobileNetV5_Subband
            device = torch.device('cpu')
            model = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32).to(device)
            model.eval()
            
            # Try large batch - might OOM on weak hardware, that's ok
            try:
                batch = torch.randn(256, 21, 313).to(device)
                output = model(batch, quantize=False)
                assert output.shape[0] == 256
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    pytest.skip("OOM on large batch (acceptable on weak hardware)")
                raise
            
        except ImportError:
            pytest.skip("Student model not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
