"""Integration test: SNN activity detector → codec encode → compress.

Verifies the full adaptive SNAC pipeline:
  MambaSNN.classify_per_timestep() → encode(snn=snn) → compress()
  produces a valid LMQ3 adaptive packet with per-timestep FSQ levels.
"""
import sys
import os
import struct
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'ai_models', 'snn'))


@pytest.fixture
def mamba_snn():
    """Small MambaSNN for testing (not production-sized)."""
    from mamba_ssm_minimal import MambaSNN
    return MambaSNN(in_channels=21, d_model=20, d_state=8, n_layers=1, use_subband=True)


@pytest.fixture
def mock_encoder():
    """Mock TNN encoder that produces [1, 32, 79] latent."""
    class _Encoder(torch.nn.Module):
        def encode(self, x):
            torch.manual_seed(42)
            return torch.randn(1, 32, 79)
    return _Encoder()


@pytest.fixture
def subband():
    """Synthetic SubbandDecomposition."""
    from lamquant_codec.codec_types import SubbandDecomposition
    np.random.seed(42)
    return SubbandDecomposition(
        l3_approx=np.random.randn(21, 313).astype(np.float32),
        l1_detail=np.zeros((21, 1250)),
        l2_detail=np.zeros((21, 625)),
        l3_detail=np.zeros((21, 313)),
        lpc_coeffs=np.zeros((21, 4), dtype=np.float32),
    )


class TestSNNClassification:
    """Test MambaSNN.classify_per_timestep produces valid FSQ levels."""

    def test_output_shape(self, mamba_snn):
        x = torch.randn(1, 21, 313)
        levels = mamba_snn.classify_per_timestep(x, target_T=79)
        assert levels.shape == (1, 79), f"Expected (1, 79), got {levels.shape}"

    def test_output_values_are_valid_fsq_levels(self, mamba_snn):
        x = torch.randn(1, 21, 313)
        levels = mamba_snn.classify_per_timestep(x, target_T=79)
        unique = set(levels[0].tolist())
        assert unique.issubset({2, 3, 5}), f"Unexpected FSQ levels: {unique}"

    def test_batch_dimension(self, mamba_snn):
        x = torch.randn(4, 21, 313)
        levels = mamba_snn.classify_per_timestep(x, target_T=79)
        assert levels.shape == (4, 79)

    def test_deterministic_with_eval(self, mamba_snn):
        mamba_snn.eval()
        x = torch.randn(1, 21, 313)
        l1 = mamba_snn.classify_per_timestep(x, target_T=79)
        l2 = mamba_snn.classify_per_timestep(x, target_T=79)
        assert torch.equal(l1, l2), "classify_per_timestep should be deterministic in eval mode"


class TestEncodeWithSNN:
    """Test encode() with snn= parameter produces adaptive FSQ levels."""

    def test_encode_without_snn_has_no_levels(self, subband, mock_encoder):
        from lamquant_codec.encode import encode
        tokens = encode(subband, mock_encoder)
        assert tokens.fsq_levels is None

    def test_encode_with_snn_has_levels(self, subband, mock_encoder, mamba_snn):
        from lamquant_codec.encode import encode
        tokens = encode(subband, mock_encoder, snn=mamba_snn)
        assert tokens.fsq_levels is not None
        assert len(tokens.fsq_levels) == 79, f"Expected 79 levels, got {len(tokens.fsq_levels)}"

    def test_encode_levels_are_valid(self, subband, mock_encoder, mamba_snn):
        from lamquant_codec.encode import encode
        tokens = encode(subband, mock_encoder, snn=mamba_snn)
        unique = set(tokens.fsq_levels)
        assert unique.issubset({2, 3, 5}), f"Invalid FSQ levels: {unique}"

    def test_explicit_levels_override_snn(self, subband, mock_encoder, mamba_snn):
        from lamquant_codec.encode import encode
        explicit = [3] * 79
        tokens = encode(subband, mock_encoder, snn=mamba_snn, fsq_levels=explicit)
        assert tokens.fsq_levels == explicit, "Explicit fsq_levels should override SNN"


class TestAdaptiveCompress:
    """Test compress() routes to LMQ3 for adaptive schedules."""

    def test_uniform_levels_use_lmq1(self, subband, mock_encoder):
        from lamquant_codec.encode import encode
        from lamquant_codec.compress import compress
        tokens = encode(subband, mock_encoder)
        packet = compress(tokens)
        assert packet.data[:4] != b'LMQ3', "Uniform levels should use LMQ1, not LMQ3"
        assert packet.mode == 'neural'

    def test_adaptive_levels_use_lmq3(self, subband, mock_encoder, mamba_snn):
        from lamquant_codec.encode import encode
        from lamquant_codec.compress import compress
        tokens = encode(subband, mock_encoder, snn=mamba_snn)
        # Only routes to LMQ3 if levels are non-uniform
        if len(set(tokens.fsq_levels)) > 1:
            packet = compress(tokens)
            assert packet.data[:4] == b'LMQ3', \
                f"Adaptive levels should produce LMQ3, got {packet.data[:4]!r}"
            assert packet.mode == 'neural_adaptive'
            assert packet.metadata.get('adaptive') is True

    def test_lmq3_header_structure(self, subband, mock_encoder, mamba_snn):
        from lamquant_codec.encode import encode
        from lamquant_codec.compress import compress
        tokens = encode(subband, mock_encoder, snn=mamba_snn)
        if len(set(tokens.fsq_levels)) <= 1:
            pytest.skip("SNN produced uniform levels — no LMQ3 to test")
        packet = compress(tokens)
        data = packet.data
        # Parse LMQ3 header (25 bytes)
        magic, n_runs, lat_dim, lat_T, vmin, vmax, rans_len, lpc_len = \
            struct.unpack('<4sBHHffII', data[:25])
        assert magic == b'LMQ3'
        assert lat_dim == 32, f"Expected latent_dim=32, got {lat_dim}"
        assert lat_T == 79, f"Expected latent_T=79, got {lat_T}"
        assert n_runs >= 1, "Must have at least 1 run in schedule"
        assert rans_len > 0, "rANS payload must be non-empty"

    def test_adaptive_smaller_than_uniform(self, subband, mock_encoder, mamba_snn):
        """Adaptive compression should generally produce smaller packets
        than uniform when there are quiet regions (L=2 uses fewer bits)."""
        from lamquant_codec.encode import encode
        from lamquant_codec.compress import compress
        tokens_uniform = encode(subband, mock_encoder)
        tokens_adaptive = encode(subband, mock_encoder, snn=mamba_snn)
        if len(set(tokens_adaptive.fsq_levels)) <= 1:
            pytest.skip("SNN produced uniform levels")
        p_uniform = compress(tokens_uniform)
        p_adaptive = compress(tokens_adaptive)
        # Adaptive should be smaller or comparable (not always smaller
        # due to schedule overhead, but generally true for mixed activity)
        ratio = len(p_adaptive.data) / max(len(p_uniform.data), 1)
        assert ratio < 2.0, (
            f"Adaptive packet ({len(p_adaptive.data)} bytes) should not be "
            f"2x larger than uniform ({len(p_uniform.data)} bytes)")


class TestProductionWireFormatE2E:
    """Track A.6 — end-to-end wire-format roundtrips that bind the
    production Encoder / NeuralWriter / LMQReader / typed decompress
    surfaces to the LMQ3 adaptive contract.
    """

    def test_lmq3_roundtrip_via_typed_pipeline(self, subband, mock_encoder, mamba_snn):
        """encode(snn=..) → compress → decompress preserves the full
        per-timestep FSQ schedule. fsq_levels[0] still works for legacy
        readers that only sample t=0.
        """
        from lamquant_codec.encode import encode
        from lamquant_codec.compress import compress
        from lamquant_codec.decompress import decompress
        tokens = encode(subband, mock_encoder, snn=mamba_snn)
        assert tokens.fsq_levels is not None
        if len(set(tokens.fsq_levels)) <= 1:
            pytest.skip("SNN produced uniform schedule — no LMQ3 path exercised.")
        packet = compress(tokens, subband, quality_mode=2)
        assert packet.data[:4] == b'LMQ3'
        roundtrip = decompress(packet)
        # `==` on lists enforces both equal length and equal contents,
        # so a single assertion covers length + per-element roundtrip.
        assert roundtrip.fsq_levels == tokens.fsq_levels, \
            "per-timestep schedule must roundtrip exactly"
        assert roundtrip.side_info['adaptive'] is True

    def test_neuralwriter_writes_adaptive_window(self, tmp_path, subband, mock_encoder, mamba_snn):
        """NeuralWriter auto-sets FLAG_ADAPTIVE_FSQ + zeroes header
        fsq_levels when the payload magic is LMQ3. LMQReader reads back
        the same bits.
        """
        from lamquant_codec.encode import encode
        from lamquant_codec.compress import compress
        from lamquant_codec.fileformat import (
            NeuralWriter, LMQReader, FLAG_ADAPTIVE_FSQ,
        )
        tokens = encode(subband, mock_encoder, snn=mamba_snn)
        if len(set(tokens.fsq_levels)) <= 1:
            pytest.skip("SNN produced uniform schedule — no LMQ3 path exercised.")
        packet = compress(tokens, subband, quality_mode=2)
        assert packet.data[:4] == b'LMQ3'

        path = str(tmp_path / 'adaptive.lmq')
        with NeuralWriter(path) as w:
            w.write_window(packet.data, timestamp_us=12345)

        with LMQReader(path) as r:
            wins = list(r)
        assert len(wins) == 1
        win = wins[0]
        assert win.header.flags & FLAG_ADAPTIVE_FSQ, \
            "FLAG_ADAPTIVE_FSQ must be set on LMQ3 windows"
        assert win.header.fsq_levels == b'\x00' * 10, \
            "header fsq_levels MUST be zero for adaptive — schedule is in payload"
        assert win.payload[:4] == b'LMQ3'

    def test_encoder_raises_when_no_snn_pin(self, tmp_path, monkeypatch):
        """Encoder(adaptive=True) with placeholder registry pin and no
        explicit --snn-checkpoint MUST raise AdaptiveFSQError — silent
        fallback to LMQ1 is forbidden.

        Lazy-import contract: Encoder._ensure_codec does
        `from lamquant_codec.models.snn import resolve_production_snn`
        INSIDE the method, so each call freshly looks up the attribute
        on the module. Monkeypatching the module attribute therefore
        takes effect even though imports happen at call time. If a
        future refactor hoists the import to module scope, this test
        will need `monkeypatch.setattr` on the importing module as well.
        """
        from lamquant_codec.fileformat import Encoder
        from lamquant_codec.errors import AdaptiveFSQError
        import lamquant_codec.models.snn as snn_mod

        monkeypatch.setattr(snn_mod, 'resolve_production_snn', lambda: None)

        signal = np.random.randn(21, 2500).astype(np.float32) * 50
        encoder = Encoder(adaptive=True)
        with pytest.raises(AdaptiveFSQError, match="no SNN is available"):
            encoder.encode(signal)

    def test_encoder_opt_out_via_adaptive_false(self, monkeypatch):
        """adaptive=False short-circuits SNN resolution entirely. The
        production student_subband checkpoint produces an LMQ1 packet
        even with the placeholder registry pin, and the resolver MUST
        NOT be called (defensive — confirms the opt-out is structural,
        not just a swallowed AdaptiveFSQError).
        """
        from lamquant_codec.fileformat import Encoder
        from pathlib import Path
        import lamquant_codec.models.snn as snn_mod

        ckpt = Path('/mnt/4tb/LamQuant/weights/student_subband.ckpt')
        if not ckpt.is_file():
            pytest.skip(f"production checkpoint {ckpt} missing — skip "
                        f"opt-out smoke (CI image without weights/).")

        def _boom():
            raise AssertionError(
                "resolve_production_snn must NOT be called when adaptive=False")
        monkeypatch.setattr(snn_mod, 'resolve_production_snn', _boom)

        signal = np.random.randn(21, 2500).astype(np.float32) * 50
        encoder = Encoder(checkpoint=str(ckpt), adaptive=False)
        payload, levels = encoder.encode(signal)
        assert payload[:4] == b'LMQ1', f"adaptive=False must emit LMQ1, got {payload[:4]!r}"
        assert levels == b'\x02' * 10


class TestHiPPOInitialization:
    """Verify HiPPO-LegS A_log initialization in SelectiveSSM."""

    def test_a_log_hippo_values(self):
        from mamba_ssm_minimal import SelectiveSSM
        ssm = SelectiveSSM(d_model=40, d_state=16)
        # HiPPO-LegS: A_n = n + 0.5, so A_log = log(n + 0.5)
        expected = torch.log(torch.arange(1, 17, dtype=torch.float32) + 0.5)
        actual = ssm.A_log[0]  # first row (all rows should be identical at init)
        assert torch.allclose(actual, expected, atol=1e-6), \
            f"A_log should be HiPPO-LegS init. Expected {expected[:4]}, got {actual[:4]}"

    def test_a_log_shape(self):
        from mamba_ssm_minimal import SelectiveSSM
        ssm = SelectiveSSM(d_model=40, d_state=16, expand=2)
        assert ssm.A_log.shape == (80, 16), f"Expected (80, 16), got {ssm.A_log.shape}"
