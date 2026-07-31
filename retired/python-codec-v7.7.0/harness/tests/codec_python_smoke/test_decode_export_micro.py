"""Micro-coverage tests for ``lamquant_codec.decode`` + ``lamquant_codec.export``.

Both are tiny entry-point modules. Tests pin the public-API surface
(decode returns an EEGPacket; export.main argparses without crashing).
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


# ============================================================
# decode.decode
# ============================================================

from lamquant_codec.codec_types import EEGPacket, LatentTokens
from lamquant_codec.decode import decode


def _fake_decoder():
    """Module that exposes a .decode(latent, target_len, quantize) call.

    Returns a constant zero tensor matching the documented shape. The
    actual decoder math is the production codec's responsibility; this
    test pins only the wrapper's I/O contract.
    """
    m = MagicMock()
    def _decode(latent, *, target_len, quantize):
        B = latent.shape[0]
        return torch.zeros(B, 21, target_len)
    m.decode.side_effect = _decode
    return m


class TestDecode:
    def test_returns_eeg_packet(self) -> None:
        latent = np.random.RandomState(0).randn(32, 313).astype(np.float32)
        tokens = LatentTokens(tokens=None, latent=latent,
                               snac_preset="balanced")
        pkt = decode(tokens, _fake_decoder(), target_len=313)
        assert isinstance(pkt, EEGPacket)

    def test_2d_latent_promoted_to_3d(self) -> None:
        latent = np.random.RandomState(1).randn(32, 313).astype(np.float32)
        tokens = LatentTokens(tokens=None, latent=latent,
                               snac_preset="balanced")
        pkt = decode(tokens, _fake_decoder())
        # The wrapper unsqueezes 2D inputs to 3D before the model call.
        assert pkt.signal.shape == (21, 313)

    def test_uses_tokens_when_latent_none(self) -> None:
        toks = np.random.RandomState(2).randn(32, 313).astype(np.float32)
        tokens = LatentTokens(tokens=toks, latent=None,
                               snac_preset="balanced")
        pkt = decode(tokens, _fake_decoder())
        assert pkt.signal.shape == (21, 313)

    def test_compressed_bytes_recorded(self) -> None:
        latent = np.random.RandomState(3).randn(32, 313).astype(np.float32)
        tokens = LatentTokens(tokens=None, latent=latent,
                               snac_preset="fast")
        pkt = decode(tokens, _fake_decoder(),
                      compressed_bytes=1234)
        assert pkt.compressed_bytes == 1234

    def test_mode_is_neural(self) -> None:
        latent = np.random.RandomState(4).randn(32, 313).astype(np.float32)
        tokens = LatentTokens(tokens=None, latent=latent,
                               snac_preset="balanced")
        pkt = decode(tokens, _fake_decoder())
        assert pkt.mode == "neural"


# ============================================================
# export.main argparse plumbing
# ============================================================


class TestExportMain:
    def test_main_requires_checkpoint(self, monkeypatch) -> None:
        """Without ``--checkpoint`` argparse must exit with code 2."""
        from lamquant_codec import export as exp
        monkeypatch.setattr(sys, "argv", ["lamquant-export"])
        with pytest.raises(SystemExit) as e:
            exp.main()
        assert e.value.code == 2  # argparse "missing required arg"

    def test_main_runs_with_mocked_checkpoint(self, monkeypatch, tmp_path) -> None:
        """With a mocked checkpoint, the export pipeline calls all four
        exporters and exits cleanly.
        """
        from lamquant_codec import export as exp
        out_dir = tmp_path / "out"

        # Mock the heavy bits: torch.load + model factory + the four
        # exporters. The test pins that main() calls them in order
        # with the documented arg shapes; it does NOT pin the output
        # bytes.
        mock_model = MagicMock()
        mock_model.eval = MagicMock(return_value=mock_model)

        with patch.object(exp, "export_to_header") as eth, \
             patch.object(exp, "compute_firmware_crc") as cfc, \
             patch.object(exp, "export_fsq_lattice") as efl, \
             patch.object(exp, "export_toeplitz_seeds") as ets, \
             patch("lamquant_codec.models.encoder.TernaryMobileNetV5_Subband.from_checkpoint",
                    return_value=mock_model):
            monkeypatch.setattr(sys, "argv", [
                "lamquant-export",
                "--checkpoint", str(tmp_path / "fake.ckpt"),
                "--output", str(out_dir),
            ])
            exp.main()

        # All four exporters called exactly once
        assert eth.call_count == 1
        assert cfc.call_count == 1
        assert efl.call_count == 1
        assert ets.call_count == 1
        # Output directory created
        assert out_dir.is_dir()
