"""Decoding: LatentTokens → EEGPacket.

Runs the decoder head and inverse-lifts back to the time domain, producing
an EEGPacket that any benchmark or GUI layer can consume.

Decoder model is dependency-injected — callers pass a model with a
`.decode(latent, target_len=T, quantize=True)` method. The production path
uses SubbandCodec which owns the model and LPC coefficients together; for
pure pipeline use, callers must also supply the LPC coefficients if they
want full time-domain reconstruction.
"""
import numpy as np
import torch
from lamquant_codec.codec_types import LatentTokens, EEGPacket


def decode(tokens: LatentTokens, model, *, target_len: int = 313,
           compressed_bytes: int = 0,
           sample_rate: int = 250,
           metadata: dict = None) -> EEGPacket:
    """Run the decoder head on a LatentTokens to produce an EEGPacket.

    NOTE: this returns the L3-domain reconstruction only. If the caller
    needs the original 2500-sample time-domain signal, they must also run
    the inverse lifting DWT + LPC synthesis on the decoder output. For a
    production one-shot path, use SubbandCodec.decompress()+decode() which
    chains both stages with the stored LPC coefficients.

    Args:
        tokens: LatentTokens from decompress().
        model: torch.nn.Module with .decode(latent, target_len, quantize).
        target_len: L3 time length (default 313).
        compressed_bytes: Wire size (for CR metadata).
        sample_rate: Hz (metadata only — L3 is 31.3 Hz effective).
        metadata: Extra metadata passed through.

    Returns:
        EEGPacket containing the L3 reconstruction.
    """
    latent_np = tokens.latent if tokens.latent is not None else tokens.tokens
    latent = torch.as_tensor(np.asarray(latent_np), dtype=torch.float32)
    if latent.ndim == 2:
        latent = latent.unsqueeze(0)

    with torch.no_grad():
        recon = model.decode(latent, target_len=target_len, quantize=True)

    signal = recon.squeeze(0).cpu().numpy() if hasattr(recon, 'cpu') else np.asarray(recon[0])
    return EEGPacket.from_reconstruction(
        signal=signal,
        compressed_bytes=compressed_bytes,
        mode='neural',
        sample_rate=sample_rate,
        metadata=metadata or {'snac_preset': tokens.snac_preset},
    )


__all__ = ['decode']
