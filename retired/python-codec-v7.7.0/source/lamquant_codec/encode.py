"""Encoding: SubbandDecomposition → LatentTokens.

Runs the TNN encoder on the L3 approximation and FSQ-quantizes the latent.
The encoder is dependency-injected (pass a model with an `.encode(x)`
method) — this module does not own or load checkpoints.

When an SNN activity detector is provided, it classifies each timestep
as quiet/active/seizure and produces a per-timestep FSQ level schedule
(L=2 quiet, L=3 active, L=5 seizure). This drives adaptive SNAC
compression — quiet segments get aggressive compression (high CR),
active/seizure segments preserve clinical signal (low CR).

For the production path, lamquant_neural.codec.SubbandCodec (in the private
LamQuant-Neural wheel) wraps a checkpoint and satisfies this interface;
call encode(subband, codec.model).
"""
import numpy as np
import torch
from lamquant_codec.codec_types import SubbandDecomposition, LatentTokens


def encode(subband: SubbandDecomposition, model, *, fsq_levels=None,
           snac_preset: str = 'compact',
           snn=None) -> LatentTokens:
    """Encode a SubbandDecomposition into LatentTokens.

    Args:
        subband: Output of decompose().
        model: torch.nn.Module with `.encode(x)` taking [1, C, T] → latent.
        fsq_levels: Optional per-block FSQ level schedule. If provided,
            overrides SNN-derived levels.
        snac_preset: Multi-scale FSQ preset label (metadata only).
        snn: Optional Mamba SNN activity detector. If provided and
            fsq_levels is None, runs classify_per_timestep() on the L3
            input to produce adaptive FSQ levels.

    Returns:
        LatentTokens with the quantized token array and FSQ level schedule.
    """
    x = torch.as_tensor(subband.l3_approx, dtype=torch.float32)
    if x.ndim == 2:
        x = x.unsqueeze(0)

    with torch.no_grad():
        latent = model.encode(x)

    lat_np = latent.squeeze(0).cpu().numpy() if hasattr(latent, 'cpu') else np.asarray(latent)

    # Adaptive FSQ: SNN classifies activity → per-timestep FSQ levels.
    # Priority: explicit fsq_levels > SNN-derived > None (uniform FSQ).
    level_schedule = None
    if fsq_levels is not None:
        level_schedule = list(fsq_levels)
    elif snn is not None:
        level_schedule = _snn_classify(snn, x, target_T=lat_np.shape[-1])

    return LatentTokens(
        tokens=lat_np,
        latent=lat_np,
        fsq_levels=level_schedule,
        snac_preset=snac_preset,
        shape=tuple(lat_np.shape),
        vmin=-1.0,
        vmax=1.0,
    )


def _snn_classify(snn, l3_input, target_T=79):
    """Run SNN activity detector and return FSQ level schedule.

    Args:
        snn: MambaSNN (or any model with classify_per_timestep method).
        l3_input: [1, 21, 313] L3 approximation tensor.
        target_T: target temporal resolution (must match latent timesteps).

    Returns:
        List of int FSQ levels, length target_T. Each is 2, 3, or 5.
    """
    levels = snn.classify_per_timestep(l3_input, target_T=target_T)
    # levels is [B, T] int tensor → take first batch, convert to list
    return levels[0].cpu().tolist()


__all__ = ['encode']
