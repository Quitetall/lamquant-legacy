"""Subband preprocessing pipeline orchestrators.

Compose LPC + lifting DWT into the Gen 7.1 preprocessing pipeline.
These are shared between codec (decompose.py, codec.py) and training.
"""
import numpy as np

from lamquant_codec.ops.lpc import (
    analyze as lpc_analyze,
    synthesize_channel as lpc_synthesize_channel,
)
from lamquant_codec.ops.lifting import (
    forward_3level, inverse_3level,
    forward_3level_int, inverse_3level_int,
)


def _validate_signal(signal, name="signal"):
    """Validate signal at trust boundaries. Rejects NaN, Inf, wrong shape."""
    if not isinstance(signal, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(signal).__name__}")
    if signal.ndim != 2:
        raise ValueError(f"{name} must be [C, T], got shape {signal.shape}")
    C, T = signal.shape
    if C == 0:
        raise ValueError(f"{name} has 0 channels")
    if T < 2:
        raise ValueError(f"{name} has T={T} samples, need at least 2 for lifting")
    if not np.all(np.isfinite(signal)):
        bad = ~np.isfinite(signal)
        n_bad = int(np.count_nonzero(bad))
        ch, idx = np.argwhere(bad)[0]
        raise ValueError(
            f"{name} contains {n_bad} non-finite values "
            f"(first at channel {ch}, sample {idx}, value={signal[ch, idx]}). "
            f"Hardware fault or upstream corruption."
        )


def hp_filter(signal, fs=250.0, fc=0.5):
    """Apply 2nd-order Butterworth highpass at fc Hz.
    Args:
        signal: [C, T] or [T] numpy array
        fs: sample rate
        fc: cutoff frequency
    Returns:
        filtered signal, same shape
    """
    from scipy.signal import butter, sosfilt
    sos = butter(2, fc, btype='high', fs=fs, output='sos')
    if signal.ndim == 1:
        return sosfilt(sos, signal).astype(signal.dtype)
    return np.stack([sosfilt(sos, signal[c]) for c in range(signal.shape[0])]).astype(signal.dtype)


def preprocess_subband(signal, order=8, autocorr_len=256):
    """Full Gen 7.1 preprocessing pipeline for multi-channel EEG.

    Args:
        signal: [C, T] numpy array (HP-filtered EEG, T=2500)
        order: LPC order

    Returns:
        l3_approx: [C, 313] -- TNN encoder input
        lpc_coeffs: [C, order] -- for packet encoding
        subbands_per_ch: list of dicts -- all subbands per channel
    """
    _validate_signal(signal)
    C, T = signal.shape

    # LPC analysis
    lpc_coeffs, residual = lpc_analyze(signal, order, autocorr_len)

    # 3-level lifting DWT per channel
    l3_approx_list = []
    subbands_per_ch = []
    for c in range(C):
        subs = forward_3level(residual[c])
        l3_approx_list.append(subs['l3_approx'])
        subbands_per_ch.append(subs)

    l3_approx = np.stack(l3_approx_list)
    return l3_approx, lpc_coeffs, subbands_per_ch


def reconstruct_from_subband(l3_approx_recon, lpc_coeffs, subbands_per_ch):
    """Full inverse pipeline: TNN output -> inverse lifting -> LPC synthesis -> EEG.

    This is what the training loss must use:
      loss = MSE(reconstruct_from_subband(TNN_decode(latent), ...), original)

    Uses integer inverse lifting to match preprocess_subband_single's integer
    forward lifting. The TNN output is rounded to integer before inverse lifting
    since the original L3 approximation was integer-valued.

    Args:
        l3_approx_recon: [C, 313] -- TNN decoder output (reconstructed L3 approx)
        lpc_coeffs: [C, order] -- LPC coefficients from analysis
        subbands_per_ch: list of dicts -- detail subbands from analysis

    Returns:
        signal: [C, T] -- reconstructed EEG
    """
    C = l3_approx_recon.shape[0]
    if C != len(subbands_per_ch):
        raise ValueError(
            f"Channel mismatch: l3_approx has {C} channels "
            f"but subbands_per_ch has {len(subbands_per_ch)}")
    if C != lpc_coeffs.shape[0]:
        raise ValueError(
            f"Channel mismatch: l3_approx has {C} channels "
            f"but lpc_coeffs has {lpc_coeffs.shape[0]} rows")
    signals = []
    for c in range(C):
        # Replace L3 approximation with TNN reconstruction
        # Round to integer to match the integer lifting domain
        subs = {k: np.round(v).astype(np.int64) for k, v in subbands_per_ch[c].items()}
        subs['l3_approx'] = np.round(l3_approx_recon[c]).astype(np.int64)

        # Inverse integer lifting (matches forward_int used in preprocess_subband_single)
        residual = inverse_3level_int(subs).astype(np.float64)

        # LPC synthesis
        signal = lpc_synthesize_channel(residual, lpc_coeffs[c])
        signals.append(signal)

    return np.stack(signals)


def preprocess_subband_single(signal_np, order=8, autocorr_len=256):
    """Preprocess a single [C, T] numpy array using INTEGER lifting.

    Pipeline: LPC (float analysis) -> integer quantize residual -> integer lifting.
    The L3 approximation is integer-valued, cast to float32 for the TNN.
    Matches firmware: LPC -> integer lifting -> TNN input.

    Returns (l3_approx [C, 313] float32, coeffs, subs).
    """
    signal_np = np.asarray(signal_np)
    if signal_np.ndim != 2:
        raise ValueError(f"Expected [C, T] array, got shape {signal_np.shape}")
    if not np.all(np.isfinite(signal_np)):
        bad = ~np.isfinite(signal_np)
        ch, idx = np.argwhere(bad)[0]
        raise ValueError(
            f"Signal contains non-finite value at channel {ch}, sample {idx}. "
            f"Hardware fault or upstream corruption.")
    signal = signal_np.astype(np.float64)
    C, T = signal.shape

    # LPC analysis (float -- coefficients computed once, used on both sides)
    lpc_coeffs, residual = lpc_analyze(signal, order, autocorr_len)

    # Quantize residual to int16 range for integer lifting
    # Scale per-channel to use full int16 dynamic range
    l3_list = []
    subbands_per_ch = []
    for c in range(C):
        # Round to integer (the firmware uses Q31 -> int16 conversion)
        residual_int = np.round(residual[c]).astype(np.int64)

        # Integer lifting (bit-exact, matches firmware)
        subs = forward_3level_int(residual_int)
        l3_list.append(subs['l3_approx'].astype(np.float64))

        # Store float versions of subbands for detail encoding compatibility
        subs_float = {k: v.astype(np.float64) for k, v in subs.items()}
        subbands_per_ch.append(subs_float)

    l3_approx = np.stack(l3_list).astype(np.float32)
    return l3_approx, lpc_coeffs, subbands_per_ch


def preprocess_subband_torch(signal_tensor, order=8, autocorr_len=256):
    """Preprocess a batch of signals through LPC + lifting.

    Args:
        signal_tensor: [B, C, T] torch tensor (HP-filtered, T=2500)

    Returns:
        l3_approx: [B, C, 313] torch tensor -- TNN input
        metadata: list of (lpc_coeffs, subbands_per_ch) per batch item
    """
    import torch
    B, C, T = signal_tensor.shape
    device = signal_tensor.device
    dtype = signal_tensor.dtype

    l3_list = []
    metadata = []

    for b in range(B):
        sig_np = signal_tensor[b].detach().cpu().numpy()
        l3, coeffs, subs = preprocess_subband_single(sig_np, order, autocorr_len)
        l3_list.append(l3)
        metadata.append((coeffs, subs))

    l3_approx = torch.tensor(np.stack(l3_list), dtype=dtype, device=device)
    return l3_approx, metadata


def reconstruct_subband_torch(l3_recon_tensor, metadata):
    """Inverse pipeline for training loss computation.

    Args:
        l3_recon_tensor: [B, C, 313] torch tensor -- TNN decoder output
        metadata: from preprocess_subband_torch

    Returns:
        recon: [B, C, T] torch tensor -- reconstructed EEG
    """
    import torch
    B, C, L3_len = l3_recon_tensor.shape
    device = l3_recon_tensor.device
    dtype = l3_recon_tensor.dtype

    recons = []
    for b in range(B):
        l3_np = l3_recon_tensor[b].detach().cpu().numpy().astype(np.float64)
        coeffs, subs = metadata[b]
        recon = reconstruct_from_subband(l3_np, coeffs, subs)
        recons.append(recon)

    return torch.tensor(np.stack(recons), dtype=dtype, device=device)


__all__ = [
    'hp_filter',
    'preprocess_subband', 'reconstruct_from_subband',
    'preprocess_subband_single',
    'preprocess_subband_torch', 'reconstruct_subband_torch',
]
