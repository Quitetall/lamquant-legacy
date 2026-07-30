"""ADC noise floor estimation via bit-level temporal analysis.

Hypothesis: ADC thermal noise is temporally uncorrelated. Signal is
temporally correlated (adjacent EEG samples are similar). For each
bit position, if the lag-1 autocorrelation is near zero, that bit
carries noise, not signal.

The estimate bootstraps the search for optimal noise_bits. Empirical
sweeps (training with different masking levels, measuring R) refine it.
"""
import numpy as np
from typing import Dict, List


def estimate_noise_bits(signal_int: np.ndarray, *,
                        max_bits: int = 10,
                        corr_threshold: float = 0.05) -> int:
    """Estimate ADC noise floor in LSBs for a single window.

    For each bit position k (0 = LSB, 1, 2, ...), extracts the k-th bit
    across all samples and computes lag-1 autocorrelation. Bits with
    |autocorrelation| < threshold are noise (temporally uncorrelated).

    Args:
        signal_int: [C, T] or [T] integer signal (int16/int32/int64)
        max_bits: maximum bits to test (default 10)
        corr_threshold: autocorrelation below which a bit is considered
            noise (default 0.05)

    Returns:
        noise_bits: number of contiguous bottom bits that are noise
            (0 = no detectable noise)
    """
    flat = np.asarray(signal_int, dtype=np.int64).ravel()
    n = len(flat)
    if n < 64:
        return 0

    noise_bits = 0
    for k in range(max_bits):
        # Extract bit k from every sample
        bit_k = (flat >> k) & 1
        bit_k_f = bit_k.astype(np.float64)

        # Lag-1 autocorrelation: corr(bit_k[:-1], bit_k[1:])
        mean = bit_k_f.mean()
        var = bit_k_f.var()
        if var < 1e-12:
            # Constant bit (all 0 or all 1) — this is signal, not noise
            break

        centered = bit_k_f - mean
        autocorr = np.dot(centered[:-1], centered[1:]) / (var * (n - 1))

        if abs(autocorr) < corr_threshold:
            noise_bits = k + 1
        else:
            break  # once we hit a correlated bit, all higher bits carry signal

    return noise_bits


def estimate_noise_bits_batch(signals: List[np.ndarray], *,
                              max_bits: int = 10,
                              corr_threshold: float = 0.05) -> List[int]:
    """Estimate noise_bits for a batch of windows.

    Args:
        signals: list of [C, T] or [T] integer arrays
        max_bits: maximum bits to test
        corr_threshold: autocorrelation threshold

    Returns:
        list of noise_bits values, one per window
    """
    return [estimate_noise_bits(s, max_bits=max_bits,
                                corr_threshold=corr_threshold)
            for s in signals]


def noise_profile(signal_int: np.ndarray, *,
                  max_bits: int = 10) -> Dict:
    """Full diagnostic profile of the noise floor for a single window.

    Returns per-bit autocorrelation, per-channel breakdown, and the
    estimated noise_bits. Useful for understanding why a particular
    estimate was made.

    Args:
        signal_int: [C, T] or [T] integer signal
        max_bits: maximum bits to analyze

    Returns:
        dict with keys:
            'noise_bits': int
            'per_bit': list of dicts per k=0..max_bits-1, each with:
                'bit_position': int (0 = LSB)
                'autocorrelation': float (lag-1)
                'is_noise': bool
                'fraction_set': float (fraction of samples where this bit is 1)
            'per_channel': list of int — noise_bits per channel (if 2D)
            'signal_shape': tuple
            'n_samples': int
    """
    sig = np.asarray(signal_int, dtype=np.int64)
    flat = sig.ravel()
    n = len(flat)

    per_bit = []
    noise_bits = 0
    found_signal = False
    for k in range(max_bits):
        bit_k = (flat >> k) & 1
        bit_k_f = bit_k.astype(np.float64)
        mean = bit_k_f.mean()
        var = bit_k_f.var()

        if var < 1e-12:
            autocorr = 1.0  # constant → maximally correlated (signal)
        else:
            centered = bit_k_f - mean
            autocorr = float(np.dot(centered[:-1], centered[1:]) / (var * (n - 1)))

        is_noise = abs(autocorr) < 0.05 and not found_signal
        if is_noise:
            noise_bits = k + 1
        else:
            found_signal = True

        per_bit.append({
            'bit_position': k,
            'autocorrelation': round(autocorr, 4),
            'is_noise': is_noise,
            'fraction_set': round(float(mean), 4),
        })

    # Per-channel breakdown (if 2D)
    per_channel = []
    if sig.ndim == 2:
        for c in range(sig.shape[0]):
            per_channel.append(estimate_noise_bits(sig[c], max_bits=max_bits))

    return {
        'noise_bits': noise_bits,
        'per_bit': per_bit,
        'per_channel': per_channel,
        'signal_shape': tuple(int(x) for x in sig.shape),
        'n_samples': n,
    }


__all__ = ['estimate_noise_bits', 'estimate_noise_bits_batch', 'noise_profile']
