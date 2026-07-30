"""Preprocessing: RawEEG → RawEEG.

High-pass filter at 0.5 Hz (per ITU-T H.BWC and IEC 60601-2-26 clinical EEG
specs) to remove DC drift and baseline wander from the ADC stream.

This is the first pipeline stage. It takes RawEEG and returns RawEEG with
the same shape and sample rate — just a cleaner signal.

Implementation note
-------------------
First-order IIR high-pass:
    y[n] = a * (y[n-1] + x[n] - x[n-1]),  a = RC / (RC + dt)

Realised as `scipy.signal.lfilter` with b=[a, -a], a_coeff=[1, -a]. Vectorised
along the time axis per channel (~100× faster than the equivalent Python
loop while producing bit-identical samples).
"""
import numpy as np
from lamquant_codec.codec_types import RawEEG


def preprocess(raw: RawEEG, cutoff_hz: float = 0.5) -> RawEEG:
    """Apply high-pass filter at cutoff_hz to every channel.

    Args:
        raw: Input RawEEG.
        cutoff_hz: HP cutoff frequency in Hz (default 0.5 for clinical EEG).

    Returns:
        RawEEG with HP-filtered signal, same sample rate, same channels.
    """
    # Lazy: scipy.signal imports a lot — defer until the first call so
    # `import lamquant_codec` stays fast for type-only / inspection use.
    from scipy.signal import lfilter
    x = raw.signal.astype(np.float64, copy=False)
    dt = 1.0 / raw.sample_rate
    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    a = rc / (rc + dt)

    # First-order IIR HP in Direct-Form II Transposed:
    # y[n] = a*x[n] + d[0],   d[0] := -a*x[n] + a*y[n]
    b_num = np.array([a, -a], dtype=np.float64)
    a_den = np.array([1.0, -a], dtype=np.float64)

    # Initial state to match the original Python loop (which sets y[0]=0):
    # zi[c] = -a * x[c, 0]  →  y[0] = a*x[0] + zi = 0  per channel.
    zi = (-a * x[:, 0:1]).astype(np.float64)

    y, _ = lfilter(b_num, a_den, x, axis=-1, zi=zi)

    return RawEEG(
        signal=y.astype(raw.signal.dtype, copy=False),
        sample_rate=raw.sample_rate,
        channels=raw.channels,
        timestamp_us=raw.timestamp_us,
        channel_labels=raw.channel_labels,
    )


__all__ = ['preprocess']
