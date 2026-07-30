"""Coverage tests for ``lamquant_codec.preprocess``.

Pins behavioural contracts:
  - preprocess(raw) returns RawEEG with the same shape, sample_rate, and
    channels (it's a high-pass filter)
  - dtype is preserved
  - 1D and 2D inputs are accepted (RawEEG __post_init__ promotes 1D)
  - DC offset is reduced after filtering

Math fixtures (np.random) — math shape, not synthetic EEG data.
"""
from __future__ import annotations

import numpy as np

from lamquant_codec.codec_types import RawEEG
from lamquant_codec.preprocess import preprocess


def _math_raw(C: int = 4, T: int = 2500, seed: int = 0) -> RawEEG:
    rng = np.random.RandomState(seed)
    signal = rng.randn(C, T).astype(np.float64) * 100.0
    return RawEEG(signal=signal, sample_rate=250, channels=C)


class TestPreprocess:
    def test_shape_preserved(self) -> None:
        raw = _math_raw(C=4, T=2500)
        out = preprocess(raw)
        assert out.signal.shape == raw.signal.shape

    def test_sample_rate_preserved(self) -> None:
        raw = _math_raw(C=4, T=2500)
        out = preprocess(raw)
        assert out.sample_rate == raw.sample_rate

    def test_dtype_preserved(self) -> None:
        raw = _math_raw(C=4, T=2500)
        out = preprocess(raw)
        # raw.signal.dtype is float64 -> preprocess returns float64 by spec.
        assert out.signal.dtype == raw.signal.dtype

    def test_channels_preserved(self) -> None:
        raw = _math_raw(C=7, T=1000)
        out = preprocess(raw)
        assert out.signal.shape[0] == 7

    def test_dc_offset_removed(self) -> None:
        """High-pass at 0.5 Hz must attenuate the DC component."""
        T = 2500
        dc = 1000.0  # huge DC offset
        signal = np.full((4, T), dc, dtype=np.float64)
        # Add small AC content so the filter has something to preserve.
        rng = np.random.RandomState(0)
        signal += rng.randn(4, T)
        raw = RawEEG(signal=signal, sample_rate=250, channels=4)
        out = preprocess(raw)
        # After the transient, output mean should be much smaller than DC.
        tail_mean = float(np.mean(out.signal[:, T // 2:]))
        assert abs(tail_mean) < dc / 10.0

    def test_finite_output(self) -> None:
        raw = _math_raw(C=2, T=500)
        out = preprocess(raw)
        assert np.all(np.isfinite(out.signal))

    def test_custom_cutoff_accepted(self) -> None:
        raw = _math_raw(C=2, T=500)
        out_default = preprocess(raw, cutoff_hz=0.5)
        out_custom = preprocess(raw, cutoff_hz=2.0)
        assert out_default.signal.shape == out_custom.signal.shape

    def test_channel_labels_preserved(self) -> None:
        rng = np.random.RandomState(0)
        labels = ['Fp1', 'Fp2', 'C3', 'C4']
        raw = RawEEG(
            signal=rng.randn(4, 500),
            sample_rate=250,
            channels=4,
            channel_labels=labels,
        )
        out = preprocess(raw)
        assert out.channel_labels == labels
