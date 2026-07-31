"""Coverage tests for ``lamquant_codec.benchmark``.

Pins quality-metric contracts:
  - prd == 0 on a perfect reconstruction
  - prd > 0 when reconstruction differs
  - pearson_r == 1.0 on a perfect reconstruction
  - pearson_r == 0 when one of the inputs is constant (std=0)
  - per_channel_r returns one R per channel
  - compression_ratio = raw / compressed
  - snr_db == inf on perfect reconstruction
  - rmse == 0 on perfect reconstruction
  - is_lossless on perfect int reconstruction
  - per_band_prd returns dict with all bands
  - full_report returns all keys

Math fixtures (np.random) — NOT synthetic EEG.
"""
from __future__ import annotations

import math
import pytest
import numpy as np

from lamquant_codec.benchmark import Benchmark
from lamquant_codec.codec_types import EEGPacket


def _math_signal(C: int = 3, T: int = 1024, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(C, T).astype(np.float64) * 100.0


def _make_packet(signal: np.ndarray, *, mode: str = 'neural',
                 compressed_bytes: int = 1000,
                 sample_rate: int = 250) -> EEGPacket:
    return EEGPacket(
        signal=signal,
        sample_rate=sample_rate,
        n_channels=signal.shape[0],
        mode=mode,
        compressed_bytes=compressed_bytes,
        raw_bytes=signal.shape[0] * signal.shape[1] * 2,
    )


class TestPrd:
    def test_zero_when_perfect_reconstruction(self) -> None:
        orig = _math_signal()
        pkt = _make_packet(orig.copy())
        assert Benchmark.prd(orig, pkt) == 0.0

    def test_positive_when_reconstruction_differs(self) -> None:
        orig = _math_signal()
        recon = orig + 1.0
        pkt = _make_packet(recon)
        assert Benchmark.prd(orig, pkt) > 0.0

    def test_zero_signal_handled(self) -> None:
        orig = np.zeros((2, 256))
        pkt = _make_packet(np.zeros_like(orig))
        # Both zero -> match -> 0 by spec.
        assert Benchmark.prd(orig, pkt) == 0.0


class TestPearsonR:
    def test_one_for_perfect_reconstruction(self) -> None:
        orig = _math_signal()
        pkt = _make_packet(orig.copy())
        # Float math: pearson_r(x, x) = 1.0 within 1 ULP. Don't pin
        # exact 1.0 — the standardisation step accumulates rounding.
        assert Benchmark.pearson_r(orig, pkt) == pytest.approx(1.0, abs=1e-12)

    def test_returns_float(self) -> None:
        orig = _math_signal()
        pkt = _make_packet(orig + np.random.RandomState(1).randn(*orig.shape))
        r = Benchmark.pearson_r(orig, pkt)
        assert isinstance(r, float)
        assert -1.0 <= r <= 1.0

    def test_zero_when_constant_input(self) -> None:
        """std==0 path returns 0 by spec."""
        orig = np.zeros((2, 256))
        recon = np.ones((2, 256))
        pkt = _make_packet(recon)
        assert Benchmark.pearson_r(orig, pkt) == 0.0


class TestPerChannelR:
    def test_shape_matches_channels(self) -> None:
        orig = _math_signal(C=5)
        pkt = _make_packet(orig.copy())
        rs = Benchmark.per_channel_r(orig, pkt)
        assert rs.shape == (5,)

    def test_perfect_reconstruction_all_ones(self) -> None:
        orig = _math_signal(C=3)
        pkt = _make_packet(orig.copy())
        rs = Benchmark.per_channel_r(orig, pkt)
        np.testing.assert_allclose(rs, np.ones(3))

    def test_constant_channel_returns_zero(self) -> None:
        orig = np.array([[1.0] * 100, [0.0] * 100])
        rng = np.random.RandomState(0)
        recon = orig + rng.randn(*orig.shape) * 0.01
        pkt = _make_packet(recon)
        rs = Benchmark.per_channel_r(orig, pkt)
        # Channel 0 has zero std -> spec returns 0
        assert rs[0] == 0.0


class TestCompressionRatio:
    def test_basic_ratio(self) -> None:
        orig = _math_signal()
        pkt = _make_packet(orig.copy(), compressed_bytes=1000)
        # raw_bytes set in _make_packet
        ratio = Benchmark.compression_ratio(orig, pkt)
        assert ratio == pytest_approx_eq(pkt.raw_bytes / pkt.compressed_bytes)

    def test_inf_when_compressed_zero(self) -> None:
        orig = _math_signal()
        pkt = _make_packet(orig.copy(), compressed_bytes=0)
        # construction overrides compressed_bytes; force via direct assign:
        pkt.compressed_bytes = 0
        assert math.isinf(Benchmark.compression_ratio(orig, pkt))


def pytest_approx_eq(value):
    import pytest
    return pytest.approx(value, rel=1e-9)


class TestSnrDb:
    def test_inf_on_perfect_reconstruction(self) -> None:
        orig = _math_signal()
        pkt = _make_packet(orig.copy())
        assert math.isinf(Benchmark.snr_db(orig, pkt))

    def test_returns_float_when_noisy(self) -> None:
        orig = _math_signal()
        rng = np.random.RandomState(0)
        recon = orig + rng.randn(*orig.shape) * 10
        pkt = _make_packet(recon)
        v = Benchmark.snr_db(orig, pkt)
        assert isinstance(v, float)

    def test_zero_signal_returns_zero(self) -> None:
        orig = np.zeros((2, 256))
        pkt = _make_packet(np.ones_like(orig))
        # signal_power < 1e-20 path -> 0
        assert Benchmark.snr_db(orig, pkt) == 0.0


class TestRmse:
    def test_zero_on_perfect(self) -> None:
        orig = _math_signal()
        pkt = _make_packet(orig.copy())
        assert Benchmark.rmse(orig, pkt) == 0.0

    def test_positive_when_noisy(self) -> None:
        orig = _math_signal()
        pkt = _make_packet(orig + 1.0)
        assert Benchmark.rmse(orig, pkt) > 0.0


class TestMaxError:
    def test_zero_on_perfect(self) -> None:
        orig = _math_signal()
        pkt = _make_packet(orig.copy())
        assert Benchmark.max_error(orig, pkt) == 0.0

    def test_picks_max(self) -> None:
        orig = np.zeros((1, 4))
        recon = np.array([[0.0, 0.0, 3.0, -5.0]])
        pkt = _make_packet(recon)
        assert Benchmark.max_error(orig, pkt) == 5.0


class TestIsLossless:
    def test_true_on_perfect_int_signal(self) -> None:
        orig = np.round(_math_signal()).astype(np.float64)
        pkt = _make_packet(orig.copy())
        assert Benchmark.is_lossless(orig, pkt) is True

    def test_false_when_off_by_one(self) -> None:
        orig = np.round(_math_signal()).astype(np.float64)
        recon = orig.copy()
        recon[0, 0] += 1.0
        pkt = _make_packet(recon)
        assert Benchmark.is_lossless(orig, pkt) is False


class TestPerBandPrd:
    def test_returns_all_bands(self) -> None:
        orig = _math_signal(C=2, T=2048)
        pkt = _make_packet(orig.copy())
        out = Benchmark.per_band_prd(orig, pkt, sample_rate=250)
        for band in ('delta', 'theta', 'alpha', 'beta', 'gamma'):
            assert band in out
            assert isinstance(out[band], float)

    def test_perfect_recon_zero_per_band(self) -> None:
        orig = _math_signal(C=2, T=2048)
        pkt = _make_packet(orig.copy())
        out = Benchmark.per_band_prd(orig, pkt, sample_rate=250)
        for v in out.values():
            assert v == 0.0


class TestFullReport:
    def test_returns_all_keys(self) -> None:
        orig = _math_signal()
        pkt = _make_packet(orig.copy())
        rep = Benchmark.full_report(orig, pkt)
        for key in ('prd', 'r', 'cr', 'snr_db', 'rmse', 'max_error',
                    'lossless', 'per_channel_r', 'per_band_prd',
                    'mode', 'compressed_bytes', 'n_samples'):
            assert key in rep

    def test_full_report_passes_through_mode(self) -> None:
        orig = _math_signal()
        pkt = _make_packet(orig.copy(), mode='lossless')
        rep = Benchmark.full_report(orig, pkt)
        assert rep['mode'] == 'lossless'
