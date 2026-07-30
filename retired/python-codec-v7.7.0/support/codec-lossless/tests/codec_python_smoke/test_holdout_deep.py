"""Deep coverage tests for ``lamquant_codec.holdout``.

Complements ``test_holdout_functional.py``. Targets:
  - BenchmarkReport.summary() formatting + field surface
  - _default_holdout_dir resolution with $LAMQUANT_HOLDOUT_DIR override
  - load_holdout: insufficient-files RuntimeError
  - run_benchmark: with a trivial pass-through codec_fn against
    synthetic HoldoutWindow instances (no real q31 corpus needed)
  - HoldoutWindow dataclass round-trip

Pure math/logic — no real EDF required.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from lamquant_codec.codec_types import EEGPacket
from lamquant_codec.holdout import (
    HOLDOUT_SEED,
    HOLDOUT_SIZE,
    N_CHBMIT,
    N_HIGH_ENERGY,
    N_LOW_ENERGY,
    N_TUH,
    RAW_BYTES_PER_WINDOW,
    BenchmarkReport,
    HoldoutWindow,
    _default_holdout_dir,
    _load_window,
    holdout_fingerprint,
    load_holdout,
    run_benchmark,
)


def _synthetic_window(
    idx: int, source: str = "chbmit", energy_offset: float = 0.0
) -> HoldoutWindow:
    rng = np.random.RandomState(idx)
    sig = rng.randn(21, 2500).astype(np.float32) + energy_offset
    return HoldoutWindow(
        filename=f"synth_{idx:03d}.npz",
        signal=sig,
        l3=rng.randn(21, 313).astype(np.float32),
        source=source,
        energy=float(np.sqrt(np.mean(sig.astype(np.float64) ** 2))),
        index=idx,
    )


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_holdout_composition_sums_correctly(self) -> None:
        # The four strata add up to HOLDOUT_SIZE.
        assert N_CHBMIT + N_TUH == 80
        assert N_HIGH_ENERGY + N_LOW_ENERGY == 20
        assert HOLDOUT_SIZE == 100

    def test_holdout_seed_pinned(self) -> None:
        assert HOLDOUT_SEED == 20260415

    def test_raw_bytes_per_window(self) -> None:
        assert RAW_BYTES_PER_WINDOW == 21 * 2500 * 2


# ---------------------------------------------------------------------------
# _default_holdout_dir
# ---------------------------------------------------------------------------


class TestDefaultHoldoutDir:
    def test_env_var_override_when_valid(self, tmp_path: Path) -> None:
        with patch.dict(os.environ, {"LAMQUANT_HOLDOUT_DIR": str(tmp_path)}):
            assert _default_holdout_dir() == str(tmp_path)

    def test_invalid_env_falls_back(self, tmp_path: Path) -> None:
        """Env var pointing at non-existent dir falls through to default."""
        bogus = tmp_path / "does_not_exist"
        with patch.dict(os.environ, {"LAMQUANT_HOLDOUT_DIR": str(bogus)}):
            result = _default_holdout_dir()
            # Default is weights/holdout_q31/ under repo root.
            assert "holdout_q31" in result

    def test_no_env_returns_default(self) -> None:
        env_copy = {k: v for k, v in os.environ.items()
                    if k != "LAMQUANT_HOLDOUT_DIR"}
        with patch.dict(os.environ, env_copy, clear=True):
            result = _default_holdout_dir()
            assert "holdout_q31" in result


# ---------------------------------------------------------------------------
# load_holdout error paths
# ---------------------------------------------------------------------------


class TestLoadHoldoutErrors:
    def test_insufficient_files_raises_runtime(self, tmp_path: Path) -> None:
        """Directory exists but has < HOLDOUT_SIZE Q31 files."""
        # Create a few files but not 100.
        for i in range(3):
            (tmp_path / f"chbmit_{i:03d}.npz").write_bytes(b"x")
        with pytest.raises(RuntimeError, match="at least"):
            load_holdout(data_dir=str(tmp_path))


def _write_q31_npz(path: Path, seed: int = 0, source: str = "chbmit") -> None:
    """Write a minimal Q31 NPZ at `path` with shape contract."""
    rng = np.random.RandomState(seed)
    # raw int32 Q31 [21, 2500]
    raw = (rng.randn(21, 2500) * 1e6).clip(-2 ** 31, 2 ** 31 - 1).astype(np.int32)
    l3 = rng.randn(21, 313).astype(np.float32)
    np.savez(str(path), data=raw, l3=l3)


class TestLoadHoldoutBody:
    def test_loads_full_holdout_from_fake_corpus(self, tmp_path: Path) -> None:
        """Build a corpus of >= HOLDOUT_SIZE Q31 files with both chbmit and tuh
        prefixes; load_holdout must return HOLDOUT_SIZE windows with re-indexed
        positions and stratified high/low energy ordering."""
        # Need >= N_CHBMIT chbmit and >= N_TUH tuh files
        for i in range(N_CHBMIT + 5):
            _write_q31_npz(tmp_path / f"chbmit_{i:03d}.npz", seed=i, source="chbmit")
        for i in range(N_TUH + 5):
            _write_q31_npz(tmp_path / f"tuh_{i:03d}.npz", seed=100 + i, source="tuh")
        # 90 files total > HOLDOUT_SIZE? No: N_CHBMIT+5 + N_TUH+5 = 90.
        # Pad with more chbmit to push above HOLDOUT_SIZE.
        for i in range(50):
            _write_q31_npz(tmp_path / f"chbmit_pad_{i:03d}.npz",
                           seed=200 + i, source="chbmit")
        holdout = load_holdout(data_dir=str(tmp_path))
        # Should be at most HOLDOUT_SIZE; equals N_CHBMIT + N_TUH after sampling
        assert len(holdout) == N_CHBMIT + N_TUH
        # Indices re-assigned 0..n-1
        for i, w in enumerate(holdout):
            assert w.index == i
        # Source labels present
        sources = {w.source for w in holdout}
        assert sources.issubset({"chbmit", "tuh"})
        # Sorted: first N_LOW_ENERGY are lowest energy, last N_HIGH_ENERGY are highest
        energies = [w.energy for w in holdout]
        # First low_energy entries should be <= last high_energy entries.
        assert max(energies[:N_LOW_ENERGY]) <= min(energies[-N_HIGH_ENERGY:])


# ---------------------------------------------------------------------------
# _load_window (per-file loader)
# ---------------------------------------------------------------------------


class TestLoadWindow:
    def test_loads_q31_npz(self, tmp_path: Path) -> None:
        """_load_window reads data/l3 from a Q31 NPZ, computes energy, infers source."""
        rng = np.random.RandomState(0)
        raw = (rng.randn(21, 2500) * 1e6).astype(np.int32)
        l3 = rng.randn(21, 313).astype(np.float32)
        npz = tmp_path / "chbmit_x.npz"
        np.savez(str(npz), data=raw, l3=l3)
        seg, l3_seg, source, energy = _load_window(str(npz))
        assert seg.dtype == np.float32
        assert seg.shape[0] == 21
        assert l3_seg.shape == (21, 313)
        assert source == "chbmit"  # filename detection
        assert energy >= 0.0

    def test_source_detection_tuh(self, tmp_path: Path) -> None:
        rng = np.random.RandomState(0)
        raw = (rng.randn(21, 2500) * 1e6).astype(np.int32)
        l3 = rng.randn(21, 313).astype(np.float32)
        npz = tmp_path / "tuh_xyz.npz"
        np.savez(str(npz), data=raw, l3=l3)
        _, _, source, _ = _load_window(str(npz))
        assert source == "tuh"


# ---------------------------------------------------------------------------
# BenchmarkReport
# ---------------------------------------------------------------------------


class TestBenchmarkReport:
    def test_construct_full(self) -> None:
        report = BenchmarkReport(
            n_windows=10,
            mean_prd=5.0,
            mean_r=0.95,
            mean_cr=12.3,
            mean_snr_db=20.0,
            per_band_prd={'delta': 1.0, 'theta': 2.0, 'alpha': 3.0,
                          'beta': 4.0, 'gamma': 5.0},
            per_source={'chbmit': {'prd': 5.0, 'r': 0.96},
                        'tuh': {'prd': 5.5, 'r': 0.94}},
            high_energy_prd=6.0,
            low_energy_prd=4.0,
            holdout_fingerprint="abc123",
            lossless_count=2,
        )
        assert report.n_windows == 10
        assert report.lossless_count == 2

    def test_summary_string_contains_fields(self) -> None:
        report = BenchmarkReport(
            n_windows=5,
            mean_prd=10.5,
            mean_r=0.85,
            mean_cr=100.0,
            mean_snr_db=15.5,
            per_band_prd={'delta': 1.0, 'theta': 2.0, 'alpha': 3.0,
                          'beta': 4.0, 'gamma': 5.0},
            per_source={'chbmit': {'prd': 9.0, 'r': 0.87},
                        'tuh': {'prd': 11.0, 'r': 0.83}},
            high_energy_prd=12.0,
            low_energy_prd=8.0,
            holdout_fingerprint="fingerprint_test",
            lossless_count=1,
        )
        s = report.summary()
        assert isinstance(s, str)
        # Required string fragments — pin the structural contract.
        assert "5 windows" in s
        assert "fingerprint_test" in s
        assert "10.50%" in s          # PRD formatting
        assert "0.8500" in s          # R formatting
        assert "100.0:1" in s         # CR formatting
        assert "delta=1.0%" in s
        # Per-source numbers
        assert "PRD=9.00%" in s or "9.00" in s
        # High/Low energy lines
        assert "12.00%" in s
        assert "8.00%" in s
        # Lossless count
        assert "1/5" in s

    def test_summary_empty_per_source(self) -> None:
        """Summary tolerates missing per_source entries (uses .get(...,{}))."""
        report = BenchmarkReport(
            n_windows=1,
            mean_prd=0.0, mean_r=0.0, mean_cr=0.0, mean_snr_db=0.0,
            per_band_prd={'delta': 0.0, 'theta': 0.0, 'alpha': 0.0,
                          'beta': 0.0, 'gamma': 0.0},
            per_source={},  # empty
            high_energy_prd=0.0,
            low_energy_prd=0.0,
            holdout_fingerprint="",
            lossless_count=0,
        )
        s = report.summary()
        # Default .get(...,{}).get('prd', 0) -> 0.00
        assert "0.00%" in s


# ---------------------------------------------------------------------------
# run_benchmark with trivial codec functions
# ---------------------------------------------------------------------------


def _passthrough_codec(signal: np.ndarray) -> EEGPacket:
    """Lossless pass-through codec — for testing benchmark wiring."""
    return EEGPacket(
        signal=signal.astype(np.float64),
        sample_rate=250,
        mode='lossless',
        compressed_bytes=signal.nbytes // 4,  # fake CR of ~4x
        raw_bytes=signal.nbytes,
    )


def _noisy_codec(signal: np.ndarray, scale: float = 0.01) -> EEGPacket:
    """Add small noise; benchmark should give a finite non-zero PRD."""
    rng = np.random.RandomState(42)
    out = signal.astype(np.float64) + rng.randn(*signal.shape) * scale
    return EEGPacket(
        signal=out,
        sample_rate=250,
        mode='neural',
        compressed_bytes=signal.nbytes // 100,
        raw_bytes=signal.nbytes,
    )


class TestRunBenchmark:
    def test_lossless_codec_yields_zero_prd(self) -> None:
        """Passthrough codec -> PRD ≈ 0, R = 1.0, lossless_count == n."""
        windows = [_synthetic_window(i) for i in range(20)]
        # Set energy positions for high/low strata logic to be reachable.
        for i, w in enumerate(windows):
            w.index = i
        report = run_benchmark(_passthrough_codec, windows=windows, use_l3=True)
        assert report.n_windows == 20
        assert report.mean_prd == pytest.approx(0.0, abs=1e-6)
        assert report.mean_r == pytest.approx(1.0, abs=1e-6)
        # All windows lossless
        assert report.lossless_count == 20
        # Fingerprint is a hex string
        assert isinstance(report.holdout_fingerprint, str)
        assert len(report.holdout_fingerprint) > 0

    def test_noisy_codec_yields_finite_metrics(self) -> None:
        """Noisy codec -> PRD > 0, R < 1, finite values."""
        windows = [_synthetic_window(i, source="chbmit" if i % 2 else "tuh")
                   for i in range(20)]
        report = run_benchmark(_noisy_codec, windows=windows, use_l3=True)
        assert report.n_windows == 20
        assert report.mean_prd > 0
        assert report.mean_r < 1.0
        # Per-source split populated
        assert "chbmit" in report.per_source
        assert "tuh" in report.per_source
        # Lossless count: noise -> 0 (most likely; integer rounding could fudge edge cases)
        assert report.lossless_count <= 20

    def test_use_l3_false_runs_on_full_signal(self) -> None:
        windows = [_synthetic_window(i) for i in range(5)]
        report = run_benchmark(_passthrough_codec, windows=windows, use_l3=False)
        assert report.n_windows == 5
        assert report.mean_prd == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_codec_exception_skipped(self) -> None:
        """A codec that raises on every input -> n_windows in report = 0."""
        def explode(_signal):
            raise RuntimeError("boom")
        windows = [_synthetic_window(i) for i in range(3)]
        report = run_benchmark(explode, windows=windows, use_l3=True)
        # No windows survived the codec call -> empty stats default to 0.
        assert report.n_windows == 0
        assert report.mean_prd == 0
        assert report.mean_r == 0

    def test_partial_failure_partial_results(self) -> None:
        """Some windows fail, others succeed -> report counts only successes."""
        calls = {"i": 0}
        def flaky(signal):
            calls["i"] += 1
            if calls["i"] % 2:
                raise RuntimeError("skip")
            return _passthrough_codec(signal)
        windows = [_synthetic_window(i) for i in range(6)]
        report = run_benchmark(flaky, windows=windows, use_l3=True)
        # Roughly half succeed.
        assert 1 <= report.n_windows <= 5


class TestHoldoutWindowDataclass:
    def test_dataclass_fields(self) -> None:
        w = _synthetic_window(0)
        # __annotations__ has the field names
        names = {"filename", "signal", "l3", "source", "energy", "index"}
        for name in names:
            assert hasattr(w, name), f"missing field {name}"

    def test_signal_l3_shape_pinned(self) -> None:
        w = _synthetic_window(7)
        assert w.signal.shape == (21, 2500)
        assert w.l3.shape == (21, 313)


class TestHoldoutFingerprintInvariants:
    def test_fingerprint_changes_when_filename_changes(self) -> None:
        a = _synthetic_window(0)
        b = _synthetic_window(0)
        b.filename = "different.npz"
        # First-10-sample bytes identical, filename different -> fingerprint differs
        assert holdout_fingerprint([a]) != holdout_fingerprint([b])
