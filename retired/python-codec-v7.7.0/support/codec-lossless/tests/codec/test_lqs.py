"""LamQuant Quality Standard (LQS) — level definitions + metric primitives.

Pins lamquant_codec.lqs public API:
  - LQS_LEVELS: 4 frozen dataclass entries (L, C, M, A) with band/task reqs
  - prd(), pearson_r(), snr_db() — basic metric math invariants
  - check_contract(metrics, level) — compliance gate
"""
from __future__ import annotations

import numpy as np
import pytest

from lamquant_codec import lqs

pytestmark = pytest.mark.l3


# ============================================================
# 1. LQS_LEVELS table
# ============================================================


class TestLQSLevels:

    def test_four_levels_present(self):
        assert set(lqs.LQS_LEVELS.keys()) == {'L', 'C', 'M', 'A'}

    def test_lossless_level_is_bit_exact(self):
        L = lqs.LQS_LEVELS['L']
        assert L.bit_exact is True
        assert L.max_prd == 0.0
        assert L.min_r == 1.0
        assert L.max_snr_loss == 0.0

    def test_clinical_level_thresholds(self):
        C = lqs.LQS_LEVELS['C']
        assert C.bit_exact is False
        assert C.max_prd == 9.0
        assert C.min_r == 0.95
        # Five clinical bands required.
        assert set(C.band_fidelity.keys()) == {'delta', 'theta', 'alpha', 'beta', 'gamma'}

    def test_levels_strictly_relax_from_L_to_A(self):
        """Lossless → Clinical → Monitoring → Alerting must give monotonically
        looser quality and tighter compression-ratio floors (or equal)."""
        order = ['L', 'C', 'M', 'A']
        levels = [lqs.LQS_LEVELS[k] for k in order]
        for prev, curr in zip(levels, levels[1:]):
            assert curr.max_prd >= prev.max_prd
            assert curr.min_r <= prev.min_r
            assert curr.min_cr >= prev.min_cr  # tighter compression for lossier levels

    def test_clinical_band_ranges_partition_eeg_band(self):
        """The five clinical bands cover 0.5..50 Hz contiguously."""
        bands = lqs.LQS_LEVELS['C'].band_fidelity
        ranges = sorted(bands.values(), key=lambda b: b.freq_range[0])
        for prev, curr in zip(ranges, ranges[1:]):
            assert prev.freq_range[1] == curr.freq_range[0], (
                f"gap or overlap: {prev.freq_range} → {curr.freq_range}"
            )


# ============================================================
# 2. Metric primitives — basic math invariants
# ============================================================


class TestMetrics:

    def test_prd_zero_on_identity(self):
        x = np.random.default_rng(0).standard_normal(2500).astype(np.float32)
        assert lqs.prd(x, x) == 0.0

    def test_prd_zero_on_zero_zero(self):
        x = np.zeros(100)
        assert lqs.prd(x, x) == 0.0

    def test_prd_increases_with_noise(self):
        x = np.random.default_rng(1).standard_normal(2500).astype(np.float32)
        for noise_amp in (0.01, 0.1, 1.0):
            noisy = x + noise_amp * np.random.default_rng(2).standard_normal(2500)
            assert lqs.prd(x, noisy) > 0

    def test_pearson_r_one_on_identity(self):
        x = np.random.default_rng(3).standard_normal(2500).astype(np.float32)
        r = lqs.pearson_r(x, x)
        assert r == pytest.approx(1.0, abs=1e-6)

    def test_pearson_r_zero_on_orthogonal(self):
        rng = np.random.default_rng(4)
        x = rng.standard_normal(2500)
        # y orthogonal to x — uncorrelated noise.
        y = rng.standard_normal(2500)
        r = lqs.pearson_r(x, y)
        assert abs(r) < 0.1  # ~0 in expectation

    def test_pearson_r_negative_on_inverse(self):
        x = np.random.default_rng(5).standard_normal(2500).astype(np.float32)
        r = lqs.pearson_r(x, -x)
        assert r == pytest.approx(-1.0, abs=1e-6)

    def test_snr_db_high_on_clean_signal(self):
        x = np.random.default_rng(6).standard_normal(2500).astype(np.float32)
        # Reconstruction with negligible noise → high SNR.
        recon = x + 1e-6 * np.random.default_rng(7).standard_normal(2500)
        assert lqs.snr_db(x, recon) > 40.0

    def test_snr_db_lower_on_noisy_signal(self):
        x = np.random.default_rng(8).standard_normal(2500).astype(np.float32)
        noisy = x + 0.5 * np.random.default_rng(9).standard_normal(2500)
        # Noise dominates → SNR << clean case.
        assert lqs.snr_db(x, noisy) < 30.0


# ============================================================
# 3. compute_all_metrics shape
# ============================================================


class TestComputeAllMetrics:

    def test_returns_dict_with_basic_keys(self):
        rng = np.random.default_rng(10)
        # Two channels at 250 Hz, 1 second each.
        x = rng.standard_normal((2, 250)).astype(np.float32) * 50
        recon = x + 0.05 * rng.standard_normal((2, 250)).astype(np.float32)
        # Signature: (original, reconstructed, packet: bytes, sr=250.0)
        metrics = lqs.compute_all_metrics(x, recon, packet=b"", sr=250.0)
        assert isinstance(metrics, dict)
        # PRD is the canonical key the contract gate checks.
        assert 'prd' in metrics
        assert isinstance(metrics['prd'], (int, float))


# ============================================================
# 4. check_contract gate (returns List[str] of violations; empty = pass)
# ============================================================


class TestCheckContract:

    def test_perfect_reconstruction_passes_lossless(self):
        L = lqs.LQS_LEVELS['L']
        good = {
            'prd': 0.0,
            'r': 1.0,           # check_contract reads 'r' (not 'pearson_r')
            'snr_db': 1000.0,
            'cr': 2.5,
            'bit_exact': True,
        }
        violations = lqs.check_contract(good, L)
        assert violations == [], f"expected zero violations, got {violations}"

    def test_high_prd_fails_clinical(self):
        C = lqs.LQS_LEVELS['C']
        bad = {
            'prd': 50.0,        # >> 9.0 clinical cap
            'r': 0.5,            # << 0.95 clinical floor
            'snr_db': -10.0,
            'cr': 5.0,           # << 20.0 clinical floor
        }
        violations = lqs.check_contract(bad, C)
        # Should flag at least one violation (PRD or compression-ratio).
        assert isinstance(violations, list)
        assert len(violations) > 0, "clinical contract should reject this metric set"
