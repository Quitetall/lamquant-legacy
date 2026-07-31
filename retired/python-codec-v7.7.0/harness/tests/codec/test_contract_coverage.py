"""Coverage tests for `lamquant_codec.contract`.

Pins the contract enforcement gate:
  - CONTRACTS table has lossless + neural
  - Violation.__str__ formats predictably
  - check_contract returns [] on lossless EEGPacket round-trip
  - check_contract flags every lossy bound when crossed
  - check_contract_strict raises AssertionError on violations
  - check_contract_strict is silent on a passing packet

Tests use math fixtures (np.random) — no synthetic EEG semantics. The
focus is exercising the gate logic, not the codec it gates.
"""
from __future__ import annotations

import numpy as np
import pytest

from lamquant_codec.contract import (
    CONTRACTS,
    Violation,
    check_contract,
    check_contract_strict,
)
from lamquant_codec.codec_types import EEGPacket

pytestmark = [pytest.mark.l3]


# ============================================================
# Violation dataclass — string formatting
# ============================================================


class TestViolation:

    def test_str_includes_field_expected_actual(self):
        v = Violation(field="prd", expected="<= 9.0%", actual="42.0%")
        rendered = str(v)
        assert "prd" in rendered
        assert "<= 9.0%" in rendered
        assert "42.0%" in rendered

    def test_dataclass_fields_pin(self):
        v = Violation(field="r", expected=">= 0.95", actual="0.1")
        assert v.field == "r"
        assert v.expected == ">= 0.95"
        assert v.actual == "0.1"


# ============================================================
# CONTRACTS table — production presets
# ============================================================


class TestContractsTable:

    def test_lossless_and_neural_present(self):
        assert "lossless" in CONTRACTS
        assert "neural" in CONTRACTS

    def test_lossless_contract_invariants(self):
        c = CONTRACTS["lossless"]
        assert c.mode == "lossless"
        assert c.lossless is True
        assert c.max_prd == 0.0
        assert c.min_r == 1.0
        assert c.bands is None

    def test_neural_contract_invariants(self):
        c = CONTRACTS["neural"]
        assert c.mode == "neural"
        assert c.lossless is False
        # All five clinical bands present.
        assert c.bands is not None
        assert set(c.bands.keys()) == {"delta", "theta", "alpha", "beta", "gamma"}
        # Each band carries a max_prd float.
        for band_name, bounds in c.bands.items():
            assert isinstance(bounds.get("max_prd"), float), band_name
        # CR window is sensible (min < max).
        assert c.min_cr < c.max_cr


# ============================================================
# check_contract — happy path on a perfect lossless reconstruction
# ============================================================


class TestCheckContractPasses:

    def test_lossless_zero_violation_on_bit_exact(self):
        rng = np.random.default_rng(0)
        signal = rng.integers(-1000, 1000, size=(4, 256)).astype(np.float64)
        # EEGPacket.raw_bytes = shape[-2]*shape[-1]*2 = 4*256*2 = 2048.
        # Compressed bytes chosen so CR lands inside [3, 10].
        raw = signal.shape[-2] * signal.shape[-1] * 2
        packet = EEGPacket.from_lossless(
            signal=signal.copy(),
            compressed_bytes=raw // 4,
        )
        violations = check_contract(signal, packet, CONTRACTS["lossless"])
        assert violations == [], f"expected zero violations, got {violations}"

    def test_strict_passes_silently_on_zero_violation(self):
        rng = np.random.default_rng(1)
        signal = rng.integers(-1000, 1000, size=(4, 256)).astype(np.float64)
        raw = signal.shape[-2] * signal.shape[-1] * 2
        packet = EEGPacket.from_lossless(
            signal=signal.copy(),
            compressed_bytes=raw // 4,
        )
        # Must NOT raise.
        check_contract_strict(signal, packet, CONTRACTS["lossless"])


# ============================================================
# check_contract — flags every lossless violation when triggered
# ============================================================


class TestCheckContractLosslessFailures:

    def test_lossless_non_bit_exact_flagged(self):
        rng = np.random.default_rng(2)
        signal = rng.integers(-1000, 1000, size=(2, 64)).astype(np.float64)
        raw = signal.shape[-2] * signal.shape[-1] * 2
        # Reconstruction differs by 1 LSB — lossless contract rejects it.
        bad = signal + 1.0
        packet = EEGPacket.from_lossless(
            signal=bad,
            compressed_bytes=raw // 4,
        )
        violations = check_contract(signal, packet, CONTRACTS["lossless"])
        fields = {v.field for v in violations}
        # Lossless flag + PRD must fire (max_prd=0).
        assert "lossless" in fields
        # PRD will also be > 0, so it should be flagged.
        assert "prd" in fields

    def test_strict_raises_on_violation(self):
        rng = np.random.default_rng(3)
        signal = rng.integers(-1000, 1000, size=(2, 64)).astype(np.float64)
        raw = signal.shape[-2] * signal.shape[-1] * 2
        bad = signal + 5.0
        packet = EEGPacket.from_lossless(
            signal=bad,
            compressed_bytes=raw // 4,
        )
        with pytest.raises(AssertionError) as exc_info:
            check_contract_strict(signal, packet, CONTRACTS["lossless"])
        assert "lossless" in str(exc_info.value)


# ============================================================
# check_contract — exercises CR bounds (min and max)
# ============================================================


class TestCheckContractCR:

    def test_low_cr_violates_min(self):
        rng = np.random.default_rng(4)
        signal = rng.integers(-100, 100, size=(2, 64)).astype(np.float64)
        raw = signal.shape[-2] * signal.shape[-1] * 2
        # Compressed_bytes > raw_bytes → CR < 1 < min_cr (3.0)
        packet = EEGPacket.from_lossless(
            signal=signal.copy(),
            compressed_bytes=raw * 2,
        )
        violations = check_contract(signal, packet, CONTRACTS["lossless"])
        cr_violations = [v for v in violations if v.field == "cr"]
        assert len(cr_violations) >= 1

    def test_high_cr_violates_max(self):
        rng = np.random.default_rng(5)
        signal = rng.integers(-100, 100, size=(2, 64)).astype(np.float64)
        # raw=64*2*2=256, compressed_bytes=1 → CR ~ 256 >> max_cr=10
        packet = EEGPacket.from_lossless(
            signal=signal.copy(),
            compressed_bytes=1,
        )
        violations = check_contract(signal, packet, CONTRACTS["lossless"])
        cr_violations = [v for v in violations if v.field == "cr"]
        assert len(cr_violations) >= 1


# ============================================================
# check_contract — per-band PRD on long enough signal
# ============================================================


class TestCheckContractPerBand:

    def test_per_band_path_exercised_on_long_signal(self):
        """When original.shape[-1] >= 64, per-band PRD branch fires."""
        rng = np.random.default_rng(6)
        signal = rng.standard_normal((4, 256)).astype(np.float64) * 100
        recon = signal.copy()  # identical → all per-band PRDs ~ 0
        raw = signal.shape[-2] * signal.shape[-1] * 2
        packet = EEGPacket.from_lossless(
            signal=recon,
            compressed_bytes=raw // 5,  # CR ~ 5, in window
        )
        # Use the neural contract so the bands dict is present.
        c = CONTRACTS["neural"]
        violations = check_contract(signal, packet, c)
        # Bit-exact recon → PRD=0 < band caps, but neural CR window starts
        # at 200, so cr will fail. We only assert the per-band path doesn't
        # crash and doesn't add band violations.
        band_violations = [v for v in violations
                           if v.field.startswith("band_prd.")]
        assert band_violations == []
