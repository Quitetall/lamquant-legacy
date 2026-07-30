"""Functional tests for ``lamquant_codec.ops.noise``.

Pins INVARIANTS, not numeric values:
  - return type is int (or list/dict per function)
  - return value is bounded in [0, max_bits]
  - constant signal => 0 noise bits
  - random integer signal with N random LSBs => detected noise_bits ≈ N

The exact numeric returns of estimate_noise_bits depend on the
autocorrelation cutoff + sample seed and may drift with refactors —
tests verify shape and reasonable order-of-magnitude only.
"""
from __future__ import annotations

import numpy as np
import pytest

from lamquant_codec.ops.noise import (
    estimate_noise_bits,
    estimate_noise_bits_batch,
    noise_profile,
)


def test_returns_int() -> None:
    sig = np.random.RandomState(42).randint(-1000, 1000, size=1024,
                                              dtype=np.int32)
    out = estimate_noise_bits(sig)
    assert isinstance(out, int)


def test_constant_signal_zero_noise_bits() -> None:
    """A constant signal has no detectable noise floor."""
    sig = np.full(1024, 42, dtype=np.int32)
    assert estimate_noise_bits(sig) == 0


def test_too_short_returns_zero() -> None:
    """n < 64 samples is below the analysis threshold."""
    sig = np.array([1, 2, 3, 4], dtype=np.int32)
    assert estimate_noise_bits(sig) == 0


def test_noise_bits_bounded_by_max_bits() -> None:
    """noise_bits must be in [0, max_bits]."""
    sig = np.random.RandomState(0).randint(
        -2**15, 2**15, size=2048, dtype=np.int32)
    for max_bits in (4, 8, 10):
        nb = estimate_noise_bits(sig, max_bits=max_bits)
        assert 0 <= nb <= max_bits


def test_random_lsbs_detected() -> None:
    """A clean ramp + random LSBs should detect noise on the LSBs.

    Order-of-magnitude only — exact bit count drifts with the
    autocorrelation cutoff.
    """
    rng = np.random.RandomState(7)
    n = 2048
    ramp = (np.arange(n) << 4)  # signal in bits 4+
    noise = rng.randint(0, 16, size=n)  # noise in bits 0-3
    sig = (ramp + noise).astype(np.int32)
    nb = estimate_noise_bits(sig)
    assert nb >= 1  # at least the LSB should look noisy


def test_batch_returns_list_of_ints() -> None:
    sigs = [np.random.RandomState(s).randint(-100, 100, size=256,
                                              dtype=np.int32)
            for s in range(3)]
    out = estimate_noise_bits_batch(sigs)
    assert isinstance(out, list)
    assert len(out) == 3
    assert all(isinstance(x, int) for x in out)


def test_noise_profile_returns_dict() -> None:
    sig = np.random.RandomState(1).randint(-500, 500, size=1024,
                                            dtype=np.int32)
    out = noise_profile(sig)
    assert isinstance(out, dict)


def test_2d_signal_handled() -> None:
    """[C, T] shape should work via flattening."""
    sig = np.random.RandomState(2).randint(-500, 500, size=(4, 512),
                                            dtype=np.int32)
    nb = estimate_noise_bits(sig)
    assert isinstance(nb, int)
    assert 0 <= nb <= 10
