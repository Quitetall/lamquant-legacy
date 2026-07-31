"""Functional tests for ``lamquant_codec.holdout``.

Pins contract:
  - HoldoutWindow is a dataclass with the documented field set
  - holdout_fingerprint is deterministic over a fixed input
  - load_holdout raises FileNotFoundError on missing dir
  - BenchmarkReport class exists and exposes a public surface

No real holdout corpus is required — the tests use synthetic
HoldoutWindow instances.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from lamquant_codec.holdout import (
    BenchmarkReport,
    HoldoutWindow,
    holdout_fingerprint,
    load_holdout,
)


def _synthetic_window(idx: int, source: str = "chbmit") -> HoldoutWindow:
    rng = np.random.RandomState(idx)
    return HoldoutWindow(
        filename=f"synth_{idx:03d}.npz",
        signal=rng.randn(21, 2500).astype(np.float32),
        l3=rng.randn(21, 313).astype(np.float32),
        source=source,
        energy=float(idx),
        index=idx,
    )


class TestHoldoutWindow:
    def test_constructs_with_required_fields(self) -> None:
        w = _synthetic_window(0)
        assert w.filename == "synth_000.npz"
        assert w.signal.shape == (21, 2500)
        assert w.l3.shape == (21, 313)
        assert w.source == "chbmit"
        assert isinstance(w.energy, float)
        assert w.index == 0


class TestHoldoutFingerprint:
    def test_returns_hex_string(self) -> None:
        wins = [_synthetic_window(i) for i in range(3)]
        fp = holdout_fingerprint(wins)
        assert isinstance(fp, str)
        # Contract is "hex digest" — don't pin truncation length;
        # implementation may shorten the prefix without breaking semantics.
        assert len(fp) > 0
        int(fp, 16)  # parses as hex

    def test_deterministic(self) -> None:
        wins1 = [_synthetic_window(i) for i in range(3)]
        wins2 = [_synthetic_window(i) for i in range(3)]
        assert holdout_fingerprint(wins1) == holdout_fingerprint(wins2)

    def test_order_sensitive(self) -> None:
        a = [_synthetic_window(i) for i in range(3)]
        b = [_synthetic_window(i) for i in (2, 0, 1)]
        assert holdout_fingerprint(a) != holdout_fingerprint(b)


class TestLoadHoldout:
    def test_missing_dir_raises_filenotfounderror(self, tmp_path: Path) -> None:
        bogus = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError, match="holdout dataset"):
            load_holdout(data_dir=str(bogus))


class TestBenchmarkReport:
    def test_class_exists(self) -> None:
        # Construct on minimal-positional/keyword args is tolerant —
        # we only pin that the symbol resolves to a class.
        assert isinstance(BenchmarkReport, type)
