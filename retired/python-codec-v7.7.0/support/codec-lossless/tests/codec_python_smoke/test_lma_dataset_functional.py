"""Functional tests for ``lamquant_codec.training.lma_dataset``.

Pins contract:
  - _validate_stem rejects path-traversal patterns + leading-char attacks
  - select_random_windows returns the requested count with values in [0, T)
  - LmaSignalDataset + LmaL3Dataset both subclass torch.utils.data.Dataset
  - public re-exports (TARGET_SR, TARGET_CHANNELS) carry expected types

Doesn't require a real LMA file — focuses on the input-validation +
shape-contract surface. Heavy paths (subprocess to ``lml ls``, NPZ
decoding) are exercised by the integration suite once real data exists.
"""
from __future__ import annotations

import numpy as np
import pytest
from torch.utils.data import Dataset

from lamquant_codec.training import lma_dataset
from lamquant_codec.training.lma_dataset import (
    LmaL3Dataset,
    LmaSignalDataset,
    TARGET_CHANNELS,
    TARGET_SR,
    _validate_stem,
    select_random_windows,
)


class TestValidateStem:
    """``_validate_stem`` is the path-traversal guard. Tests pin the
    SHAPE of rejection (ValueError on bad input, silent pass on good),
    not specific error messages.
    """

    def test_accepts_simple_stem(self) -> None:
        _validate_stem("aaaaaaaq_s001_t000")

    def test_rejects_forward_slash(self) -> None:
        with pytest.raises(ValueError):
            _validate_stem("foo/bar")

    def test_rejects_backslash(self) -> None:
        with pytest.raises(ValueError):
            _validate_stem("foo\\bar")

    def test_rejects_dotdot(self) -> None:
        with pytest.raises(ValueError):
            _validate_stem("..foo")

    def test_rejects_leading_dash(self) -> None:
        with pytest.raises(ValueError):
            _validate_stem("-rm")

    def test_rejects_leading_tilde(self) -> None:
        with pytest.raises(ValueError):
            _validate_stem("~admin")

    def test_rejects_leading_dot(self) -> None:
        with pytest.raises(ValueError):
            _validate_stem(".hidden")

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            _validate_stem("")


class TestSelectRandomWindows:
    """``select_random_windows(n_total, windows_per_epoch, rng)`` returns a
    list of integer indices; full enumeration when ``n_total <=
    windows_per_epoch``, sampled-with-replacement otherwise.
    """

    def test_full_enumeration_when_small(self) -> None:
        import random
        rng = random.Random(0)
        out = select_random_windows(n_total_windows=4,
                                     windows_per_epoch=8, rng=rng)
        # n_total <= windows_per_epoch -> full enumeration
        assert sorted(out) == [0, 1, 2, 3]

    def test_sampled_count_matches_windows_per_epoch(self) -> None:
        import random
        rng = random.Random(1)
        out = select_random_windows(n_total_windows=10_000,
                                     windows_per_epoch=64, rng=rng)
        assert len(out) == 64

    def test_indices_in_range(self) -> None:
        import random
        rng = random.Random(2)
        out = select_random_windows(n_total_windows=500,
                                     windows_per_epoch=32, rng=rng)
        for idx in out:
            assert isinstance(idx, int)
            assert 0 <= idx < 500


class TestDatasetClasses:
    def test_signal_dataset_subclasses_dataset(self) -> None:
        assert issubclass(LmaSignalDataset, Dataset)

    def test_l3_dataset_subclasses_dataset(self) -> None:
        assert issubclass(LmaL3Dataset, Dataset)


class TestModuleConstants:
    def test_target_sr_is_positive_number(self) -> None:
        # Some pipelines store sample rate as float (Hz); don't pin
        # the dtype, just the positive-Hz invariant.
        assert isinstance(TARGET_SR, (int, float))
        assert TARGET_SR > 0

    def test_target_channels_is_positive_int(self) -> None:
        assert isinstance(TARGET_CHANNELS, int)
        assert TARGET_CHANNELS > 0
