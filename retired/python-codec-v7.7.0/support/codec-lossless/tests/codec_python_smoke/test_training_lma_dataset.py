"""Smoke import + surface check for ``lamquant_codec.training.lma_dataset``.

The LMA-direct dataset is the post-ADR-0017 training pipeline. This
test pins its public class surface (must expose at least one Dataset
subclass) without asserting on internal field names — those will
drift as the dataset evolves.
"""
from __future__ import annotations

import importlib
import types

import pytest


def test_module_importable() -> None:
    mod = importlib.import_module("lamquant_codec.training.lma_dataset")
    assert isinstance(mod, types.ModuleType)


def test_exposes_dataset_subclass() -> None:
    """The module must export at least one ``torch.utils.data.Dataset``
    subclass — that's the LMA-direct training contract."""
    from torch.utils.data import Dataset
    from lamquant_codec.training import lma_dataset
    datasets = [
        getattr(lma_dataset, n) for n in dir(lma_dataset)
        if not n.startswith("_")
        and isinstance(getattr(lma_dataset, n), type)
        and issubclass(getattr(lma_dataset, n), Dataset)
    ]
    assert datasets, "lma_dataset exposes no Dataset subclass"


def test_training_package_imports() -> None:
    """``lamquant_codec.training`` package itself loads."""
    pkg = importlib.import_module("lamquant_codec.training")
    assert pkg.__name__ == "lamquant_codec.training"
