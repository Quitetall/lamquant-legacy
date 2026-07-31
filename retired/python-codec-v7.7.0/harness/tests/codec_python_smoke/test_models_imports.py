"""Smoke imports for ``lamquant_codec/models/*`` + ``ops/*`` modules
currently below 30% coverage. Asserts on public surface shape only.
"""
from __future__ import annotations

import importlib
import types

import pytest

MODEL_MODULES = [
    "lamquant_codec.models.siren",
    "lamquant_codec.models.snn",
    "lamquant_codec.ops.noise",
    "lamquant_codec.ops.bias",
    "lamquant_codec.ops.pipeline",
]


@pytest.mark.parametrize("mod_name", MODEL_MODULES)
def test_module_importable(mod_name: str) -> None:
    mod = importlib.import_module(mod_name)
    assert isinstance(mod, types.ModuleType)
    assert mod.__name__ == mod_name


@pytest.mark.parametrize("mod_name", MODEL_MODULES)
def test_module_has_public_surface(mod_name: str) -> None:
    mod = importlib.import_module(mod_name)
    public = [n for n in dir(mod) if not n.startswith("_")]
    assert public, f"{mod_name} has no public symbols"


def test_siren_exposes_constructor() -> None:
    """SIREN module should expose at least one Module/class. Don't pin
    the class name — the contract is 'has a neural Module'."""
    import torch.nn as nn
    from lamquant_codec.models import siren
    classes = [
        getattr(siren, n) for n in dir(siren)
        if not n.startswith("_")
        and isinstance(getattr(siren, n), type)
        and issubclass(getattr(siren, n), nn.Module)
    ]
    assert classes, "siren module exposes no nn.Module subclass"


def test_snn_exposes_loader_or_class() -> None:
    """``lamquant_codec.models.snn`` is the production-SNN registry/loader
    (NOT a model-definition module — the actual MambaSNN class lives
    in ``mamba_ssm_minimal``). Pin the contract as "exposes at least
    one callable" — the loader functions or any registry class
    satisfy this without locking the test to a specific name.
    """
    from lamquant_codec.models import snn
    callables = [
        getattr(snn, n) for n in dir(snn)
        if not n.startswith("_") and callable(getattr(snn, n))
    ]
    assert callables, "snn module exposes no callable"
