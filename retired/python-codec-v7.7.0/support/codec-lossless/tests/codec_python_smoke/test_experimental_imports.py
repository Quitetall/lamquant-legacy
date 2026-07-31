"""Smoke imports for ``lamquant_codec.experimental.*``.

These are research-track modules that may not be production-stable —
the test only confirms the file evaluates without import error and
exposes a public surface. No behavior is pinned.
"""
from __future__ import annotations

import importlib
import types

import pytest

EXPERIMENTAL_MODULES = [
    "lamquant_codec.experimental",
    "lamquant_codec.experimental.learned_entropy",
    "lamquant_codec.experimental.token_compression",
]


@pytest.mark.parametrize("mod_name", EXPERIMENTAL_MODULES)
def test_experimental_module_importable(mod_name: str) -> None:
    mod = importlib.import_module(mod_name)
    assert isinstance(mod, types.ModuleType)
    assert mod.__name__ == mod_name


@pytest.mark.parametrize(
    "mod_name", ["lamquant_codec.experimental.learned_entropy",
                 "lamquant_codec.experimental.token_compression"])
def test_experimental_exposes_callable(mod_name: str) -> None:
    mod = importlib.import_module(mod_name)
    callables = [
        name for name in dir(mod)
        if not name.startswith("_") and callable(getattr(mod, name))
    ]
    assert callables, f"{mod_name} exposes no callable"
