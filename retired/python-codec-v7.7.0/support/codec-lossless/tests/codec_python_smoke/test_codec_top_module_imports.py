"""Smoke imports for ``lamquant_codec/*.py`` top-level files that
currently sit at 0% coverage. Asserts on shape/surface only, not on
implementation-derived numeric values.
"""
from __future__ import annotations

import importlib
import types

import pytest

CODEC_TOP_MODULES = [
    "lamquant_codec.cli_codec",
    "lamquant_codec.cli_entry",
    "lamquant_codec.decode",
    "lamquant_codec.export",
    "lamquant_codec.file_info",
    "lamquant_codec.holdout",
    "lamquant_codec.setup_cmd",
]


@pytest.mark.parametrize("mod_name", CODEC_TOP_MODULES)
def test_top_module_is_importable(mod_name: str) -> None:
    mod = importlib.import_module(mod_name)
    assert isinstance(mod, types.ModuleType)
    assert mod.__name__ == mod_name


@pytest.mark.parametrize("mod_name", CODEC_TOP_MODULES)
def test_top_module_exposes_surface(mod_name: str) -> None:
    mod = importlib.import_module(mod_name)
    public = [n for n in dir(mod) if not n.startswith("_")]
    assert public, f"{mod_name} appears empty"
