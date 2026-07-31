"""Smoke import for ``lamquant_codec.legacy.lossless_legacy``.

The legacy module is preserved-but-not-edited. This test only checks
the file evaluates cleanly — no behavior pinned, no deprecation path
exercised.
"""
from __future__ import annotations

import importlib
import types

import pytest


def test_legacy_module_importable() -> None:
    mod = importlib.import_module(
        "lamquant_codec.legacy.lossless_legacy"
    )
    assert isinstance(mod, types.ModuleType)


def test_legacy_module_exposes_surface() -> None:
    from lamquant_codec.legacy import lossless_legacy as mod
    public = [n for n in dir(mod) if not n.startswith("_")]
    assert public, "lossless_legacy is empty"
