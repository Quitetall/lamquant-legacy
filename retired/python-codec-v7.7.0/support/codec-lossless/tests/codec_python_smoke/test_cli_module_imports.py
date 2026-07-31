"""Smoke imports for ``lamquant_codec.cli.*``.

Per ``feedback_futureproof_tests``: pin shape, not implementation values.
Each test loads a CLI sub-module and asserts only that it's a module
object with at least one callable. The point is to exercise the
import-time code path (~5-30% module-coverage per module) so refactors
that break top-of-file constants or shim imports get caught here even
if no inline call is asserted.

These tests deliberately do NOT exercise interactive paths
(``prompt_toolkit``, ``rich``, TTY-bound code). They check the import +
public-surface shape only.
"""
from __future__ import annotations

import importlib
import types

import pytest

CLI_SUBMODULES = [
    "lamquant_codec.cli.backend",
    "lamquant_codec.cli.box",
    "lamquant_codec.cli.cockpit",
    "lamquant_codec.cli.compress",
    "lamquant_codec.cli.config",
    "lamquant_codec.cli.eagle",
    "lamquant_codec.cli.menu",
    "lamquant_codec.cli.readout",
    "lamquant_codec.cli.state",
    "lamquant_codec.cli.syscheck",
    "lamquant_codec.cli.terminal",
]


@pytest.mark.parametrize("mod_name", CLI_SUBMODULES)
def test_module_is_importable(mod_name: str) -> None:
    """Module-load smoke: confirms top-level definitions evaluate cleanly."""
    mod = importlib.import_module(mod_name)
    assert isinstance(mod, types.ModuleType)
    assert mod.__name__ == mod_name


@pytest.mark.parametrize("mod_name", CLI_SUBMODULES)
def test_module_exposes_callable_or_class(mod_name: str) -> None:
    """Each CLI sub-module should expose at least one public callable
    (function/class). The exact name is not pinned — the contract is
    'this module has a public API surface'.
    """
    mod = importlib.import_module(mod_name)
    public_callables = [
        name for name in dir(mod)
        if not name.startswith("_") and callable(getattr(mod, name))
    ]
    assert public_callables, (
        f"{mod_name} has no public callables; module appears to be empty"
    )
