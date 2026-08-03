"""Smoke imports for ai_models modules at 0% coverage.

Per ``feedback_futureproof_tests``: surface-only assertions. No inline
numeric expectations; no implementation-derived equality checks. The
contract pinned here is "module loads + exposes callables".

Excluded:
  - ``ai_models/training_cockpit.py`` — GUI scope (per
    ``feedback_frontend_only_scope``); not the edit target.
"""
from __future__ import annotations

import pytest  # decomp(lossless-carve): skip when ai_models absent
pytest.importorskip("subband_preprocess", reason="Neural-coupled test; requires LamQuant-Neural sibling clone")

import importlib
import types

import pytest

AI_MODELS_MODULES = [
    "ai_models.student.run_diagnostics",
    "ai_models.student.ship_fast_preset",
    "ai_models.training_plotter",
]


@pytest.mark.parametrize("mod_name", AI_MODELS_MODULES)
def test_module_importable(mod_name: str) -> None:
    mod = importlib.import_module(mod_name)
    assert isinstance(mod, types.ModuleType)
    assert mod.__name__ == mod_name


@pytest.mark.parametrize("mod_name", AI_MODELS_MODULES)
def test_module_has_public_surface(mod_name: str) -> None:
    mod = importlib.import_module(mod_name)
    public = [n for n in dir(mod) if not n.startswith("_")]
    assert public, f"{mod_name} appears empty"


def test_run_diagnostics_exposes_metric_functions() -> None:
    """run_diagnostics is a metric-collection script — pins that at
    least one ``metric_*`` callable exists. The exact set of metrics
    drifts as new diagnostics land, so the contract is "has metrics",
    not a specific name list."""
    mod = importlib.import_module("ai_models.student.run_diagnostics")
    metrics = [n for n in dir(mod)
               if n.startswith("metric_")
               and callable(getattr(mod, n))]
    assert metrics, (
        "run_diagnostics exposes no metric_* functions"
    )


def test_ship_fast_preset_exposes_main_or_run() -> None:
    """ship_fast_preset must expose a ``main`` or ``run`` entry."""
    mod = importlib.import_module("ai_models.student.ship_fast_preset")
    entries = [n for n in ("main", "run") if hasattr(mod, n)
               and callable(getattr(mod, n))]
    assert entries, (
        "ship_fast_preset exposes no main/run entry point"
    )
