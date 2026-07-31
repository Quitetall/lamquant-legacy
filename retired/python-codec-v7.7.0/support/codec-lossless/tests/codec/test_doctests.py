"""Doctest runner — executes every `>>>` example in the documented public API.

Driving doctests through a regular pytest test (rather than
`pytest --doctest-modules`) keeps fast-lane scope small and lets us
white-list which modules to scan. This avoids accidentally running
doctests in modules that don't have any (and would otherwise cost
collection time).

Adding a new module with doctests:
  1. Add `>>> ` examples in its docstrings.
  2. Append the module to `_DOCTEST_MODULES` below.
  3. CI fast lane picks it up automatically on next run.

A failure here means a docstring example diverged from the code's
actual behaviour — the test fails with the offending example shown in
context, so future contributors land on the right docstring directly.
"""
from __future__ import annotations

import doctest

import pytest

# White-listed modules whose docstrings carry runnable examples.
# Order is significant only for failure-attribution clarity; collection
# is independent.
_DOCTEST_MODULES = [
    "lamquant_codec.errors",
    "lamquant_codec.transform_graph",
]

pytestmark = pytest.mark.doctest


@pytest.mark.parametrize("module_name", _DOCTEST_MODULES)
def test_module_doctests(module_name: str):
    """Run every doctest in the named module. Fails loudly on divergence."""
    module = __import__(module_name, fromlist=["*"])
    results = doctest.testmod(
        module,
        verbose=False,
        optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS,
    )
    assert results.failed == 0, (
        f"{module_name} has {results.failed} failing doctest(s) out of "
        f"{results.attempted}. Inspect docstrings — examples diverged "
        f"from actual behaviour."
    )
    # Smoke check: every white-listed module must actually contain
    # at least one doctest. If `attempted == 0`, the module landed on
    # the white list by mistake or its docstrings were stripped.
    assert results.attempted > 0, (
        f"{module_name} is on the doctest white list but has zero `>>>` "
        f"examples. Either add some or remove it from _DOCTEST_MODULES."
    )
