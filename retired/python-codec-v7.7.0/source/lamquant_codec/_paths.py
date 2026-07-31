"""Centralized repo-root resolver — relocation-resilient.

Pre-2026-05-18, callers across ``lamquant_codec/`` computed the
LamQuant repo root via ``Path(__file__).resolve().parent.parent``
(or ``.parent.parent.parent`` from the ``cli/`` subpackage). That
worked when the package lived at the repo root, but broke when the
package was relocated to ``reference_implementations/python_codec/
lamquant_codec/`` per the repo-cleanup ADR.

This module walks upward from the package's own location until it
finds a ``pyproject.toml`` — a stable marker that survives any
future nesting. Every caller imports ``REPO_ROOT`` from here and
gets the right path regardless of where the package sits.

If no ``pyproject.toml`` is found in any ancestor (e.g., the package
was installed via wheel into ``site-packages``), the resolver falls
back to the current working directory at import time. That matches
the historical assumption that codec tooling is invoked from the
repo root in development.
"""

from __future__ import annotations

from pathlib import Path


def _find_repo_root() -> Path:
    """Walk up from this file looking for ``pyproject.toml``.

    Returns the directory containing the first ``pyproject.toml``
    encountered. On failure (wheel install, packaged installer),
    returns ``Path.cwd()`` so callers continue to work in the
    development-from-repo case the helper is most often called from.
    """
    here = Path(__file__).resolve().parent
    for ancestor in [here, *here.parents]:
        if (ancestor / "pyproject.toml").exists():
            return ancestor
    return Path.cwd()


REPO_ROOT: Path = _find_repo_root()
"""Absolute path to the LamQuant repo root (directory holding ``pyproject.toml``)."""
