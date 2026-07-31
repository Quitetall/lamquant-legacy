"""Deprecation shim — ``lamquant_codec.cli.cockpit`` moved to ``legacy/`` (ADR 0017).

The 1381-line Python training cockpit that used to live here has
been sequestered to ``legacy/python_cockpit/cockpit.py``. The Rust
TUI cockpit (``lamquant`` binary, ``lamquant-core/src/tui/panels/
cockpit.rs``) is the canonical training cockpit going forward.

This shim re-exports the legacy module's public surface so external
callers (the ``lamquant`` top-level CLI's ``cmd_train`` no-args path
in particular) keep working through the transition window. Direct
imports of ``lamquant_codec.cli.cockpit.cockpit_main`` succeed but
print a ``DeprecationWarning`` on first use.

The shim only resolves in a full repo checkout — the codec wheel
intentionally does NOT ship the ``legacy/`` tree, so a
``pip install lamquant_codec`` deployment will not import this
module's actual logic. See ADR 0017 for the wheel-install contract.
"""

import warnings

warnings.warn(
    "lamquant_codec.cli.cockpit moved to legacy/python_cockpit/ "
    "(ADR 0017). The Rust TUI cockpit (`lamquant`) is the canonical "
    "training cockpit. This shim stays as a compat layer for the "
    "transition window — hard-removal lands in a future sprint.",
    DeprecationWarning,
    stacklevel=2,
)

try:
    # Re-export the full public surface from the sequestered module.
    # The wildcard import grabs `cockpit_main` along with everything
    # else the legacy module exposes; no need to re-name it
    # explicitly.
    from legacy.python_cockpit.cockpit import *  # noqa: F401, F403
except ImportError:
    # Wheel install (no ``legacy/`` tree) — define a stub that prints
    # a clean error pointing at the Rust TUI replacement.
    import sys

    def cockpit_main(*_args, **_kwargs):
        # NOTE: the stub accepts *args/**kwargs intentionally — any
        # caller (including the lamquant.py train fallback path)
        # passes nothing today, but the signature stays permissive
        # so a future caller swapping in test fixtures doesn't have
        # to special-case the wheel-install path.
        print(
            "error: lamquant_codec.cli.cockpit was moved to legacy/\n"
            "       (ADR 0017). The codec wheel does not ship the\n"
            "       legacy/ tree; install LamQuant from a repo checkout\n"
            "       or use the Rust TUI cockpit via `lamquant` instead.",
            file=sys.stderr,
        )
        # Exit code 0: this is the "intentional sequester" path, not
        # an internal error. Scripts that wrapped the pre-ADR
        # `cockpit_main()` already treated a missing cockpit as a
        # warn-then-continue case. Matches the legacy module's
        # contract (`cockpit_main()` returns 0 after a normal exit
        # of the interactive loop).
        return 0
