"""Lossless roundtrip assertion with diagnostic failure messages.

Extracted from 6 independent _roundtrip() methods across test_lml_paranoid.py,
codec/test_l7_paranoid.py, test_lml_conformance.py, etc. (2026-04-28).

The old pattern: each test class defined its own _roundtrip(self, signal) with
varying levels of diagnostic output. Some had bare `assert np.array_equal(a, b)`,
others had max_diff reporting. This module provides one canonical implementation.
"""
import numpy as np
import pytest

import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(_REPO, 'reference_implementations', 'python_codec', 'lamquant_codec'))

from lamquant_codec.lossless import _compress_bytes, _decompress_bytes


def assert_lml_roundtrip(signal, *, n_levels=3, label=""):
    """Compress → decompress → compare. Fail with full diagnostics on any difference.

    Parameters
    ----------
    signal : array-like
        Input signal, any shape. Converted to float64.
    n_levels : int
        Lifting decomposition levels (default 3).
    label : str
        Optional label included in failure message for context.

    Raises
    ------
    pytest.fail
        If the roundtrip is not bit-exact at integer precision.
    """
    signal = np.asarray(signal, dtype=np.float64)
    compressed = _compress_bytes(signal, n_levels=n_levels)
    decompressed = _decompress_bytes(compressed)

    assert decompressed.shape == signal.shape, (
        f"{'[' + label + '] ' if label else ''}"
        f"Shape mismatch: input {signal.shape} → output {decompressed.shape}"
    )

    sig_int = np.round(signal).astype(np.int64)
    dec_int = np.round(decompressed).astype(np.int64)

    if not np.array_equal(sig_int, dec_int):
        diff = np.abs(sig_int - dec_int)
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        pytest.fail(
            f"{'[' + label + '] ' if label else ''}"
            f"BIT MISMATCH: shape={signal.shape}, "
            f"max_diff={diff.max()}, n_diff={np.count_nonzero(diff)}, "
            f"at {idx}: original={sig_int[idx]}, decoded={dec_int[idx]}"
        )


def assert_lml_compression_valid(signal, *, n_levels=3):
    """Verify compressed output has valid LML1 structure.

    Checks: non-empty, contains LML1 magic after ASCII prefix, roundtrips.
    Returns the compressed bytes for further inspection.
    """
    signal = np.asarray(signal, dtype=np.float64)
    compressed = _compress_bytes(signal, n_levels=n_levels)

    assert len(compressed) > 0, "Compression produced empty output"

    nl = compressed.index(b'\n')
    assert compressed[nl + 1:nl + 5] == b'LML1', (
        f"Expected LML1 magic after ASCII prefix, "
        f"got {compressed[nl + 1:nl + 5]!r}"
    )

    # Verify roundtrip
    assert_lml_roundtrip(signal, n_levels=n_levels)

    return compressed
