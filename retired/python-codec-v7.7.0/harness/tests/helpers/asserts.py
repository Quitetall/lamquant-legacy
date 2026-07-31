"""Typed assertion helpers + structured drift reporting.

Replaces stringly-typed `pytest.raises(ValueError, match=r"...")` with
exception-class assertions. When a drift is detected, the failure message
points at the production files most likely to contain the bug.

Usage:

    from tests.helpers.asserts import (
        assert_raises_lml,
        assert_bytes_equal,
        assert_array_equal_strict,
    )

    # Exception class assertion (better than match=r"...")
    assert_raises_lml(LmlMagicError, decompress, b"XXXX")

    # Cross-lang byte drift — failure includes diagnostic from ErrorCode
    assert_bytes_equal(py_bytes, rs_bytes, context="LML1 header")

    # Strict array equality with dtype + shape pinning
    assert_array_equal_strict(actual, expected,
                              expected_dtype=np.int64,
                              expected_shape=(21, 2500))
"""
from __future__ import annotations

from typing import Any, Callable, Type

import numpy as np

from lamquant_codec.errors import (
    LmlCrossLangDriftError,
    LmlDtypeDriftError,
    LmlError,
)

__all__ = [
    "assert_raises_lml",
    "assert_bytes_equal",
    "assert_array_equal_strict",
]


def assert_raises_lml(
    expected_class: Type[LmlError],
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> LmlError:
    """Assert that calling `fn(*args, **kwargs)` raises `expected_class` (or a subclass).

    Returns the caught exception so the caller can do further assertions on
    its message / attributes if needed.

    Why not use `pytest.raises(expected_class, match=r"...")`?  Because regex-
    matching the message is stringly-typed and breaks every time someone
    edits the error copy. The `ErrorCode` system already gives every error
    a stable identifier; assertion should test that, not message text.
    """
    try:
        fn(*args, **kwargs)
    except expected_class as e:
        return e
    except LmlError as e:
        raise AssertionError(
            f"Expected {expected_class.__name__}, got "
            f"{type(e).__name__}:\n  {e}"
        ) from e
    except Exception as e:
        raise AssertionError(
            f"Expected {expected_class.__name__}, got generic "
            f"{type(e).__name__}: {e}"
        ) from e
    raise AssertionError(
        f"Expected {expected_class.__name__}, but no exception was raised. "
        f"Function returned normally."
    )


def assert_bytes_equal(
    actual: bytes,
    expected: bytes,
    *,
    context: str = "bytes",
) -> None:
    """Byte-exact equality. On failure, raise LmlCrossLangDriftError with
    the first divergence offset and a hex preview.
    """
    if actual == expected:
        return
    if len(actual) != len(expected):
        raise LmlCrossLangDriftError(
            f"{context}: lengths differ — actual={len(actual)}, "
            f"expected={len(expected)}"
        )
    for i, (a, e) in enumerate(zip(actual, expected)):
        if a != e:
            preview_actual = actual[max(0, i - 4):i + 4].hex(" ")
            preview_expected = expected[max(0, i - 4):i + 4].hex(" ")
            raise LmlCrossLangDriftError(
                f"{context}: first divergence at offset {i} — "
                f"actual=0x{a:02X}, expected=0x{e:02X}\n"
                f"    actual[i-4:i+4]:   {preview_actual}\n"
                f"    expected[i-4:i+4]: {preview_expected}"
            )


def assert_array_equal_strict(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    expected_dtype: np.dtype | type | None = None,
    expected_shape: tuple[int, ...] | None = None,
    context: str = "array",
) -> None:
    """Strict array equality: shape, dtype, and bit-exact values all checked.

    Loose `np.testing.assert_array_equal` does NOT check dtype — float32
    output can sneak in where int64 was contracted. This helper raises
    `LmlDtypeDriftError` when dtype drifts, and a generic AssertionError
    when values differ.
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)

    if expected_shape is not None and actual.shape != expected_shape:
        raise AssertionError(
            f"{context}: shape mismatch — actual {actual.shape} vs "
            f"contracted {expected_shape}"
        )

    if expected_dtype is not None and actual.dtype != np.dtype(expected_dtype):
        raise LmlDtypeDriftError(
            f"{context}: dtype mismatch — actual {actual.dtype} vs "
            f"contracted {np.dtype(expected_dtype)}"
        )

    if actual.shape != expected.shape:
        raise AssertionError(
            f"{context}: shape mismatch between actual {actual.shape} "
            f"and expected {expected.shape}"
        )

    if actual.dtype != expected.dtype:
        raise LmlDtypeDriftError(
            f"{context}: dtype drift — actual {actual.dtype} vs "
            f"expected {expected.dtype}"
        )

    if not np.array_equal(actual, expected):
        # Summarise the first mismatch for fast triage.
        diff = np.where(actual != expected)
        if len(diff) > 0 and len(diff[0]) > 0:
            idx = tuple(d[0] for d in diff)
            raise AssertionError(
                f"{context}: value drift — first mismatch at index {idx}: "
                f"actual={actual[idx]!r}, expected={expected[idx]!r} "
                f"({len(diff[0])} of {actual.size} elements differ)"
            )
        raise AssertionError(f"{context}: arrays differ but no diff index found")
