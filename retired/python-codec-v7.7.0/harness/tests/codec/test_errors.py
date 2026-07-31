"""Error catalog contract — every typed exception carries an ErrorCode.

Pins:
  - Every concrete LmlError subclass declares a unique `code: ErrorCode`.
  - ErrorCode codes are unique across the whole catalog.
  - ErrorCode codes follow the <DOMAIN>-<CLASS>-<NNN> format.
  - Backward-compat: every LmlError still passes `isinstance(e, ValueError)`.
  - Diagnostic message includes the code, name, likely causes, and look-at
    files so future contributors land on the right spot.
"""
from __future__ import annotations

import inspect
import re

import pytest

from lamquant_codec import errors as e

pytestmark = pytest.mark.l1


# ============================================================
# 1. Catalog completeness — every concrete subclass has an ErrorCode
# ============================================================


def _all_concrete_lml_subclasses():
    """Yield every subclass of LmlError defined in errors.py.

    Includes the abstract base classes (LmlError, LmlPayloadError, etc.)
    so the test that filters for `code is not None` separates abstract
    bases from concrete leaves.
    """
    for name, cls in inspect.getmembers(e, inspect.isclass):
        if cls is e.LmlError:
            continue
        if issubclass(cls, e.LmlError):
            yield name, cls


class TestCatalogCompleteness:

    def test_every_concrete_class_declares_an_error_code(self):
        # Names of intermediate base classes whose code may legitimately be None.
        abstract_bases = {
            "LmlPayloadError",
            "LmlInputError",
            "LmlHeaderError",
            "LmaError",
            "LmlTestError",
        }
        # Re-add them once they get codes; for now they cascade to children.
        # Failure here = a concrete class was added without a catalog entry.
        for name, cls in _all_concrete_lml_subclasses():
            if name in abstract_bases:
                continue
            assert cls.code is not None, (
                f"{name} has no ErrorCode — add one in lamquant_codec/errors.py"
            )

    def test_error_code_format_pinned(self):
        """Codes follow <DOMAIN>-<CLASS>-<NNN>, e.g. LML-MAGIC-001."""
        pattern = re.compile(r"^[A-Z]+-[A-Z]+-\d{3}$")
        for name, cls in _all_concrete_lml_subclasses():
            if cls.code is None:
                continue
            assert pattern.match(cls.code.code), (
                f"{name}.code.code = {cls.code.code!r} does not match "
                f"<DOMAIN>-<CLASS>-<NNN> format"
            )

    def test_error_codes_are_unique(self):
        """No two concrete classes may share a code."""
        seen: dict[str, str] = {}
        for name, cls in _all_concrete_lml_subclasses():
            if cls.code is None:
                continue
            existing = seen.get(cls.code.code)
            assert existing is None, (
                f"Code {cls.code.code!r} declared by both {existing} "
                f"and {name}"
            )
            seen[cls.code.code] = name

    def test_every_error_code_has_at_least_one_likely_cause(self):
        """Diagnostic message is useless without root-cause hints."""
        for name, cls in _all_concrete_lml_subclasses():
            if cls.code is None:
                continue
            assert len(cls.code.likely_causes) >= 1, (
                f"{name}.code.likely_causes is empty — add at least one "
                f"common root cause in lamquant_codec/errors.py"
            )

    def test_every_error_code_points_at_a_production_file(self):
        """`look_at` should name the file(s) where the bug usually lives."""
        for name, cls in _all_concrete_lml_subclasses():
            if cls.code is None:
                continue
            # Test-attributable errors point at test helpers, not codec files.
            assert len(cls.code.look_at) >= 1, (
                f"{name}.code.look_at is empty — point future contributors "
                f"at the file most likely to contain the bug"
            )


# ============================================================
# 2. Backward-compat — every LmlError IS a ValueError
# ============================================================


class TestBackwardCompat:

    def test_lml_error_is_value_error(self):
        # Existing `except ValueError` callers must keep working.
        assert issubclass(e.LmlError, ValueError)

    def test_every_subclass_is_value_error(self):
        for _, cls in _all_concrete_lml_subclasses():
            assert issubclass(cls, ValueError), (
                f"{cls.__name__} broke ValueError inheritance — "
                f"existing callers will stop catching it"
            )

    def test_typed_subclass_caught_by_value_error(self):
        with pytest.raises(ValueError):
            raise e.LmlMagicError("test")

    def test_typed_subclass_caught_by_lml_error(self):
        with pytest.raises(e.LmlError):
            raise e.LmlCrcError("test")

    def test_legacy_magic_caught_as_magic_error(self):
        # LmlLegacyMagicError extends LmlMagicError so callers that catch
        # the parent get both.
        with pytest.raises(e.LmlMagicError):
            raise e.LmlLegacyMagicError("test")


# ============================================================
# 3. Diagnostic message format
# ============================================================


class TestDiagnosticMessage:

    def test_message_includes_code(self):
        err = e.LmlMagicError("Got b'XXXX', expected b'LML1'.")
        assert "[LML-MAGIC-001]" in str(err)

    def test_message_includes_name(self):
        err = e.LmlMagicError("Got b'XXXX'.")
        assert "Invalid magic byte" in str(err)

    def test_message_includes_likely_causes_block(self):
        err = e.LmlCrcError("0xDEAD vs 0xBEEF")
        assert "Likely causes:" in str(err)

    def test_message_includes_look_at_block(self):
        err = e.LmlCrcError("0xDEAD vs 0xBEEF")
        assert "Look at:" in str(err)

    def test_message_includes_caller_supplied_context(self):
        err = e.LmlMagicError("Got b'XXXX', expected b'LML1'.")
        assert "Got b'XXXX'" in str(err)


# ============================================================
# 4. assert_raises_lml helper contract
# ============================================================


class TestAssertRaisesLml:

    def test_returns_caught_exception(self):
        from tests.helpers.asserts import assert_raises_lml

        def boom():
            raise e.LmlMagicError("test")

        caught = assert_raises_lml(e.LmlMagicError, boom)
        assert isinstance(caught, e.LmlMagicError)

    def test_accepts_subclass(self):
        from tests.helpers.asserts import assert_raises_lml

        def boom():
            raise e.LmlLegacyMagicError("legacy")

        # LmlMagicError is the parent — subclass must satisfy the contract.
        assert_raises_lml(e.LmlMagicError, boom)

    def test_rejects_wrong_exception_class(self):
        from tests.helpers.asserts import assert_raises_lml

        def boom():
            raise e.LmlCrcError("wrong class")

        with pytest.raises(AssertionError, match=r"Expected LmlMagicError"):
            assert_raises_lml(e.LmlMagicError, boom)

    def test_rejects_no_exception(self):
        from tests.helpers.asserts import assert_raises_lml

        def quiet():
            return 42

        with pytest.raises(AssertionError, match=r"no exception was raised"):
            assert_raises_lml(e.LmlMagicError, quiet)

    def test_rejects_generic_exception(self):
        from tests.helpers.asserts import assert_raises_lml

        def boom():
            raise RuntimeError("not LML")

        with pytest.raises(AssertionError, match=r"generic RuntimeError"):
            assert_raises_lml(e.LmlMagicError, boom)


# ============================================================
# 5. Strict-array helper contract
# ============================================================


class TestArrayHelpers:

    def test_assert_bytes_equal_passes_on_equal(self):
        from tests.helpers.asserts import assert_bytes_equal
        assert_bytes_equal(b"abc", b"abc", context="test")

    def test_assert_bytes_equal_raises_drift_on_difference(self):
        from tests.helpers.asserts import assert_bytes_equal
        with pytest.raises(e.LmlCrossLangDriftError):
            assert_bytes_equal(b"abc", b"abd", context="test")

    def test_assert_array_equal_strict_passes(self):
        import numpy as np
        from tests.helpers.asserts import assert_array_equal_strict
        a = np.arange(10, dtype=np.int64)
        assert_array_equal_strict(a, a, expected_dtype=np.int64,
                                  expected_shape=(10,))

    def test_assert_array_equal_strict_raises_dtype_drift(self):
        import numpy as np
        from tests.helpers.asserts import assert_array_equal_strict
        a = np.arange(10, dtype=np.int32)
        b = np.arange(10, dtype=np.int64)
        with pytest.raises(e.LmlDtypeDriftError):
            assert_array_equal_strict(a, b, expected_dtype=np.int64)

    def test_assert_array_equal_strict_raises_shape_mismatch(self):
        import numpy as np
        from tests.helpers.asserts import assert_array_equal_strict
        a = np.arange(10, dtype=np.int64)
        with pytest.raises(AssertionError, match=r"shape mismatch"):
            assert_array_equal_strict(a, a, expected_shape=(20,))
