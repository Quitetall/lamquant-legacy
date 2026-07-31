"""Typed exception hierarchy + diagnostic error catalog — clinical-grade.

Every error raised by the production codec inherits from `LmlError` and
carries an `ErrorCode` that names a unique error class, lists likely
causes, and points future contributors at the production files where the
underlying bug usually lives.

Single editable source of truth: every catalog entry lives in this file.
Tests assert against exception classes (not message regexes) so message
copy-edits never break tests.

Example raised string:

    [LML-MAGIC-001] Invalid magic byte: Got b'XXXX', expected b'LML1'.
      Likely causes:
        - File is not an LML1 packet (truncated upload? wrong format?).
        - Header offset miscomputed by an upstream container reader.
        - Bytes were re-encoded as text and lost the binary magic.
      Look at: lamquant_codec/lossless.py, lamquant-core/src/lml.rs

Usage in production code:

    raise LmlMagicError(f"Got {magic!r}, expected b'LML1'.")

Usage in tests:

    with pytest.raises(LmlMagicError):
        decompress(b"XXXX...")

The full diagnostic message is automatically formatted from the
`ErrorCode` of the raised subclass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Optional, Tuple

__all__ = [
    # Core
    "ErrorCode",
    "LmlError",
    # Codec runtime
    "LmlMagicError",
    "LmlLegacyMagicError",
    "LmlVersionError",
    "LmlTruncatedError",
    "LmlHeaderError",
    "LmlReservedBitsSetError",
    "LmlChannelCountError",
    "LmlPayloadError",
    "LmlCrcError",
    "LmlNoiseStrippedError",
    "LmlInputError",
    "LmlSignalShapeError",
    "LmlEmptySignalError",
    # Archive (LMA)
    "LmaError",
    "LmaMagicError",
    "LmaManifestError",
    "LmaShaMismatchError",
    "LmaUnknownMethodError",
    # Adaptive FSQ — SNN-driven per-timestep level scheduling
    "AdaptiveFSQError",
    # Test-attributable failures (used by helpers + assertions)
    "LmlTestError",
    "LmlCrossLangDriftError",
    "LmlDeterminismError",
    "LmlDtypeDriftError",
    "LmlResourceLeakError",
    "LmlTestFixtureMissingError",
]


# ============================================================
# Diagnostic catalog — one ErrorCode per concrete error class
# ============================================================


@dataclass(frozen=True)
class ErrorCode:
    """A single attributable error class.

    Fields:
      code:          stable short identifier shown to operators (LML-MAGIC-001).
                     Format: <DOMAIN>-<CLASS>-<NNN>.
                     Stable across releases. Never re-use a code.
      name:          one-line human description.
      likely_causes: tuple of bullet strings explaining the most common
                     reasons this error fires. Ordered by frequency.
      look_at:       tuple of file paths future contributors should
                     inspect when triaging this code.
      docs_url:      optional spec section reference.

    >>> code = ErrorCode(
    ...     code="LML-DEMO-001", name="Demo error",
    ...     likely_causes=("first cause",),
    ...     look_at=("path/to/file.py",),
    ... )
    >>> msg = code.format_diagnostic("specific context here")
    >>> "[LML-DEMO-001]" in msg
    True
    >>> "Demo error: specific context here" in msg
    True
    >>> "first cause" in msg
    True
    """
    code: str
    name: str
    likely_causes: Tuple[str, ...] = ()
    look_at: Tuple[str, ...] = ()
    docs_url: Optional[str] = None

    def format_diagnostic(self, message: str) -> str:
        parts = [f"[{self.code}] {self.name}: {message}".rstrip()]
        if self.likely_causes:
            parts.append("  Likely causes:")
            for cause in self.likely_causes:
                parts.append(f"    - {cause}")
        if self.look_at:
            parts.append(f"  Look at: {', '.join(self.look_at)}")
        if self.docs_url:
            parts.append(f"  Spec: {self.docs_url}")
        return "\n".join(parts)


# ============================================================
# Base class — formats every subclass's message via its ErrorCode
# ============================================================


class LmlError(ValueError):
    """Base class for every LamQuant codec error.

    Inherits from ValueError so existing `except ValueError` callers keep
    working. New code should catch `LmlError` (or a specific subclass).

    Subclasses override `code: ClassVar[ErrorCode]` to attribute a unique
    diagnostic. `__init__` automatically formats the full message.

    >>> err = LmlError("plain message")
    >>> isinstance(err, ValueError)
    True
    >>> str(err)
    'plain message'
    """

    code: ClassVar[Optional[ErrorCode]] = None

    def __init__(self, message: str = ""):
        if self.__class__.code is not None:
            super().__init__(self.__class__.code.format_diagnostic(message))
        else:
            super().__init__(message)


# ============================================================
# Codec runtime — wire-format / packet-level errors
# ============================================================


class LmlMagicError(LmlError):
    code: ClassVar[ErrorCode] = ErrorCode(
        code="LML-MAGIC-001",
        name="Invalid magic byte",
        likely_causes=(
            "File is not an LML1 packet (truncated upload? wrong format?).",
            "Header offset miscomputed by an upstream container reader.",
            "Bytes were re-encoded as text and lost the binary magic.",
        ),
        look_at=(
            "lamquant_codec/lossless.py::_decompress_bytes_ref",
            "lamquant_codec/ops/fused_lml.py::fused_decompress",
            "lamquant-core/src/lml.rs::decompress",
        ),
        docs_url="docs/lml-format-v1.md#3.1-Magic",
    )


class LmlLegacyMagicError(LmlMagicError):
    code: ClassVar[ErrorCode] = ErrorCode(
        code="LML-MAGIC-002",
        name="Legacy iteration magic",
        likely_causes=(
            "File predates the LML1 cutover and uses LMQ4 / LMQ5 / LML  magic.",
            "Production decoders accept LML1 only; legacy decoder is opt-in.",
        ),
        look_at=(
            "lamquant_codec/legacy/lossless_legacy.py::_decompress_legacy_bytes_ref",
        ),
        docs_url="docs/lml-format-v1.md#legacy",
    )


class LmlVersionError(LmlError):
    code: ClassVar[ErrorCode] = ErrorCode(
        code="LML-VERSION-001",
        name="Unsupported wire-format version",
        likely_causes=(
            "File written by a newer LamQuant release; this reader is older.",
            "Forwards-incompatible spec change without a reader update.",
        ),
        look_at=(
            "lamquant_codec/lossless.py",
            "lamquant-core/src/lml.rs::decompress",
        ),
    )


class LmlTruncatedError(LmlError):
    code: ClassVar[ErrorCode] = ErrorCode(
        code="LML-TRUNC-001",
        name="Input truncated",
        likely_causes=(
            "File transfer / write was interrupted before completion.",
            "Container reader passed a slice that ends before the payload.",
            "Storage layer dropped trailing bytes silently.",
        ),
        look_at=(
            "lamquant_codec/lossless.py::_decompress_bytes_ref",
            "lamquant-core/src/container.rs::read_file",
        ),
    )


class LmlHeaderError(LmlError):
    code: ClassVar[ErrorCode] = ErrorCode(
        code="LML-HEADER-001",
        name="Invalid header field",
        likely_causes=(
            "Header field carries a value outside the spec's allowed range.",
            "u16/u32 silent overflow at the writer side.",
            "Wrong endianness or struct format string.",
        ),
        look_at=(
            "lamquant_codec/lossless.py::_compress_bytes_ref",
            "lamquant-core/src/lml.rs::compress",
        ),
    )


class LmlReservedBitsSetError(LmlHeaderError):
    code: ClassVar[ErrorCode] = ErrorCode(
        code="LML-HEADER-002",
        name="Reserved flag bits set",
        likely_causes=(
            "An older encoder used bit 0 as KLT flag — that semantic is now "
            "reserved per LML1 spec §3.2.",
            "Future format extension being read by an older reader.",
            "Header byte corrupted in transit.",
        ),
        look_at=(
            "lamquant_codec/lossless.py::_compress_bytes_ref",
            "lamquant-core/src/lml.rs::decompress",
        ),
        docs_url="docs/lml-format-v1.md#3.2-Flags",
    )


class LmlChannelCountError(LmlHeaderError):
    code: ClassVar[ErrorCode] = ErrorCode(
        code="LML-HEADER-003",
        name="Channel count out of range",
        likely_causes=(
            "Caller passed a transposed signal (T, C) instead of (C, T).",
            "n_ch > 1024 — not supported by the current LML1 spec.",
            "Empty signal (n_ch == 0).",
        ),
        look_at=(
            "lamquant_codec/lossless.py::_compress_bytes_ref",
        ),
    )


class LmlPayloadError(LmlError):
    code: ClassVar[ErrorCode] = ErrorCode(
        code="LML-PAYLOAD-001",
        name="Payload integrity failure",
        likely_causes=(
            "Bit flip or byte drop during transfer / storage.",
            "Mismatched encoder / decoder revisions producing different bytes.",
        ),
        look_at=(
            "lamquant_codec/lossless.py::_decompress_bytes_ref",
            "lamquant-core/src/lml.rs::decompress",
        ),
    )


class LmlCrcError(LmlPayloadError):
    code: ClassVar[ErrorCode] = ErrorCode(
        code="LML-CRC-001",
        name="CRC-32 mismatch",
        likely_causes=(
            "Storage corruption between write and read.",
            "Encoder bug producing the wrong CRC over the right payload.",
            "Decoder reading the wrong bytes for the CRC field "
            "(endianness or offset drift).",
        ),
        look_at=(
            "lamquant-core/src/crc32.rs",
            "lamquant_codec/lossless.py (zlib.crc32 call)",
        ),
        docs_url="docs/lml-format-v1.md#3.3-CRC",
    )


class LmlNoiseStrippedError(LmlPayloadError):
    code: ClassVar[ErrorCode] = ErrorCode(
        code="LML-NOISE-001",
        name="Refused to double-strip noise bits",
        likely_causes=(
            "Signal was already noise-stripped upstream and called again "
            "with noise_bits > 0.",
            "Caller didn't pass through the encoder's metadata that records "
            "previously applied noise stripping.",
        ),
        look_at=(
            "lamquant_codec/lossless.py::_compress_bytes_ref",
            "lamquant_codec/edf_to_lml.py (provenance metadata)",
        ),
    )


class LmlInputError(LmlError):
    code: ClassVar[ErrorCode] = ErrorCode(
        code="LML-INPUT-001",
        name="Caller-supplied input is malformed",
        likely_causes=(
            "Wrong array shape or dtype reaching the encoder.",
            "Empty signal supplied.",
        ),
        look_at=(
            "lamquant_codec/lossless.py",
            "lamquant_codec/codec_types.py",
        ),
    )


class LmlSignalShapeError(LmlInputError):
    code: ClassVar[ErrorCode] = ErrorCode(
        code="LML-INPUT-002",
        name="Signal shape is not [C, T]",
        likely_causes=(
            "Caller passed a 1-D signal without channel axis.",
            "Caller passed a batched [B, C, T] tensor without flattening.",
            "Signal axes were transposed.",
        ),
        look_at=(
            "lamquant_codec/lossless.py::_compress_bytes_ref",
            "lamquant_codec/decompose.py::decompose",
        ),
    )


class LmlEmptySignalError(LmlInputError):
    code: ClassVar[ErrorCode] = ErrorCode(
        code="LML-INPUT-003",
        name="Empty signal",
        likely_causes=(
            "Upstream EDF window had zero samples (decimation collapsed it).",
            "Channel selector returned an empty channel list.",
        ),
        look_at=(
            "lamquant_codec/edf_to_lml.py",
            "lamquant_codec/lossless.py::_compress_bytes_ref",
        ),
    )


# ============================================================
# LMA archive errors (separate domain)
# ============================================================


class LmaError(LmlError):
    """Base class for LMA archive-format errors."""
    code: ClassVar[Optional[ErrorCode]] = None


class LmaMagicError(LmaError):
    code: ClassVar[ErrorCode] = ErrorCode(
        code="LMA-MAGIC-001",
        name="Invalid LMA magic",
        likely_causes=(
            "File is not an LMA archive (it's LML1, raw EDF, or junk).",
            "Truncated archive lost the leading 4 bytes.",
        ),
        look_at=(
            "lamquant_codec/lma.py::unpack_lma",
            "lamquant-core/src/lma.rs::unpack_archive",
        ),
    )


class LmaManifestError(LmaError):
    code: ClassVar[ErrorCode] = ErrorCode(
        code="LMA-MANIFEST-001",
        name="Manifest is unreadable or malformed",
        likely_causes=(
            "Manifest length prefix exceeds file size or 256 MB cap.",
            "Manifest JSON failed to parse — schema drift?",
            "Method string in manifest is not in {lml, secondary, zstd, store}.",
        ),
        look_at=(
            "lamquant-core/src/lma.rs::Method::from_str",
            "lamquant_codec/lma.py::_read_lma_manifest",
        ),
    )


class LmaShaMismatchError(LmaError):
    code: ClassVar[ErrorCode] = ErrorCode(
        code="LMA-SHA-001",
        name="SHA-256 chain mismatch",
        likely_causes=(
            "Storage corruption inside the archive.",
            "Manifest hash field was edited without re-hashing.",
            "Per-entry hash recomputed from the wrong byte range.",
        ),
        look_at=(
            "lamquant-core/src/lma.rs::unpack_archive (verify=true path)",
            "lamquant_codec/lma.py::verify_lma",
        ),
    )


class LmaUnknownMethodError(LmaError):
    code: ClassVar[ErrorCode] = ErrorCode(
        code="LMA-METHOD-001",
        name="Unknown compression method in manifest",
        likely_causes=(
            "Archive was written by a newer LamQuant release with a method "
            "that this reader doesn't recognise.",
            "Manifest was hand-edited and contains a typo.",
        ),
        look_at=(
            "lamquant-core/src/lma.rs::Method::from_str",
            "lamquant_codec/lma.py (compressor registry)",
        ),
    )


# ============================================================
# Adaptive FSQ — SNN-driven per-timestep level scheduling
# ============================================================


class AdaptiveFSQError(LmlError):
    """Raised when adaptive FSQ is requested but the SNN cannot satisfy it.

    Refuses the silent fallback to uniform LMQ1 — production callers must
    opt out explicitly via `Encoder(adaptive=False)` or the CLI flag
    `--no-adaptive-fsq`. Failing loud beats writing an LMQ1 file when the
    user asked for LMQ3 (PCCP audit-trail correctness, Bible Rule 27 +
    Rule 30: graceful failure + hostile-caller interface)."""

    code: ClassVar[ErrorCode] = ErrorCode(
        code="LMQ-ADAPTIVE-001",
        name="Adaptive FSQ requested but SNN unavailable",
        likely_causes=(
            "registry.yaml.models.snn.production_sha256 is still a "
            "PLACEHOLDER_* — capture via `pccp_gate.py --capture --model snn`.",
            "--snn-checkpoint flag points at a missing or unreadable file.",
            "Loaded checkpoint has no `classify_per_timestep` method "
            "(architecture mismatch — wrong checkpoint family).",
            "Production wants adaptive (default ON) but no SNN was attached.",
        ),
        look_at=(
            "lamquant_codec/models/snn.py::resolve_production_snn",
            "lamquant_codec/models/snn.py::load_mamba_snn",
            "lamquant_codec/fileformat.py::Encoder._ensure_codec",
            "pccp/registry.yaml",
        ),
        docs_url="pccp/01-modifications.md#class-c-3-adaptive-snac-cr-range",
    )


# ============================================================
# Test-attributable failures (used by tests/helpers/asserts.py)
# ============================================================


class LmlTestError(LmlError):
    """Base class for LamQuant test-helper assertion failures.

    Distinct from production errors — these signal that a test invariant
    was violated, not that the codec itself misbehaved.
    """
    code: ClassVar[Optional[ErrorCode]] = None


class LmlCrossLangDriftError(LmlTestError):
    code: ClassVar[ErrorCode] = ErrorCode(
        code="TEST-DRIFT-001",
        name="Python ↔ Rust output divergence",
        likely_causes=(
            "Endianness drift in a header field.",
            "Different field widths (u16 vs u32) on either side.",
            "Fused (numba) path and reference path diverge.",
            "rayon thread count or BLAS version changed binary output.",
        ),
        look_at=(
            "tests/codec/test_l5_cross_lang.py",
            "lamquant_codec/lossless.py vs lamquant-core/src/lml.rs",
            "tests/helpers/rust_bindings.py",
        ),
    )


class LmlDeterminismError(LmlTestError):
    code: ClassVar[ErrorCode] = ErrorCode(
        code="TEST-DETERMINISM-001",
        name="Same input yielded different output across runs",
        likely_causes=(
            "RNG seed not pinned (leaked global np.random state).",
            "PYTHONHASHSEED randomisation affecting dict iteration order.",
            "BLAS / numba caching invalidated between calls.",
            "Floating-point reduction order changed (e.g. parallel sum).",
        ),
        look_at=(
            "tests/helpers/signals.py::synth_signal",
            "tests/conftest.py (any fixture using np.random.seed globally)",
        ),
    )


class LmlDtypeDriftError(LmlTestError):
    code: ClassVar[ErrorCode] = ErrorCode(
        code="TEST-DTYPE-001",
        name="Output dtype differs from contract",
        likely_causes=(
            "Encoder cast to float64 where float32 was expected.",
            "Decoder returned int32 where the contract specified int64.",
            "numpy.savez silently widened dtype on save.",
        ),
        look_at=(
            "lamquant_codec/lossless.py (output dtype contract)",
            "lamquant_codec/decompose.py",
        ),
    )


class LmlResourceLeakError(LmlTestError):
    code: ClassVar[ErrorCode] = ErrorCode(
        code="TEST-RESOURCE-001",
        name="Test left a temp file / FD / process behind",
        likely_causes=(
            "Encoder didn't close a file handle on the error path.",
            "Subprocess killed without reaping stdout pipe.",
            "tempfile.NamedTempFile dropped without delete=False handling.",
        ),
        look_at=(
            "tests/helpers/edf_factory.py",
            "lamquant-core/src/lma.rs (CleanupGuard pattern)",
        ),
    )


class LmlTestFixtureMissingError(LmlTestError):
    code: ClassVar[ErrorCode] = ErrorCode(
        code="TEST-FIXTURE-001",
        name="Required test fixture not present",
        likely_causes=(
            "Real-EEG dataset (q31_events) not on this machine.",
            "Trained checkpoint missing under weights/.",
            "PyO3 wheel not built — run `maturin develop --features python`.",
        ),
        look_at=(
            "tests/helpers/data_paths.py",
            "tests/conftest.py (skip-on-missing fixtures)",
        ),
    )
