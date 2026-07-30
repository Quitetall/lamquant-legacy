"""Functional tests for ``lamquant_codec.integrity``.

Pins contract:
  - sha256_of_file returns the correct 64-char hex digest for known bytes
    (golden via Python's hashlib — stable across runtimes)
  - verify_checkpoint raises FileNotFoundError when the path is missing
  - IntegrityError subclasses RuntimeError
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from lamquant_codec.integrity import (
    IntegrityError,
    sha256_of_file,
    verify_checkpoint,
)


def test_sha256_of_known_bytes(tmp_path: Path) -> None:
    """SHA-256 is deterministic + matches hashlib.sha256."""
    payload = b"LamQuant integrity test fixture"
    p = tmp_path / "blob.bin"
    p.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert sha256_of_file(p) == expected


def test_sha256_chunked_matches_oneshot(tmp_path: Path) -> None:
    """Chunked read must match a single-shot hash even on chunk boundaries."""
    payload = b"X" * (3 * (1 << 20) + 17)  # 3 MB + a tail
    p = tmp_path / "big.bin"
    p.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    # Force tiny chunks to exercise the loop
    assert sha256_of_file(p, chunk_bytes=1024) == expected
    assert sha256_of_file(p) == expected


def test_sha256_empty_file(tmp_path: Path) -> None:
    """SHA-256 of an empty file is the well-known empty digest."""
    p = tmp_path / "empty.bin"
    p.write_bytes(b"")
    EMPTY_SHA256 = (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    assert sha256_of_file(p) == EMPTY_SHA256


def test_verify_checkpoint_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        verify_checkpoint("anything", tmp_path / "does_not_exist.ckpt")


def test_integrity_error_is_runtime_error() -> None:
    """IntegrityError must be a RuntimeError subclass so existing
    `except RuntimeError:` handlers catch it."""
    assert issubclass(IntegrityError, RuntimeError)
