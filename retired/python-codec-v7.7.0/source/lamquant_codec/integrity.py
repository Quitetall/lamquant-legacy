"""Runtime model-file integrity verification.

Per PCCP cybersecurity pillar (`pccp/06-cybersecurity.md` Section 2),
every checkpoint loaded at runtime must be verified against the SHA-256
pinned in `pccp/registry.yaml`. Mismatch indicates either:
  - tampering (substitution attack)
  - desync between deployed binary and deployed model files
  - PCCP-authorized update not yet promoted into the registry

Strict mode (default for production) raises IntegrityError on mismatch.
Permissive mode (developer use) logs a warning and proceeds.

Usage:
    from lamquant_codec.integrity import verify_checkpoint, IntegrityError

    try:
        verify_checkpoint("encoder", "weights/student_subband.ckpt")
    except IntegrityError as e:
        sys.exit(f"refusing to load tampered checkpoint: {e}")

    # Permissive (dev):
    verify_checkpoint("encoder", path, strict=False)

    # Verify registry without touching disk (e.g., after lazy-load):
    expected = registry_sha("encoder")
    actual = sha256_of_file(path)
    if actual != expected: ...

Design:
    - Zero PyTorch dependency. Reads bytes + computes SHA-256 only.
    - PyYAML is REQUIRED for registry parsing (V4 Pro Finding 1 on the
      bones-A+B+C commit: under FDA PCCP, registry interpretation must
      use a robust, well-tested parser, not a hand-rolled one). Install
      with: `pip install pyyaml`. Failure to import yields a clear
      IntegrityError, never a silent permissive parse.
    - Registry path is auto-discovered relative to the package, with
      LAMQUANT_REGISTRY_PATH env var override.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Optional

__all__ = [
    "IntegrityError",
    "registry_path",
    "registry_sha",
    "sha256_of_file",
    "verify_checkpoint",
]


class IntegrityError(RuntimeError):
    """Raised when a checkpoint's SHA-256 does not match the registry."""


# ---------------------------------------------------------------------------
# Registry discovery + parse
# ---------------------------------------------------------------------------


def registry_path() -> Path:
    """Locate `pccp/registry.yaml`.

    Resolution order:
      1. $LAMQUANT_REGISTRY_PATH (if set)
      2. <repo_root>/pccp/registry.yaml — repo_root inferred by walking
         up from this file looking for a `pccp` sibling.
    """
    override = os.environ.get("LAMQUANT_REGISTRY_PATH")
    if override:
        return Path(override)

    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        candidate = parent / "pccp" / "registry.yaml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate pccp/registry.yaml. Set LAMQUANT_REGISTRY_PATH."
    )


def _load_registry_yaml(path: Path) -> dict:
    """Parse the registry via PyYAML. PCCP-required: no fallback parser
    so registry interpretation matches the gate hook byte-for-byte.

    PyYAML is imported lazily here (not at module load) so downstream
    consumers can import `lamquant_codec.integrity` for type-only or
    documentation purposes without needing PyYAML installed. Actual
    verification calls hit this function and require PyYAML (V4 Pro
    Finding 5 of the bones-A+B+C-fixes commit)."""
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise IntegrityError(
            "PyYAML is required for PCCP integrity verification. Install with:\n"
            "    pip install pyyaml\n"
            "(See pccp/06-cybersecurity.md for the security rationale.)"
        ) from exc
    with path.open("r") as f:
        return yaml.safe_load(f)


def registry_sha(model: str) -> str:
    """Return the production_sha256 pinned in the registry for `model`."""
    reg = _load_registry_yaml(registry_path())
    spec = reg.get("models", {}).get(model)
    if spec is None:
        raise KeyError(f"No registry entry for model '{model}'")
    sha = spec.get("production_sha256")
    if not sha:
        raise IntegrityError(f"Registry has no production_sha256 for model '{model}'")
    if isinstance(sha, str) and sha.startswith("PLACEHOLDER"):
        raise IntegrityError(
            f"Model '{model}' has placeholder SHA — capture via "
            f"`pccp_gate.py --capture --model {model} --candidate <ckpt>`"
        )
    return sha


# ---------------------------------------------------------------------------
# File hashing + verification
# ---------------------------------------------------------------------------


def sha256_of_file(path: Path | str, chunk_bytes: int = 1 << 20) -> str:
    """Compute SHA-256 of a file. Streams in 1 MB chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checkpoint(model: str, path: Path | str, *, strict: bool = True) -> str:
    """Verify checkpoint SHA-256 matches the registry pin for `model`.

    Returns the actual SHA on success. On mismatch:
      strict=True (default): raises IntegrityError
      strict=False:          writes warning to stderr, returns the actual SHA

    Raises:
      FileNotFoundError if `path` does not exist.
      KeyError if `model` is not registered.
      IntegrityError on mismatch in strict mode, or on placeholder pin.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    expected = registry_sha(model)
    actual = sha256_of_file(p)

    if actual != expected:
        msg = (
            f"Integrity check FAILED for {model} at {p}\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            f"This may indicate model file tampering, a desynchronized\n"
            f"deployment, or an unpromoted PCCP-authorized update."
        )
        if strict:
            raise IntegrityError(msg)
        sys.stderr.write(f"[lamquant_codec.integrity] WARNING: {msg}\n")

    return actual
