"""Deep coverage tests for ``lamquant_codec.integrity``.

Complements ``test_integrity_functional.py``. Targets:
  - registry_path: LAMQUANT_REGISTRY_PATH override + repo walk
  - registry_sha: read by-model from a tmp_path registry YAML
  - registry_sha: KeyError for unknown model, IntegrityError on placeholder
  - verify_checkpoint: success path, mismatch in strict + permissive
  - _load_registry_yaml: PyYAML required, IntegrityError on missing yaml

tmp_path fixtures only — never write outside tmp_path.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from lamquant_codec.integrity import (
    IntegrityError,
    _load_registry_yaml,
    registry_path,
    registry_sha,
    sha256_of_file,
    verify_checkpoint,
)


REAL_SHA = (
    "deadbeef" * 8  # 64 hex chars
)


def _write_registry(tmp_path: Path, body: str) -> Path:
    """Write a YAML registry at tmp_path/pccp/registry.yaml. Returns Path."""
    pccp = tmp_path / "pccp"
    pccp.mkdir(parents=True, exist_ok=True)
    p = pccp / "registry.yaml"
    p.write_text(body)
    return p


# ---------------------------------------------------------------------------
# registry_path
# ---------------------------------------------------------------------------


class TestRegistryPath:
    def test_env_var_override(self, tmp_path: Path) -> None:
        # Don't require the path to exist — registry_path() is a resolution helper.
        target = tmp_path / "custom.yaml"
        with patch.dict(os.environ, {"LAMQUANT_REGISTRY_PATH": str(target)}):
            assert registry_path() == target

    def test_default_finds_repo_registry(self) -> None:
        """Without override, registry_path walks up looking for pccp/registry.yaml.
        The real repo has one at the canonical location."""
        env = {k: v for k, v in os.environ.items()
               if k != "LAMQUANT_REGISTRY_PATH"}
        with patch.dict(os.environ, env, clear=True):
            p = registry_path()
            assert p.exists()
            assert p.name == "registry.yaml"
            assert p.parent.name == "pccp"


# ---------------------------------------------------------------------------
# _load_registry_yaml
# ---------------------------------------------------------------------------


class TestLoadRegistryYaml:
    def test_loads_valid_yaml(self, tmp_path: Path) -> None:
        reg = _write_registry(tmp_path, """
models:
  encoder:
    production_sha256: 0123456789abcdef
""")
        data = _load_registry_yaml(reg)
        assert isinstance(data, dict)
        assert "models" in data
        assert data["models"]["encoder"]["production_sha256"] == "0123456789abcdef"

    def test_empty_file_returns_none(self, tmp_path: Path) -> None:
        """yaml.safe_load on empty input returns None."""
        reg = _write_registry(tmp_path, "")
        assert _load_registry_yaml(reg) is None


# ---------------------------------------------------------------------------
# registry_sha
# ---------------------------------------------------------------------------


class TestRegistrySha:
    def test_reads_specific_model(self, tmp_path: Path) -> None:
        reg = _write_registry(tmp_path, f"""
models:
  encoder:
    production_sha256: {REAL_SHA}
  snn:
    production_sha256: aabbccdd00112233aabbccdd00112233aabbccdd00112233aabbccdd00112233
""")
        with patch.dict(os.environ, {"LAMQUANT_REGISTRY_PATH": str(reg)}):
            assert registry_sha("encoder") == REAL_SHA
            assert registry_sha("snn").startswith("aabbccdd")

    def test_unknown_model_raises_keyerror(self, tmp_path: Path) -> None:
        reg = _write_registry(tmp_path, """
models:
  encoder:
    production_sha256: 0123
""")
        with patch.dict(os.environ, {"LAMQUANT_REGISTRY_PATH": str(reg)}):
            with pytest.raises(KeyError, match="no_such_model"):
                registry_sha("no_such_model")

    def test_missing_sha_raises_integrity_error(self, tmp_path: Path) -> None:
        reg = _write_registry(tmp_path, """
models:
  encoder: {}
""")
        with patch.dict(os.environ, {"LAMQUANT_REGISTRY_PATH": str(reg)}):
            with pytest.raises(IntegrityError, match="no production_sha256"):
                registry_sha("encoder")

    def test_placeholder_sha_raises_integrity_error(self, tmp_path: Path) -> None:
        reg = _write_registry(tmp_path, """
models:
  encoder:
    production_sha256: PLACEHOLDER_encoder_sha
""")
        with patch.dict(os.environ, {"LAMQUANT_REGISTRY_PATH": str(reg)}):
            with pytest.raises(IntegrityError, match="placeholder"):
                registry_sha("encoder")


# ---------------------------------------------------------------------------
# sha256_of_file
# ---------------------------------------------------------------------------


class TestSha256OfFile:
    def test_chunk_size_param(self, tmp_path: Path) -> None:
        # Very small chunks force many loop iterations
        p = tmp_path / "x.bin"
        p.write_bytes(b"abc" * 1000)
        h1 = sha256_of_file(p, chunk_bytes=7)
        h2 = sha256_of_file(p, chunk_bytes=1 << 20)
        assert h1 == h2

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        p = tmp_path / "x.bin"
        p.write_bytes(b"hello")
        h_str = sha256_of_file(str(p))
        h_path = sha256_of_file(p)
        assert h_str == h_path


# ---------------------------------------------------------------------------
# verify_checkpoint
# ---------------------------------------------------------------------------


class TestVerifyCheckpoint:
    def test_success_path(self, tmp_path: Path) -> None:
        """SHA matches the registry pin -> returns actual SHA, no raise."""
        # Compute the canonical SHA of a small payload, pin it in registry.
        ckpt = tmp_path / "model.ckpt"
        payload = b"a verified checkpoint payload"
        ckpt.write_bytes(payload)
        import hashlib
        expected = hashlib.sha256(payload).hexdigest()
        reg = _write_registry(tmp_path, f"""
models:
  encoder:
    production_sha256: {expected}
""")
        with patch.dict(os.environ, {"LAMQUANT_REGISTRY_PATH": str(reg)}):
            result = verify_checkpoint("encoder", ckpt)
        assert result == expected

    def test_mismatch_strict_raises(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "bad.ckpt"
        ckpt.write_bytes(b"wrong content")
        # Pin some unrelated SHA
        reg = _write_registry(tmp_path, f"""
models:
  encoder:
    production_sha256: {REAL_SHA}
""")
        with patch.dict(os.environ, {"LAMQUANT_REGISTRY_PATH": str(reg)}):
            with pytest.raises(IntegrityError, match="FAILED"):
                verify_checkpoint("encoder", ckpt)

    def test_mismatch_permissive_warns_but_returns(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        ckpt = tmp_path / "drift.ckpt"
        ckpt.write_bytes(b"drift content")
        reg = _write_registry(tmp_path, f"""
models:
  encoder:
    production_sha256: {REAL_SHA}
""")
        with patch.dict(os.environ, {"LAMQUANT_REGISTRY_PATH": str(reg)}):
            actual = verify_checkpoint("encoder", ckpt, strict=False)
        # Returns the actual hex SHA, not the expected one.
        assert isinstance(actual, str)
        assert len(actual) == 64
        # WARNING is on stderr in permissive mode
        out = capsys.readouterr()
        assert "WARNING" in out.err

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        """verify_checkpoint accepts str paths."""
        ckpt = tmp_path / "x.ckpt"
        ckpt.write_bytes(b"payload")
        import hashlib
        expected = hashlib.sha256(b"payload").hexdigest()
        reg = _write_registry(tmp_path, f"""
models:
  encoder:
    production_sha256: {expected}
""")
        with patch.dict(os.environ, {"LAMQUANT_REGISTRY_PATH": str(reg)}):
            assert verify_checkpoint("encoder", str(ckpt)) == expected


class TestModuleSurface:
    def test_all_public_exports_resolvable(self) -> None:
        from lamquant_codec import integrity as m
        for name in m.__all__:
            assert hasattr(m, name)
