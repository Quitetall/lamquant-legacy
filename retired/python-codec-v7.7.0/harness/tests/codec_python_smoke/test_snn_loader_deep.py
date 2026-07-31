"""Deep coverage tests for ``lamquant_codec.models.snn``.

This is the production SNN LOADER/REGISTRY (not the model definition).
Targets:
  - resolve_production_snn: registry parse + path containment rules
  - _sniff_architecture: shape inference + sparse-key rejection
  - load_mamba_snn: type-validation at entry, missing-file failure mode,
    O_NOFOLLOW path

Tests use unittest.mock to replace integrity helpers (registry_path,
_load_registry_yaml, registry_sha) and to substitute torch.load behaviour.
We do NOT mock torch.device or any value that participates in isinstance
checks at module boundary.

No real EDF data needed — pure logic / mocking.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from lamquant_codec.errors import AdaptiveFSQError
from lamquant_codec.models.snn import (
    _DEFAULT_D_MODEL,
    _DEFAULT_D_STATE,
    _DEFAULT_IN_CHANNELS,
    _DEFAULT_N_LAYERS,
    _MAX_CKPT_BYTES,
    _sniff_architecture,
    load_mamba_snn,
    resolve_production_snn,
)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_max_ckpt_bytes_is_positive(self) -> None:
        assert _MAX_CKPT_BYTES > 0
        assert _MAX_CKPT_BYTES == 32 * 1024 * 1024

    def test_defaults_are_positive(self) -> None:
        for v in (_DEFAULT_D_MODEL, _DEFAULT_D_STATE,
                  _DEFAULT_N_LAYERS, _DEFAULT_IN_CHANNELS):
            assert v > 0


# ---------------------------------------------------------------------------
# resolve_production_snn — registry-driven path resolution
# ---------------------------------------------------------------------------


def _write_registry(tmp_path: Path, registry_yaml: str) -> Path:
    """Build a fake repo layout: <tmp>/pccp/registry.yaml. Returns its Path."""
    pccp = tmp_path / "pccp"
    pccp.mkdir(parents=True, exist_ok=True)
    p = pccp / "registry.yaml"
    p.write_text(registry_yaml)
    return p


class TestResolveProductionSnn:
    def test_placeholder_sha_returns_none(self, tmp_path: Path) -> None:
        """PLACEHOLDER_* SHA -> None (uncaptured pin)."""
        reg_path = _write_registry(tmp_path, """
models:
  snn:
    production_checkpoint: weights/snn/foo.pt
    production_sha256: PLACEHOLDER_snn_sha
""")
        # Production weights file doesn't need to exist for placeholder test
        with patch.dict("os.environ",
                         {"LAMQUANT_REGISTRY_PATH": str(reg_path)}):
            assert resolve_production_snn() is None

    def test_missing_pin_returns_none(self, tmp_path: Path) -> None:
        reg_path = _write_registry(tmp_path, """
models:
  snn: {}
""")
        with patch.dict("os.environ",
                         {"LAMQUANT_REGISTRY_PATH": str(reg_path)}):
            assert resolve_production_snn() is None

    def test_no_snn_block_returns_none(self, tmp_path: Path) -> None:
        reg_path = _write_registry(tmp_path, """
models:
  encoder: {}
""")
        with patch.dict("os.environ",
                         {"LAMQUANT_REGISTRY_PATH": str(reg_path)}):
            assert resolve_production_snn() is None

    def test_no_models_block_returns_none(self, tmp_path: Path) -> None:
        reg_path = _write_registry(tmp_path, """
canonical: {}
""")
        with patch.dict("os.environ",
                         {"LAMQUANT_REGISTRY_PATH": str(reg_path)}):
            assert resolve_production_snn() is None

    def test_absolute_path_rejected(self, tmp_path: Path) -> None:
        """Registry pin MUST be repo-relative — absolute is rejected."""
        reg_path = _write_registry(tmp_path, """
models:
  snn:
    production_checkpoint: /etc/passwd
    production_sha256: aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899
""")
        with patch.dict("os.environ",
                         {"LAMQUANT_REGISTRY_PATH": str(reg_path)}):
            assert resolve_production_snn() is None

    def test_path_escape_rejected(self, tmp_path: Path) -> None:
        """Path traversal (..) that escapes the repo subtree -> None."""
        reg_path = _write_registry(tmp_path, """
models:
  snn:
    production_checkpoint: ../../../etc/passwd
    production_sha256: aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899
""")
        with patch.dict("os.environ",
                         {"LAMQUANT_REGISTRY_PATH": str(reg_path)}):
            assert resolve_production_snn() is None

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        """Registry pin pointing at non-existent path -> None."""
        reg_path = _write_registry(tmp_path, """
models:
  snn:
    production_checkpoint: weights/snn/missing.pt
    production_sha256: aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899
""")
        with patch.dict("os.environ",
                         {"LAMQUANT_REGISTRY_PATH": str(reg_path)}):
            assert resolve_production_snn() is None

    def test_valid_pin_returns_resolved_path(self, tmp_path: Path) -> None:
        """Valid registry pin + existing file -> returns Path."""
        reg_path = _write_registry(tmp_path, """
models:
  snn:
    production_checkpoint: weights/snn/ok.pt
    production_sha256: aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899
""")
        # Create the pinned file inside the fake repo root.
        ckpt_dir = tmp_path / "weights" / "snn"
        ckpt_dir.mkdir(parents=True)
        ckpt = ckpt_dir / "ok.pt"
        ckpt.write_bytes(b"fake-ckpt-bytes")
        with patch.dict("os.environ",
                         {"LAMQUANT_REGISTRY_PATH": str(reg_path)}):
            result = resolve_production_snn()
        assert result is not None
        assert result.name == "ok.pt"

    def test_registry_unreachable_returns_none(self, tmp_path: Path) -> None:
        """OSError/FileNotFoundError on registry read -> None."""
        bogus = tmp_path / "does_not_exist.yaml"
        with patch.dict("os.environ",
                         {"LAMQUANT_REGISTRY_PATH": str(bogus)}):
            assert resolve_production_snn() is None

    def test_non_str_sha_returns_none(self, tmp_path: Path) -> None:
        """Non-string SHA (e.g. accidentally an int) -> None."""
        reg_path = _write_registry(tmp_path, """
models:
  snn:
    production_checkpoint: weights/snn/x.pt
    production_sha256: 12345
""")
        with patch.dict("os.environ",
                         {"LAMQUANT_REGISTRY_PATH": str(reg_path)}):
            assert resolve_production_snn() is None


# ---------------------------------------------------------------------------
# _sniff_architecture — state_dict shape inference
# ---------------------------------------------------------------------------


class _FakeTensor:
    """Stand-in for a torch.Tensor — only the .shape attribute is exercised."""
    def __init__(self, shape: tuple) -> None:
        self.shape = shape


class TestSniffArchitecture:
    def test_non_dict_input_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="state_dict must be dict"):
            _sniff_architecture("not a dict")  # type: ignore

    def test_empty_dict_raises(self) -> None:
        with pytest.raises(AdaptiveFSQError, match="empty state_dict"):
            _sniff_architecture({})

    def test_full_inference(self) -> None:
        """spatial_mix.weight + ssm_blocks.{0,1}.fwd.A_log -> all dims inferred."""
        sd = {
            "spatial_mix.weight": _FakeTensor((48, 21)),  # d_model=48, in_ch=21
            "ssm_blocks.0.fwd.A_log": _FakeTensor((48, 32)),  # d_state=32
            "ssm_blocks.1.fwd.A_log": _FakeTensor((48, 32)),
            "extra_param": _FakeTensor((1,)),
        }
        in_ch, d_model, d_state, n_layers = _sniff_architecture(sd)
        assert in_ch == 21
        assert d_model == 48
        assert d_state == 32
        assert n_layers == 2

    def test_missing_spatial_mix_falls_back_to_defaults(self) -> None:
        sd = {
            "ssm_blocks.0.fwd.A_log": _FakeTensor((40, 16)),
        }
        in_ch, d_model, d_state, n_layers = _sniff_architecture(sd)
        # in_ch + d_model default; d_state from A_log
        assert in_ch == _DEFAULT_IN_CHANNELS
        assert d_model == _DEFAULT_D_MODEL
        assert d_state == 16
        assert n_layers == 1

    def test_no_ssm_blocks_uses_default_n_layers(self) -> None:
        sd = {
            "spatial_mix.weight": _FakeTensor((40, 21)),
        }
        _, _, _, n_layers = _sniff_architecture(sd)
        assert n_layers == _DEFAULT_N_LAYERS

    def test_sparse_ssm_blocks_rejected(self) -> None:
        """Sparse layer indices {0, 2} -> AdaptiveFSQError."""
        sd = {
            "spatial_mix.weight": _FakeTensor((40, 21)),
            "ssm_blocks.0.fwd.A_log": _FakeTensor((40, 16)),
            "ssm_blocks.2.fwd.A_log": _FakeTensor((40, 16)),  # skips 1
        }
        with pytest.raises(AdaptiveFSQError, match="sparse"):
            _sniff_architecture(sd)

    def test_unexpected_spatial_mix_shape_raises(self) -> None:
        sd = {
            "spatial_mix.weight": _FakeTensor((40,)),  # 1-D, not 2-D
        }
        with pytest.raises(AdaptiveFSQError, match="unexpected shape"):
            _sniff_architecture(sd)

    def test_bwd_a_log_works(self) -> None:
        """bwd.A_log key (bidirectional SSM) also infers d_state."""
        sd = {
            "spatial_mix.weight": _FakeTensor((40, 21)),
            "ssm_blocks.0.bwd.A_log": _FakeTensor((40, 24)),
        }
        _, _, d_state, _ = _sniff_architecture(sd)
        assert d_state == 24


# ---------------------------------------------------------------------------
# load_mamba_snn — type validation + missing file
# ---------------------------------------------------------------------------


class TestLoadMambaSnnBoundary:
    def test_non_str_non_path_checkpoint_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="expected Path or str"):
            load_mamba_snn(12345)  # type: ignore

    def test_empty_device_raises(self, tmp_path: Path) -> None:
        # Path can be anything — type-check runs before existence-check.
        # Use an empty string for device to hit the boundary check.
        with pytest.raises(TypeError, match="non-empty str"):
            load_mamba_snn(tmp_path / "x.pt", device="")

    def test_non_str_device_raises(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="non-empty str"):
            load_mamba_snn(tmp_path / "x.pt", device=42)  # type: ignore

    def test_non_bool_allow_pickle_raises(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="expected bool"):
            load_mamba_snn(tmp_path / "x.pt",
                           allow_pickle_fallback="yes")  # type: ignore

    def test_missing_path_raises_adaptive_fsq_error(self, tmp_path: Path) -> None:
        bogus = tmp_path / "missing.pt"
        with pytest.raises(AdaptiveFSQError, match="checkpoint missing"):
            load_mamba_snn(bogus)

    def test_directory_path_rejected(self, tmp_path: Path) -> None:
        d = tmp_path / "a_directory"
        d.mkdir()
        with pytest.raises(AdaptiveFSQError, match="not a regular file"):
            load_mamba_snn(d)

    def test_oversized_file_rejected(self, tmp_path: Path) -> None:
        """File > _MAX_CKPT_BYTES is refused without invoking torch.load.

        Use os.posix_fallocate-style sparse write to create a logically
        huge file without writing 32 MB+ of real bytes.
        """
        big = tmp_path / "huge.pt"
        # Write a single trailing byte at position _MAX_CKPT_BYTES so
        # st_size > cap. The intermediate range is sparse on common FS.
        with big.open("wb") as f:
            f.seek(_MAX_CKPT_BYTES + 1)
            f.write(b"\x00")
        with pytest.raises(AdaptiveFSQError, match="refusing"):
            load_mamba_snn(big)

    def test_string_path_accepted_as_str(self, tmp_path: Path) -> None:
        """Pass a str path — internal isinstance check converts to Path."""
        with pytest.raises(AdaptiveFSQError, match="checkpoint missing"):
            load_mamba_snn(str(tmp_path / "missing.pt"))
