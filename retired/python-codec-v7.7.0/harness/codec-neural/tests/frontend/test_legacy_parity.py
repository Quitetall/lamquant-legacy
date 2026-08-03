"""Exact parity against frozen retired Python codec frontend."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_META_ROOT = Path(__file__).resolve().parents[3]
_LEGACY_SOURCE = (
    _META_ROOT
    / "legacy"
    / "retired"
    / "python-codec-v7.7.0"
    / "source"
)
if not _LEGACY_SOURCE.is_dir():
    pytest.skip(
        "frozen legacy codec snapshot unavailable outside LamQuant meta-repo",
        allow_module_level=True,
    )

sys.path.insert(0, str(_LEGACY_SOURCE))

from lamquant_codec.ops.pipeline import (  # noqa: E402
    preprocess_subband as legacy_preprocess_float,
)
from lamquant_codec.ops.pipeline import (  # noqa: E402
    preprocess_subband_single as legacy_preprocess,
)
from lamquant_codec.ops.pipeline import (  # noqa: E402
    reconstruct_from_subband as legacy_reconstruct,
)
from lamquant_codec.ops.wht import forward_32 as legacy_wht  # noqa: E402
from lamquant_codec.ops.wht import (  # noqa: E402
    forward_32_torch as legacy_wht_torch,
)
from lamquant_neural.frontend import (  # noqa: E402
    forward_32,
    forward_32_torch,
    preprocess_subband,
    preprocess_subband_single,
    reconstruct_from_subband,
)


def _signal() -> np.ndarray:
    rng = np.random.default_rng(139)
    return rng.normal(0.0, 100.0, size=(3, 2500)).astype(np.float32)


def _assert_metadata_equal(legacy, current) -> None:
    np.testing.assert_array_equal(current[0], legacy[0])
    np.testing.assert_array_equal(current[1], legacy[1])
    assert len(current[2]) == len(legacy[2])
    for current_channel, legacy_channel in zip(current[2], legacy[2], strict=True):
        assert current_channel.keys() == legacy_channel.keys()
        for name in current_channel:
            np.testing.assert_array_equal(
                current_channel[name],
                legacy_channel[name],
            )


def test_integer_preprocessing_and_reconstruction_exact_parity() -> None:
    signal = _signal()
    legacy = legacy_preprocess(signal)
    current = preprocess_subband_single(signal)
    _assert_metadata_equal(legacy, current)

    legacy_signal = legacy_reconstruct(*legacy)
    current_signal = reconstruct_from_subband(*current)
    np.testing.assert_array_equal(current_signal, legacy_signal)


def test_float_preprocessing_exact_parity() -> None:
    signal = _signal().astype(np.float64)
    _assert_metadata_equal(
        legacy_preprocess_float(signal),
        preprocess_subband(signal),
    )


def test_wht_exact_parity() -> None:
    rng = np.random.default_rng(141)
    vector = rng.normal(size=32).astype(np.float32)
    np.testing.assert_array_equal(forward_32(vector), legacy_wht(vector))

    latent = torch.from_numpy(
        rng.normal(size=(2, 32, 79)).astype(np.float32)
    )
    torch.testing.assert_close(
        forward_32_torch(latent),
        legacy_wht_torch(latent),
        rtol=0,
        atol=0,
    )
