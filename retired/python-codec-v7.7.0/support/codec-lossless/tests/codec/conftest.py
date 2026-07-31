"""Codec-specific fixtures.

Lives next to the codec wire-format tests so the fixtures it owns are
visible only inside `tests/codec/**`. Cross-cutting fixtures stay in the
root `tests/conftest.py`.
"""
import numpy as np
import pytest


@pytest.fixture
def sample_eeg_q31():
    """Sample Q31-format EEG data [21 channels, 2500 samples]."""
    np.random.seed(42)
    eeg_q31 = np.random.randint(
        -2147483647, 2147483647, size=(21, 2500), dtype=np.int32
    )
    return eeg_q31


@pytest.fixture
def sample_eeg_float():
    """Sample normalized EEG [21, 2500] in microvolts."""
    np.random.seed(42)
    return np.random.randn(21, 2500).astype(np.float32) * 100
