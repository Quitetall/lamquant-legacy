"""Tests for the MNE-Python integration layer.

Verifies:
  - mne is NOT pulled by `import lamquant_codec` (lazy)
  - read/write Raw round-trips bit-exact through .lml
  - Channel info (names, sfreq, ch_types) survives the round-trip
  - V/µV unit conversion is consistent (write * 1e6 → read / 1e6 returns V)
  - Auto-dispatch by extension works
  - Helpful error message when mne is unavailable
  - .lmq read/write needs an explicit checkpoint
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

mne = pytest.importorskip('mne')   # skip whole module if mne missing


def _make_raw(n_channels: int = 21, n_samples: int = 2500,
              sfreq: float = 250.0, seed: int = 0):
    """Build a synthetic mne.io.RawArray of integer-valued µV data."""
    rng = np.random.default_rng(seed)
    # Integer µV values (matches LamQuant's ADC convention).
    data_uv = rng.integers(-2000, 2000, (n_channels, n_samples)).astype(np.float64)
    # MNE stores volts internally — multiply by 1e-6 to convert µV → V.
    data_v = data_uv * 1e-6
    ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data_v, info, verbose=False)
    return raw, data_uv


# ============================================================
# Lazy import — mne not pulled by `import lamquant_codec`
# ============================================================

def test_import_lamquant_codec_does_not_load_mne():
    """Plain `import lamquant_codec` must not trigger mne load."""
    import subprocess
    r = subprocess.run(
        [sys.executable, '-c',
         'import sys; import lamquant_codec; '
         'assert "mne" not in sys.modules, "mne loaded eagerly!"'],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f'stderr: {r.stderr}'


# ============================================================
# Lossless round-trip — .lml
# ============================================================

def test_write_lml_then_read_lml_bit_exact(tmp_path: Path):
    """Round-trip integer-valued EEG through .lml and recover bit-exact."""
    from lamquant_codec.mne_io import write_lml, read_raw_lml

    raw, expected_uv = _make_raw()
    out = tmp_path / 'rec.lml'
    written = write_lml(raw, out)
    assert written == out and out.exists()
    assert out.read_bytes()[:3] == b'LML'

    raw2 = read_raw_lml(out)
    # Channel count + sample count preserved.
    assert raw2.info['nchan'] == raw.info['nchan']
    assert raw2.n_times == raw.n_times
    assert raw2.info['sfreq'] == raw.info['sfreq']
    # Channel types are EEG.
    assert all(t == 'eeg' for t in raw2.get_channel_types())

    # Bit-exact in µV space. Use np.round (not astype-truncation) on both
    # sides — the V↔µV float roundtrip introduces ULP noise around integer
    # boundaries that astype(int64) would treat as off-by-one.
    recovered_uv = raw2.get_data() * 1e6
    assert np.array_equal(np.round(recovered_uv).astype(np.int64),
                          np.round(expected_uv).astype(np.int64))


def test_read_lml_assigns_default_10_20_for_21ch(tmp_path: Path):
    """When a file has 21 channels, default to the standard 10-20 montage."""
    from lamquant_codec.mne_io import (
        write_lml, read_raw_lml, DEFAULT_CH_NAMES_21,
    )
    raw, _ = _make_raw(n_channels=21)
    out = tmp_path / 'r.lml'
    write_lml(raw, out)
    raw2 = read_raw_lml(out)   # ch_names omitted → default montage
    assert raw2.ch_names == DEFAULT_CH_NAMES_21


def test_read_lml_custom_ch_names(tmp_path: Path):
    """Caller can pass explicit channel names."""
    from lamquant_codec.mne_io import write_lml, read_raw_lml
    raw, _ = _make_raw(n_channels=4)
    out = tmp_path / 'r.lml'
    write_lml(raw, out)
    custom = ['Cz', 'Fpz', 'Oz', 'Pz']
    raw2 = read_raw_lml(out, ch_names=custom)
    assert raw2.ch_names == custom


def test_read_lml_validates_ch_names_length(tmp_path: Path):
    from lamquant_codec.mne_io import write_lml, read_raw_lml
    raw, _ = _make_raw(n_channels=4)
    out = tmp_path / 'r.lml'
    write_lml(raw, out)
    with pytest.raises(ValueError, match='ch_names'):
        read_raw_lml(out, ch_names=['a', 'b'])     # wrong count


# ============================================================
# Auto-dispatch by extension
# ============================================================

def test_read_raw_dispatches_lml(tmp_path: Path):
    """read_raw('.lml') must work without a checkpoint."""
    from lamquant_codec.mne_io import write_lml, read_raw
    raw, expected_uv = _make_raw()
    out = tmp_path / 'r.lml'
    write_lml(raw, out)
    raw2 = read_raw(out)        # no checkpoint argument
    assert np.array_equal(
        np.round(raw2.get_data() * 1e6).astype(np.int64),
        np.round(expected_uv).astype(np.int64),
    )


def test_read_raw_lmq_requires_checkpoint(tmp_path: Path):
    """read_raw on a .lmq file must demand a checkpoint."""
    from lamquant_codec.mne_io import read_raw
    fake_lmq = tmp_path / 'r.lmq'
    fake_lmq.write_bytes(b'LMQ1' + b'\x00' * 100)
    with pytest.raises(ValueError, match='checkpoint'):
        read_raw(fake_lmq)


def test_read_raw_unknown_extension_raises(tmp_path: Path):
    from lamquant_codec.mne_io import read_raw
    f = tmp_path / 'r.txt'
    f.write_bytes(b'not eeg')
    with pytest.raises(ValueError, match='unsupported extension'):
        read_raw(f)


def test_write_raw_dispatches_lml(tmp_path: Path):
    from lamquant_codec.mne_io import write_raw, read_raw_lml
    raw, expected_uv = _make_raw()
    out = tmp_path / 'r.lml'
    write_raw(raw, out)         # no checkpoint needed
    raw2 = read_raw_lml(out)
    assert np.array_equal(
        np.round(raw2.get_data() * 1e6).astype(np.int64),
        np.round(expected_uv).astype(np.int64),
    )


def test_write_raw_lmq_requires_checkpoint(tmp_path: Path):
    from lamquant_codec.mne_io import write_raw
    raw, _ = _make_raw()
    with pytest.raises(ValueError, match='checkpoint'):
        write_raw(raw, tmp_path / 'r.lmq')      # no checkpoint kwarg


# ============================================================
# Unit conversion (V ↔ µV)
# ============================================================

def test_units_uv_false_skips_conversion(tmp_path: Path):
    """units_uv=False on both write and read preserves the V scale."""
    from lamquant_codec.mne_io import write_lml, read_raw_lml
    # Build a raw whose data is already in V-scaled integer space (so
    # rounding to int doesn't destroy it).
    rng = np.random.default_rng(0)
    data = rng.integers(-2000, 2000, (4, 500)).astype(np.float64)
    info = mne.create_info(['a', 'b', 'c', 'd'], sfreq=250.0, ch_types='eeg')
    raw = mne.io.RawArray(data, info, verbose=False)

    out = tmp_path / 'r.lml'
    write_lml(raw, out, units_uv=False)
    raw2 = read_raw_lml(out, ch_names=['a', 'b', 'c', 'd'], units_uv=False)
    assert np.array_equal(raw2.get_data().astype(np.int64),
                          data.astype(np.int64))


# ============================================================
# Lazy import surface from the package root
# ============================================================

def test_lazy_attribute_access_via_package_root():
    """`from lamquant_codec import read_raw_lml` triggers lazy load."""
    import lamquant_codec
    fn = lamquant_codec.read_raw_lml
    assert callable(fn)
    fn2 = lamquant_codec.read_raw
    assert callable(fn2)


# ============================================================
# Error message when mne missing
# ============================================================

def test_friendly_error_when_mne_missing(monkeypatch):
    """If mne is uninstalled, _require_mne() should suggest pip install."""
    from lamquant_codec import mne_io
    monkeypatch.setitem(sys.modules, 'mne', None)
    # Forcing the module to None makes import mne fail.
    with pytest.raises(ImportError, match='pip install mne'):
        mne_io._require_mne()
