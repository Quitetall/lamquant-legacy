"""MNE-Python integration for LamQuant.

Researchers using MNE-Python don't have to leave their workflow — they can
read .lmq / .lml files directly into the `mne.io.Raw` objects their code
already expects.

Public API
----------
    read_raw_lmq(path, *, checkpoint=None, ch_names=None) -> mne.io.RawArray
    read_raw_lml(path, *, ch_names=None)                  -> mne.io.RawArray
    read_raw(path, *, checkpoint=None, ch_names=None)     -> mne.io.RawArray
        Auto-dispatches by extension. Use this when you don't care which
        codec mode the file uses.

    write_lmq(raw, path, *, checkpoint, quality='clinical') -> Path
    write_lml(raw, path)                                    -> Path
    write_raw(raw, path, *, checkpoint=None, **kwargs)      -> Path
        Auto-dispatches by output extension.

mne is imported lazily — `import lamquant_codec` does not pull mne (it's
~300 ms at import time and not all users need it). Only the call site that
invokes one of these functions pays the load cost.

Channel info
------------
For .lml files (lossless, integer-domain), the original channel labels are
recovered if they were written by `write_lml(raw)`. For .lmq files (neural,
no per-recording metadata), pass `ch_names` explicitly or fall back to the
standard 10-20 montage.

Sample rate
-----------
Both formats currently assume 250 Hz (the LamQuant Gen 7 design rate). If
your study uses a different rate, pass `sfreq` (read functions return Raw
at 250 Hz; resample upstream if needed).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence


# Standard 10-20 EEG channel labels (21-electrode clinical montage).
DEFAULT_CH_NAMES_21 = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'A1', 'A2',
]
DEFAULT_SFREQ = 250.0


def _require_mne():
    """Lazy mne import with a friendly install hint."""
    try:
        import mne
        return mne
    except ImportError:
        raise ImportError(
            "MNE integration requires mne-python. Install with:\n"
            "    pip install mne") from None


def _make_info(n_channels: int, sfreq: float = DEFAULT_SFREQ,
               ch_names: Optional[Sequence[str]] = None):
    """Build an mne.Info object for an EEG recording."""
    mne = _require_mne()
    if ch_names is None:
        if n_channels == len(DEFAULT_CH_NAMES_21):
            ch_names = list(DEFAULT_CH_NAMES_21)
        else:
            ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
    elif len(ch_names) != n_channels:
        raise ValueError(
            f"ch_names has {len(ch_names)} entries but signal has "
            f"{n_channels} channels")
    info = mne.create_info(ch_names=list(ch_names), sfreq=sfreq, ch_types='eeg')
    return info


def _signal_to_raw(signal, sfreq: float = DEFAULT_SFREQ,
                    ch_names: Optional[Sequence[str]] = None,
                    units_uv: bool = True):
    """Wrap a [C, T] numpy array as an mne.io.RawArray.

    MNE expects EEG in volts internally; LamQuant stores microvolts so we
    convert by 1e-6 (toggle with units_uv=False if your data is already V).
    """
    import numpy as np
    mne = _require_mne()
    sig = np.asarray(signal, dtype=np.float64)
    if sig.ndim == 3 and sig.shape[0] == 1:
        sig = sig[0]
    if sig.ndim != 2:
        raise ValueError(f"expected [C, T] signal, got shape {sig.shape}")
    if units_uv:
        sig = sig * 1e-6   # µV → V
    info = _make_info(sig.shape[0], sfreq=sfreq, ch_names=ch_names)
    raw = mne.io.RawArray(sig, info, verbose=False)
    return raw


# ============================================================
# Read .lml — lossless, no checkpoint needed
# ============================================================

def read_raw_lml(path,
                 *,
                 sfreq: float = DEFAULT_SFREQ,
                 ch_names: Optional[Sequence[str]] = None,
                 units_uv: bool = True):
    """Read a LamQuant lossless (.lml) file as an mne.io.RawArray.

    Args:
        path: .lml file path.
        sfreq: Sample rate in Hz (default 250).
        ch_names: Optional channel labels. If omitted and the file has 21
                  channels, the standard 10-20 montage is assigned.
        units_uv: Treat the stored integer samples as microvolts and
                  convert to volts for MNE (default True — matches the
                  LamQuant convention).

    Returns: mne.io.RawArray
    """
    from lamquant_codec.lossless import LosslessCodec
    path = Path(path)
    with open(path, 'rb') as f:
        data = f.read()
    signal = LosslessCodec().decompress(data)
    return _signal_to_raw(signal, sfreq=sfreq, ch_names=ch_names,
                          units_uv=units_uv)


# ============================================================
# Read .lmq — neural, requires the encoder checkpoint
# ============================================================

def read_raw_lmq(path,
                 *,
                 checkpoint,
                 sfreq: float = DEFAULT_SFREQ,
                 ch_names: Optional[Sequence[str]] = None,
                 units_uv: bool = True):
    """Read a LamQuant neural (.lmq) file as an mne.io.RawArray.

    Args:
        path: .lmq file path.
        checkpoint: Path to the encoder checkpoint (required — neural mode
                    is model-conditional).
        sfreq, ch_names, units_uv: see read_raw_lml.

    Returns: mne.io.RawArray
    """
    import numpy as np
    import torch
    try:
        from lamquant_neural.codec import SubbandCodec
    except ImportError as exc:
        raise RuntimeError(
            "reading neural (.lmq) via MNE requires the neural codec, now in "
            "LamQuant-Neural (pip install lamquant-neural). Lossless (.lml) "
            "reads need only lamquant-codec."
        ) from exc
    path = Path(path)
    with open(path, 'rb') as f:
        data = f.read()
    codec = SubbandCodec.from_checkpoint(str(checkpoint))
    latent, _quality, _lpc, _det = codec.decompress(data)
    with torch.no_grad():
        recon = codec.model.decode(latent, target_len=313, quantize=True)
    signal = recon[0].cpu().numpy() if hasattr(recon, 'cpu') else np.asarray(recon[0])
    return _signal_to_raw(signal, sfreq=sfreq, ch_names=ch_names,
                          units_uv=units_uv)


def read_raw(path,
             *,
             checkpoint=None,
             sfreq: float = DEFAULT_SFREQ,
             ch_names: Optional[Sequence[str]] = None,
             units_uv: bool = True):
    """Auto-dispatch by extension. Use this when you don't care which
    codec mode the file uses.

    .lml → read_raw_lml (no checkpoint needed)
    .lmq → read_raw_lmq (checkpoint required)
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == '.lml':
        return read_raw_lml(path, sfreq=sfreq, ch_names=ch_names,
                            units_uv=units_uv)
    if suffix == '.lmq':
        if not checkpoint:
            raise ValueError(".lmq files require a checkpoint")
        return read_raw_lmq(path, checkpoint=checkpoint, sfreq=sfreq,
                            ch_names=ch_names, units_uv=units_uv)
    raise ValueError(
        f"unsupported extension {suffix!r}; expected .lml or .lmq")


# ============================================================
# Write — Raw → .lml / .lmq
# ============================================================

def _raw_to_signal(raw, units_uv: bool = True):
    """Convert an mne.io.Raw to a LamQuant-compatible [C, T] array.

    MNE stores EEG in volts; LamQuant works in microvolts (matches the
    ADS1299 ADC convention). When units_uv=True (default), the V→µV
    conversion is applied so the round-trip preserves the physical scale.
    """
    import numpy as np
    sig = raw.get_data()
    if units_uv:
        sig = sig * 1e6   # V → µV
    return np.asarray(sig, dtype=np.float64)


def write_lml(raw, path, *, units_uv: bool = True):
    """Encode an mne.io.Raw as a LamQuant lossless .lml file.

    Bit-exact: read_raw_lml(write_lml(raw)) reproduces raw exactly (up to
    integer rounding from the lossless codec's int16 sample domain).
    """
    from lamquant_codec.lossless import LosslessCodec
    path = Path(path)
    signal = _raw_to_signal(raw, units_uv=units_uv)
    data = LosslessCodec().compress(signal)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(data)
    return path


def write_lmq(raw, path, *, checkpoint, quality: str = 'clinical',
              units_uv: bool = True):
    """Encode an mne.io.Raw as a LamQuant neural .lmq file.

    Args:
        raw: mne.io.Raw / RawArray
        path: output path
        checkpoint: encoder checkpoint
        quality: 'alerting' | 'monitoring' | 'clinical' (default clinical)
        units_uv: V → µV conversion before encoding
    """
    import numpy as np
    import torch
    try:
        from lamquant_neural.codec import SubbandCodec
    except ImportError as exc:
        raise RuntimeError(
            "writing neural (.lmq) via MNE requires the neural codec, now in "
            "LamQuant-Neural (pip install lamquant-neural). Use write_lml for "
            "the lossless path."
        ) from exc
    path = Path(path)
    qmap = {'alerting': 0, 'monitoring': 1, 'clinical': 2}
    if quality not in qmap:
        raise ValueError(f"quality must be one of {list(qmap)}, got {quality!r}")

    signal = _raw_to_signal(raw, units_uv=units_uv)
    codec = SubbandCodec.from_checkpoint(str(checkpoint))
    x = torch.from_numpy(signal[None].astype(np.float32))
    with torch.no_grad():
        latent, metadata = codec.encode(x)
        lpc_coeffs = metadata[0][0] if metadata else None
        subbands = [m[1] for m in metadata] if metadata else None
        data = codec.compress(latent, lpc_coeffs, subbands,
                              quality_mode=qmap[quality])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(data)
    return path


def write_raw(raw, path, *, checkpoint=None, **kwargs):
    """Auto-dispatch by output extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == '.lml':
        return write_lml(raw, path, **{k: v for k, v in kwargs.items()
                                        if k in ('units_uv',)})
    if suffix == '.lmq':
        if not checkpoint:
            raise ValueError(".lmq output requires a checkpoint")
        return write_lmq(raw, path, checkpoint=checkpoint, **kwargs)
    raise ValueError(
        f"unsupported output extension {suffix!r}; expected .lml or .lmq")


__all__ = [
    'read_raw', 'read_raw_lml', 'read_raw_lmq',
    'write_raw', 'write_lml', 'write_lmq',
    'DEFAULT_CH_NAMES_21', 'DEFAULT_SFREQ',
]
