"""lamquant_codec.training — canonical LMA-direct training datasets.

Single source of truth for the LMA → PyTorch ``Dataset`` pipeline used
by every train_*.py kernel post-2026-05-16. Replaces the deprecated
NPZ + L3-precompute caches under ``ai_models/dataset_sim/`` and
``ai_models/student/``.

Public surface:
    - ``LmaSignalDataset``  yields ``(signal[21, 2500])`` raw fullband windows.
    - ``LmaL3Dataset``      yields ``(l3[21, 313], l3[21, 313], dummy_mask)``
                            — drop-in replacement for ``PrecomputedL3Dataset``.
    - helper utilities re-exported from ``lma_dataset`` (decode + LRU cache).

See ADR 0017 (BLUT canonical trainer + LMA-direct).
"""

from lamquant_codec.training.lma_dataset import (
    LmaL3Dataset,
    LmaSignalDataset,
    build_lma_entry_index,
    decode_lma_signal,
    list_lma_entries,
    load_split_stems,
    select_random_windows,
)

__all__ = [
    "LmaL3Dataset",
    "LmaSignalDataset",
    "build_lma_entry_index",
    "decode_lma_signal",
    "list_lma_entries",
    "load_split_stems",
    "select_random_windows",
]
