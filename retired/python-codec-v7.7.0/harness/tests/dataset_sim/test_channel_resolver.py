"""Unit tests for ai_models.dataset_sim.channel_resolver — Phase 3.

Pure-logic 10-20 channel-name normaliser. Targets resolve() with all
its branches (direct lookup, case variants, EEG-prefix, bipolar
extraction, dot padding, substring fallback) + select_channels /
pick_channels / extract_channel_data orchestration.

ai_models/dataset_sim/channel_resolver.py at ~48% baseline. These
tests push it to ~95%.
"""
from __future__ import annotations

import numpy as np
import pytest

from ai_models.dataset_sim.channel_resolver import (
    MIN_REQUIRED_CHANNELS,
    OPTIONAL_CHANNELS,
    SPATIAL_GROUPS,
    TARGET_CHANNELS,
    extract_channel_data,
    pick_channels,
    resolve,
    select_channels,
)
# Private helpers live in the canonical module after the
# lamquant_codec.channel_resolver move (audit 2026-05-18 F1 / B2);
# the shim's ``from ... import *`` only exposes public symbols.
from lamquant_codec.channel_resolver import _try_resolve_atom

pytestmark = pytest.mark.l2


# ============================================================
# resolve — every branch
# ============================================================


class TestResolve:

    def test_identity_for_canonical_names(self) -> None:
        for ch in TARGET_CHANNELS:
            assert resolve(ch) == ch

    def test_case_variant_uppercase(self) -> None:
        # FP1 → Fp1, CZ → Cz etc.
        assert resolve("FP1") == "Fp1"
        assert resolve("CZ") == "Cz"
        assert resolve("FZ") == "Fz"

    def test_case_variant_lowercase(self) -> None:
        assert resolve("fp1") == "Fp1"
        assert resolve("cz") == "Cz"

    def test_modern_rename_t7_to_t3(self) -> None:
        # T7/T8/P7/P8 are modern 10-20 names — map to legacy T3/T4/T5/T6.
        assert resolve("T7") == "T3"
        assert resolve("T8") == "T4"
        assert resolve("P7") == "T5"
        assert resolve("P8") == "T6"

    def test_eeg_prefix(self) -> None:
        assert resolve("EEG Fp1") == "Fp1"
        assert resolve("EEG-F3") == "F3"
        assert resolve("EEG_C3") == "C3"

    def test_eeg_prefix_with_reference_suffix(self) -> None:
        # PhysioNet TUH-style.
        assert resolve("EEG Fp1-REF") == "Fp1"
        assert resolve("EEG F3-LE") == "F3"

    def test_dot_padded_names(self) -> None:
        # EEGMMIDB convention.
        assert resolve("Fp1.") == "Fp1"
        assert resolve("C3..") == "C3"

    def test_bipolar_montage_extracts_first_electrode(self) -> None:
        # FP1-F7 → Fp1 (first part).
        assert resolve("FP1-F7") == "Fp1"
        assert resolve("P3-O1") == "P3"

    def test_bipolar_montage_falls_back_to_second(self) -> None:
        # If first not in canonical, try second.
        # "X1-T3" — X1 not canonical, T3 is.
        out = resolve("X1-T3")
        assert out == "T3"

    def test_mne_dedup_suffix_stripped(self) -> None:
        # "Fp1-0", "Cz-1" (MNE adds these on duplicates).
        assert resolve("Fp1-0") == "Fp1"
        assert resolve("Cz-1") == "Cz"

    def test_rejects_annotation_channels(self) -> None:
        assert resolve("EDF Annotations") is None
        assert resolve("BDF Annotations") is None
        # Case-insensitive.
        assert resolve("edf annotation") is None

    def test_returns_none_for_unknown_channel(self) -> None:
        # E.g. ECG, EOG, EMG — not part of 10-20 EEG montage.
        assert resolve("ECG") is None
        assert resolve("EOG1") is None
        assert resolve("EMG_left") is None

    def test_substring_fallback_for_parenthetical(self) -> None:
        # TUH: "Cz (REF)" → Cz via substring match.
        assert resolve("Cz (REF)") == "Cz"

    def test_substring_avoids_false_positives_on_overlapping_names(self) -> None:
        # Word boundary guards prevent "A10" → A1 collision.
        assert resolve("A10") != "A1"
        # And similarly Fp10 should not resolve to Fp1.
        assert resolve("Fp10") != "Fp1"


# ============================================================
# _try_resolve_atom
# ============================================================


class TestTryResolveAtom:

    def test_resolves_canonical(self) -> None:
        assert _try_resolve_atom("Fp1") == "Fp1"

    def test_resolves_case_insensitive(self) -> None:
        assert _try_resolve_atom("fp1") == "Fp1"

    def test_strips_trailing_dots(self) -> None:
        assert _try_resolve_atom("Fp1.") == "Fp1"

    def test_returns_none_unknown(self) -> None:
        assert _try_resolve_atom("zzz") is None


# ============================================================
# select_channels
# ============================================================


class TestSelectChannels:

    def _full_set(self) -> list[str]:
        # All 21 in canonical form.
        return list(TARGET_CHANNELS)

    def test_complete_set_returns_full_mapping(self) -> None:
        mapping, missing = select_channels(self._full_set())
        assert mapping is not None
        assert len(mapping) == 21
        assert missing == []

    def test_drops_duplicate_resolutions(self) -> None:
        # Two raw names both resolving to "Fp1" — first wins, second
        # ignored (canonical not in mapping check).
        names = ["Fp1", "FP1", "F3", "F4", "C3", "C4", "P3", "P4",
                 "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
                 "Fz", "Cz", "Pz", "A1", "A2"]
        mapping, _ = select_channels(names)
        assert mapping is not None
        assert mapping["Fp1"] == 0  # First match wins.

    def test_missing_optional_channels_ok(self) -> None:
        names = [ch for ch in TARGET_CHANNELS if ch not in OPTIONAL_CHANNELS]
        mapping, missing = select_channels(names)
        assert mapping is not None
        # Required 19 present, optional A1/A2 missing — still passes.
        assert missing == []

    def test_too_few_channels_returns_none(self) -> None:
        # Only 5 channels — well below MIN_REQUIRED_CHANNELS.
        mapping, missing = select_channels(["Fp1", "F3", "C3", "P3", "O1"])
        assert mapping is None
        assert len(missing) >= MIN_REQUIRED_CHANNELS - 5

    def test_require_all_true_rejects_missing_required(self) -> None:
        # Drop one required channel.
        names = [ch for ch in TARGET_CHANNELS if ch != "Fp1"]
        mapping, missing = select_channels(names, require_all=True)
        assert mapping is None
        assert "Fp1" in missing

    def test_pass_two_recovers_bipolar_second_electrode(self) -> None:
        # Use bipolar pairs that the first-electrode pass misses but
        # second electrode recovers.
        names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
                 "O1", "O2", "F7", "F8", "Cz",
                 # Bipolar pairs whose second component is T3/T4/T5/T6.
                 "XX1-T3", "XX2-T4", "XX3-T5", "XX4-T6",
                 "Fz", "Pz"]
        mapping, _ = select_channels(names)
        # Pass 2 should have picked up T3/T4/T5/T6 from the bipolar
        # second electrodes.
        assert mapping is not None
        for ch in ("T3", "T4", "T5", "T6"):
            assert ch in mapping, f"{ch} not recovered via pass 2"


# ============================================================
# pick_channels
# ============================================================


class TestPickChannels:

    def test_returns_indices_in_target_order(self) -> None:
        names = list(TARGET_CHANNELS)
        indices = pick_channels(names)
        assert indices is not None
        assert len(indices) == 21
        # Identity mapping when input matches TARGET_CHANNELS exactly.
        assert indices == list(range(21))

    def test_missing_optional_returns_minus_one(self) -> None:
        names = [ch for ch in TARGET_CHANNELS if ch not in OPTIONAL_CHANNELS]
        indices = pick_channels(names)
        assert indices is not None
        # A1 (index 19) + A2 (index 20) should be -1.
        assert indices[19] == -1
        assert indices[20] == -1

    def test_returns_none_when_below_min(self) -> None:
        assert pick_channels(["Fp1", "Cz"]) is None


# ============================================================
# extract_channel_data
# ============================================================


class TestExtractChannelData:

    def test_extracts_full_set_in_canonical_order(self) -> None:
        T = 100
        # Reverse-order data so we can verify reordering.
        names = list(reversed(TARGET_CHANNELS))
        all_data = np.arange(21 * T).reshape(21, T).astype(np.float32)
        # Row i of all_data corresponds to names[i] (reversed canonical).
        data, missing = extract_channel_data(all_data, names)
        assert data is not None
        assert data.shape == (21, T)
        # First row of `data` should be Fp1's row in `all_data`.
        fp1_idx_in_names = names.index("Fp1")
        assert (data[0] == all_data[fp1_idx_in_names]).all()

    def test_zero_fills_missing_optional(self) -> None:
        # Drop A1/A2.
        T = 50
        names = [ch for ch in TARGET_CHANNELS if ch not in OPTIONAL_CHANNELS]
        all_data = np.ones((19, T), dtype=np.float32)
        data, _ = extract_channel_data(all_data, names)
        assert data is not None
        # A1 (idx 19) and A2 (idx 20) should be zero-filled.
        assert (data[19] == 0).all()
        assert (data[20] == 0).all()

    def test_returns_none_when_too_few_channels(self) -> None:
        T = 10
        names = ["Fp1", "F3"]
        all_data = np.zeros((2, T), dtype=np.float32)
        data, missing = extract_channel_data(all_data, names)
        assert data is None
        assert len(missing) >= MIN_REQUIRED_CHANNELS - 2


# ============================================================
# Module constants
# ============================================================


class TestConstants:

    def test_target_channels_is_21(self) -> None:
        assert len(TARGET_CHANNELS) == 21

    def test_optional_channels_is_a1_a2(self) -> None:
        assert OPTIONAL_CHANNELS == {"A1", "A2"}

    def test_min_required_below_total(self) -> None:
        assert 0 < MIN_REQUIRED_CHANNELS < len(TARGET_CHANNELS)

    def test_spatial_groups_partition_target_channels(self) -> None:
        # Every target channel index 0..20 should appear in exactly
        # one spatial group.
        seen = []
        for group in SPATIAL_GROUPS:
            seen.extend(group)
        assert sorted(seen) == list(range(len(TARGET_CHANNELS)))
