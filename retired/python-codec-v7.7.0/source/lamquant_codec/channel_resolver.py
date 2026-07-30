"""
LamQuant — Canonical 10-20 Channel Resolver
============================================
Single source of truth for mapping any EDF channel name to the
canonical 21-channel 10-20 montage used by all LamQuant pipelines.

Handles:
  - Standard 10-20 names (Fp1, F3, C3, ...)
  - Case variants (FP1, fp1, fP1)
  - Modern renames (T7->T3, T8->T4, P7->T5, P8->T6)
  - Bipolar montage (FP1-F7, P3-O1, CZ-PZ) — both electrodes extracted
  - PhysioNet reference suffixes (EEG Fp1-REF, EEG F3-LE)
  - Dot-padded names (Fc5., C3.., Fp1.)
  - Whitespace and prefix stripping (EEG, EEG-, EEG_)

Imported by: edf_to_events.py, generate_activity_labels.py,
             generate_validation_split.py, validate_cross_dataset.py,
             validate_subband.py

To add support for a new dataset's naming convention, update the
CHANNEL_ALIASES dict or the resolve() function below. All downstream
scripts pick up the change automatically.
"""

import re
from typing import Optional, List, Dict, Set, Tuple

# ============================================================
# Canonical 21-channel 10-20 montage (firmware contract)
# ============================================================

TARGET_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'A1', 'A2'
]

# Channels that are OK to zero-fill if absent (ear references)
OPTIONAL_CHANNELS = {'A1', 'A2'}

# Minimum usable channels before rejecting a file
MIN_REQUIRED_CHANNELS = 16

# 8 spatial groups — matches firmware SNN readout topology
SPATIAL_GROUPS = [
    [0, 1],              # Group 0: Fp1, Fp2
    [2, 3, 16],          # Group 1: F3, F4, Fz
    [10, 11],            # Group 2: F7, F8
    [4, 5, 17],          # Group 3: C3, C4, Cz
    [12, 13],            # Group 4: T3, T4
    [14, 15],            # Group 5: T5, T6
    [6, 7, 18],          # Group 6: P3, P4, Pz
    [8, 9, 19, 20],      # Group 7: O1, O2, A1, A2
]

# ============================================================
# Alias table: raw name -> canonical name
# ============================================================

# Build a comprehensive case-insensitive lookup table.
# Every entry maps to exactly one canonical name.
_ALIASES: Dict[str, str] = {}

def _add(raw: str, canonical: str):
    _ALIASES[raw] = canonical
    _ALIASES[raw.upper()] = canonical
    _ALIASES[raw.lower()] = canonical

# Identity mappings (canonical names)
for ch in TARGET_CHANNELS:
    _add(ch, ch)

# Modern 10-20 renames
_add('T7', 'T3'); _add('T8', 'T4'); _add('P7', 'T5'); _add('P8', 'T6')

# Case variants not covered by identity
_add('FP1', 'Fp1'); _add('FP2', 'Fp2')
_add('FZ', 'Fz');   _add('CZ', 'Cz');  _add('PZ', 'Pz')
# OZ intentionally not mapped — Oz is not part of our 21-channel set

# PhysioNet reference/linked-ear suffixes: EEG Fp1-REF, EEG F3-LE
for ch in TARGET_CHANNELS:
    for prefix in ['EEG ', 'EEG-', 'EEG_']:
        for suffix in ['-REF', '-LE', '-Ref', '-ref', '']:
            _add(f'{prefix}{ch}{suffix}', ch)
            _add(f'{prefix}{ch.upper()}{suffix}', ch)

# Modern names with suffixes
for raw, canon in [('T7','T3'),('T8','T4'),('P7','T5'),('P8','T6')]:
    for prefix in ['EEG ', 'EEG-', 'EEG_']:
        for suffix in ['-REF', '-LE', '-Ref', '-ref', '']:
            _add(f'{prefix}{raw}{suffix}', canon)

# Dot-padded names (EEGMMIDB): Fp1., F3.., C3.
for ch in TARGET_CHANNELS:
    _add(f'{ch}.', ch)
    _add(f'{ch}..', ch)

# Modern renames dot-padded
_add('T7.', 'T3'); _add('T8.', 'T4'); _add('P7.', 'T5'); _add('P8.', 'T6')


# ============================================================
# Public API
# ============================================================

def resolve(raw_name: str) -> Optional[str]:
    """Resolve any EDF channel name to canonical 10-20 name.

    Handles direct aliases, case variants, EEG prefixes, bipolar
    montage names, and dot-padded names.

    Args:
        raw_name: Channel name as it appears in the EDF file.

    Returns:
        Canonical channel name (e.g., 'Fp1', 'T3'), or None if
        the channel is not part of the 10-20 montage.
    """
    name = raw_name.strip().rstrip('.')

    # Exclude annotation channels early. Tightened from a broad substring
    # check to known EDF/BDF annotation channel patterns only, so we don't
    # accidentally reject hypothetical channels containing "annotation".
    upper = name.upper().strip()
    if upper.startswith('EDF ANNOTATION') or upper.startswith('BDF ANNOTATION'):
        return None

    # Strip MNE dedup suffixes (e.g., "Fp1-0" → "Fp1", "Cz-1" → "Cz")
    name = re.sub(r'-\d+$', '', name)

    # 1. Direct lookup (covers most cases)
    if name in _ALIASES:
        return _ALIASES[name]

    # 2. Case-insensitive direct lookup
    name_upper = name.upper()
    if name_upper in _ALIASES:
        return _ALIASES[name_upper]

    # 3. Strip "EEG" prefix and re-try
    stripped = re.sub(r'^EEG[\s\-_]+', '', name, flags=re.IGNORECASE).strip()
    if stripped != name:
        # Remove reference suffix (-REF, -LE, -Ref, etc.)
        stripped = re.sub(r'-(REF|LE|Ref|ref)$', '', stripped).strip()
        if stripped in _ALIASES:
            return _ALIASES[stripped]
        if stripped.upper() in _ALIASES:
            return _ALIASES[stripped.upper()]

    # 4. Bipolar montage: "FP1-F7", "P3-O1", "CZ-PZ"
    #    Try extracting individual electrode names from both sides
    if '-' in name:
        parts = name.split('-')
        # Try first electrode
        first = re.sub(r'^EEG[\s\-_]+', '', parts[0].strip(), flags=re.IGNORECASE)
        result = _try_resolve_atom(first)
        if result is not None:
            return result
        # Try second electrode (strip MNE dedup suffix like "-0", "-1")
        if len(parts) >= 2:
            second = parts[1].strip()
            second = re.sub(r'^(\D+)\d+$', r'\1', second)  # "P8-0" -> "P8"
            # But don't strip digits from real electrode names like "O1", "A2"
            result = _try_resolve_atom(parts[1].strip())
            if result is not None:
                return result
            result = _try_resolve_atom(second)
            if result is not None:
                return result

    # 5. Dot stripping (EEGMMIDB: "Fc5.", "C3..")
    stripped_dots = name.replace('.', '').strip()
    if stripped_dots != name and stripped_dots in _ALIASES:
        return _ALIASES[stripped_dots]

    # 6. Bounded substring match as last resort (TUH: "Cz (REF)" → Cz)
    # Require canonical name to be at a word boundary to avoid false
    # positives like "A10" resolving to "A1" or "Fp10" to "Fp1".
    name_upper = name.upper()
    for canonical in TARGET_CHANNELS:
        pattern = r'(?<![A-Z0-9])' + re.escape(canonical.upper()) + r'(?![A-Z0-9])'
        if re.search(pattern, name_upper):
            return canonical

    return None


def _try_resolve_atom(name: str) -> Optional[str]:
    """Try to resolve a single electrode name (no bipolar, no prefix)."""
    name = name.strip().rstrip('.')
    if name in _ALIASES:
        return _ALIASES[name]
    if name.upper() in _ALIASES:
        return _ALIASES[name.upper()]
    return None


def select_channels(ch_names: List[str],
                    require_all: bool = False
                    ) -> Tuple[Optional[Dict[str, int]], List[str]]:
    """Map a list of raw EDF channel names to canonical indices.

    Two-pass resolution:
      Pass 1: Try each raw name directly.
      Pass 2: For still-missing channels, try the SECOND electrode
              from bipolar pairs (e.g., "P3-O1" -> O1).

    Args:
        ch_names: List of raw channel names from the EDF file.
        require_all: If True, return None unless all 19 required channels
                     are found (A1/A2 are always optional).

    Returns:
        (mapping, missing) where mapping is {canonical_name: raw_index}
        and missing is a list of unresolved required channel names.
        Returns (None, missing) if too many channels are missing.
    """
    # Pass 1: standard resolution
    mapping: Dict[str, int] = {}
    for idx, raw_name in enumerate(ch_names):
        canonical = resolve(raw_name)
        if canonical and canonical in TARGET_CHANNELS and canonical not in mapping:
            mapping[canonical] = idx

    # Pass 2: second electrode from bipolar pairs for still-missing channels
    still_missing = set(TARGET_CHANNELS) - set(mapping.keys()) - OPTIONAL_CHANNELS
    if still_missing:
        for idx, raw_name in enumerate(ch_names):
            name = raw_name.strip()
            if '-' not in name:
                continue
            parts = name.split('-')
            if len(parts) < 2:
                continue
            # Try second electrode only
            second = parts[1].strip()
            second_clean = re.sub(r'-\d+$', '', second)  # MNE dedup
            for candidate in [second, second_clean]:
                canonical = _try_resolve_atom(candidate)
                if canonical and canonical in still_missing and canonical not in mapping:
                    mapping[canonical] = idx

    # Determine what's still missing
    missing_required = [ch for ch in TARGET_CHANNELS
                        if ch not in mapping and ch not in OPTIONAL_CHANNELS]

    if require_all and missing_required:
        return None, missing_required

    if len(mapping) < MIN_REQUIRED_CHANNELS:
        return None, missing_required

    return mapping, missing_required


def pick_channels(ch_names: List[str]) -> Optional[List[int]]:
    """Return ordered indices into ch_names for our 21 channels.

    Returns None if fewer than MIN_REQUIRED_CHANNELS are found.
    Missing optional channels (A1, A2) get index -1 (caller must
    zero-fill).
    """
    mapping, missing = select_channels(ch_names)
    if mapping is None:
        return None

    indices = []
    for ch in TARGET_CHANNELS:
        if ch in mapping:
            indices.append(mapping[ch])
        else:
            indices.append(-1)  # Caller zero-fills this channel
    return indices


def extract_channel_data(all_data, ch_names: List[str]):
    """Extract and reorder channel data to match TARGET_CHANNELS.

    Args:
        all_data: numpy array [total_channels, T] from EDF file.
        ch_names: raw channel names matching all_data's first axis.

    Returns:
        (data, missing) where data is [21, T] numpy array with channels
        ordered per TARGET_CHANNELS. Missing optional channels are
        zero-filled. Returns (None, missing) if too many required
        channels are absent.
    """
    import numpy as np

    mapping, missing = select_channels(ch_names)
    if mapping is None:
        return None, missing

    T = all_data.shape[1]
    data = np.zeros((len(TARGET_CHANNELS), T), dtype=all_data.dtype)

    for i, ch in enumerate(TARGET_CHANNELS):
        if ch in mapping:
            data[i] = all_data[mapping[ch]]
        # else: zero-filled (optional channels A1, A2, or rare missing)

    return data, missing
