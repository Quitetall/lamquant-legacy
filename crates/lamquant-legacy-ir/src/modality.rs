//! Retired ADR 0069 modality tags used by the LML1/BCS1 compatibility path.
//!
//! `LegacyRecording` is generic over a **sealed** [`Modality`] marker (default
//! [`Untyped`]). This is a compile-time trust boundary, not a runtime check:
//! `LegacyRecording<Eeg>` and `LegacyRecording<Ecg>` are DIFFERENT types, so a modality-typed
//! consumer (e.g. `fn train(&LegacyRecording<Eeg>)`) simply cannot be called with the
//! wrong modality — the mismatch is a type error, not a bug caught in CI.
//!
//! **Reconciling with Pillar 1 (no cross-modality mixing) and the encoder
//! egress.** The codec's hot path (`LegacyRecording::window_views`, consumed by
//! `write_legacy_recording`) is modality-**blind** by design — bytes in, bytes out,
//! generic over any `M: Modality`, `LegacyRecording<Untyped>` included. That is
//! intentional and stays that way: the wire format has never encoded
//! modality, and this step keeps it byte-frozen. The trust boundary is
//! enforced one layer up, at the SEMANTIC accessors: [`LegacyRecording::into_modality`]
//! is the one deliberate, explicit narrowing point (untyped → typed,
//! recorded in `prov`), and [`LegacyRecording::verify`] is the boundary invariant check
//! a typed consumer calls before trusting the data. Nothing downstream of
//! `into_modality` can silently widen back to a different modality or mix
//! two modalities under one type — the sealed trait means only the markers
//! in this module implement `Modality` at all.
//!
//! All markers are zero-sized types (ZSTs); `PhantomData<M>` inside `LegacyRecording<M>`
//! costs nothing at runtime — see the `size_of` proof in `atoms.rs` tests.

mod sealed {
    /// Sealed supertrait — only this crate can implement [`super::Modality`].
    pub trait Sealed {}
}

/// A biosignal modality marker. Sealed: the only implementors are the ZST
/// markers defined in this module (`Untyped`, `Eeg`, `Ieeg`, …) — downstream
/// crates cannot invent new ones, which is what makes the `LegacyRecording<M>` typestate
/// a closed, trustworthy set rather than an open one any caller could spoof.
pub trait Modality: sealed::Sealed + Copy + Clone + core::fmt::Debug + 'static {
    /// Human-readable modality name (for logs/errors/provenance).
    const NAME: &'static str;
    /// The wire-adjacent tag byte recorded in [`ModalityProvenance::tag`].
    /// NOT part of the LML/LMO wire format itself (S3a is encoder-egress
    /// invariant) — this is IR-side bookkeeping only.
    const TAG: u8;
}

macro_rules! modality_marker {
    ($(#[$meta:meta])* $name:ident, $tag:expr, $label:expr) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;

        impl sealed::Sealed for $name {}

        impl Modality for $name {
            const NAME: &'static str = $label;
            const TAG: u8 = $tag;
        }
    };
}

modality_marker!(
    /// The default, modality-erased marker — today's `LegacyRecording` behavior
    /// (`LegacyRecording` == `LegacyRecording<Untyped>`). The encoder egress path
    /// (`window_views`/`write_legacy_recording`) runs on this marker unchanged.
    Untyped,
    255,
    "untyped"
);
modality_marker!(
    /// Scalp electroencephalography.
    Eeg, 0, "eeg"
);
modality_marker!(
    /// Intracranial electroencephalography.
    Ieeg, 1, "ieeg"
);
modality_marker!(
    /// Electrocorticography.
    Ecog, 2, "ecog"
);
modality_marker!(
    /// Stereo-EEG.
    Seeg, 3, "seeg"
);
modality_marker!(
    /// Electrocardiography.
    Ecg, 4, "ecg"
);
modality_marker!(
    /// Electromyography.
    Emg, 5, "emg"
);
modality_marker!(
    /// Electrooculography.
    Eog, 6, "eog"
);
modality_marker!(
    /// Respiration.
    Resp, 7, "resp"
);
modality_marker!(
    /// Accelerometry.
    Accel, 8, "accel"
);
modality_marker!(
    /// Anything not covered by a dedicated marker above.
    Other, 9, "other"
);

/// Map a modality wire tag ([`Modality::TAG`]) to its name ([`Modality::NAME`]),
/// or `None` for an unknown tag. The SINGLE SOURCE OF TRUTH for the tag→name
/// mapping: the table references the same `TAG`/`NAME` associated constants the
/// `modality_marker!` invocations above generate, so a marker whose tag or name
/// changes tracks automatically. Adding a NEW marker requires one new row here,
/// co-located with the definitions (a foreign crate — e.g. `lamquant-py`'s
/// `PyAbir.modality()` — must call this rather than keep its own copy).
pub fn name_for_tag(tag: u8) -> Option<&'static str> {
    const TABLE: &[(u8, &str)] = &[
        (Untyped::TAG, Untyped::NAME),
        (Eeg::TAG, Eeg::NAME),
        (Ieeg::TAG, Ieeg::NAME),
        (Ecog::TAG, Ecog::NAME),
        (Seeg::TAG, Seeg::NAME),
        (Ecg::TAG, Ecg::NAME),
        (Emg::TAG, Emg::NAME),
        (Eog::TAG, Eog::NAME),
        (Resp::TAG, Resp::NAME),
        (Accel::TAG, Accel::NAME),
        (Other::TAG, Other::NAME),
    ];
    TABLE.iter().find(|(t, _)| *t == tag).map(|(_, n)| *n)
}

/// How a modality assignment was decided — carried alongside the tag in
/// [`ModalityProvenance`] so a `verify()` failure (or an audit) can explain
/// WHY a stream ended up typed the way it did.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ModalitySource {
    /// Inferred from a channel label (e.g. "Fp1" → EEG-like 10-20 montage).
    ChannelLabel,
    /// Declared by the source format itself (e.g. a format that carries an
    /// explicit modality field).
    FormatDeclared,
    /// Explicitly set by the caller (`into_modality`, manual construction).
    Manual,
}

impl ModalitySource {
    /// The BCS1 wire byte for this source (ADR 0069/0071 L9 header, offset
    /// 7 `modality_source`): `ChannelLabel=0`, `FormatDeclared=1`,
    /// `Manual=2`.
    pub const fn to_u8(self) -> u8 {
        match self {
            Self::ChannelLabel => 0,
            Self::FormatDeclared => 1,
            Self::Manual => 2,
        }
    }

    /// Parse a BCS1 `modality_source` wire byte. `None` for anything
    /// unrecognized — callers must not silently default to a source.
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::ChannelLabel),
            1 => Some(Self::FormatDeclared),
            2 => Some(Self::Manual),
            _ => None,
        }
    }
}

/// Provenance of an `LegacyRecording`'s modality assignment: how it was decided
/// ([`ModalitySource`]) and which modality it resolved to (`tag`, matching
/// the assigned `Modality::TAG`).
#[derive(Clone, Debug)]
pub struct ModalityProvenance {
    /// How the modality was decided.
    pub source: ModalitySource,
    /// The assigned modality's [`Modality::TAG`].
    pub tag: u8,
}

/// A boundary-invariant violation caught by [`crate::LegacyRecording::verify`].
#[derive(Clone, Debug)]
pub enum VerifyError {
    /// Not every channel's column has the same length as `n_samples`.
    ChannelLengthMismatch {
        /// Index of the first offending channel.
        channel: usize,
        /// That channel's actual column length.
        actual: usize,
        /// The `LegacyRecording`'s declared `n_samples`.
        expected: usize,
    },
    /// `prov.tag` does not match the `LegacyRecording<M>`'s type-level `M::TAG` — the
    /// provenance record has drifted from the type it is attached to.
    ProvenanceTagMismatch {
        /// The tag recorded in `prov`.
        recorded: u8,
        /// The tag the type parameter `M` demands.
        expected: u8,
    },
}

impl core::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            VerifyError::ChannelLengthMismatch {
                channel,
                actual,
                expected,
            } => write!(
                f,
                "channel {channel} has {actual} samples, expected n_samples={expected}"
            ),
            VerifyError::ProvenanceTagMismatch { recorded, expected } => write!(
                f,
                "provenance tag {recorded} does not match modality tag {expected}"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for VerifyError {}

// ─────────────────── S3b: born-typed lowering (inference) ───────────────────
//
// `into_modality` (deliberate, manual) and `verify` (boundary check) are the
// S3a trust model. S3b adds the FRONT half: infer the modality from the
// recording itself at lowering time so a reader's `LegacyRecording<Untyped>` never
// defaults to `{Manual, 255}` silently — see [`infer_modality`] below and
// `LegacyRecording::with_inferred_modality` (`atoms.rs`) which wires it into every
// reader's `lower_to_abir`. The recorded inference is then the thing
// `LegacyRecording::try_into_modality` checks against an asserted type — a VERIFIED
// promotion, as opposed to `into_modality`'s unchecked override.

/// The canonical 10-20 EEG montage electrode names (upper-cased). Matched
/// per-TOKEN (see [`label_tokens`]) so a bipolar-derivation label like
/// `"Fp1-F7"` still reads as EEG — both halves are in this set.
const EEG_1020_ELECTRODES: &[&str] = &[
    "FP1", "FP2", "F3", "F4", "F7", "F8", "FZ", "C3", "C4", "CZ", "P3", "P4", "PZ", "O1", "O2",
    "T3", "T4", "T5", "T6", "T7", "T8", "A1", "A2", "M1", "M2",
];

/// The standard 12-lead ECG lead names (upper-cased). Matched per-token,
/// but — unlike the EEG electrode set above — a SINGLE occurrence is
/// deliberately NOT enough evidence: `"I"`, `"II"`, `"V1"`, … are short
/// and generic enough that one incidental match could be coincidence (a
/// non-ECG channel that happens to be numbered/lettered this way). Only
/// once at least two distinct lead-name channels co-occur does
/// [`infer_modality`] credit them as ECG evidence.
const ECG_12_LEADS: &[&str] = &[
    "I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6",
];

/// Split an already-uppercased label into tokens on the separators real
/// montage/lead naming conventions use (`-`, `_`, `/`, whitespace), so
/// `"FP1-F7"` yields `["FP1", "F7"]` and `"LEAD I"` yields `["LEAD", "I"]`.
fn label_tokens(upper: &str) -> impl Iterator<Item = &str> {
    upper
        .split(|c: char| c == '-' || c == '_' || c == '/' || c.is_whitespace())
        .filter(|t| !t.is_empty())
}

/// Classify an already-uppercased string against the substring markers
/// that are unambiguous ON THEIR OWN — no multiplicity requirement,
/// unlike the bare 12-lead-name path in [`infer_modality`]. Shared by
/// both the per-label pass and the optional `format` hint (a format that
/// deliberately declares its modality, e.g. a DICOM `Modality` attribute
/// of `"ECG"`, is checked with the exact same rule as a channel label).
///
/// Order matters only in that each branch is mutually exclusive by
/// construction (none of these substrings overlap), so it's written as a
/// priority list for readability, not because of a real collision.
fn classify_explicit(upper: &str) -> Option<u8> {
    if upper.contains("EEG") {
        Some(Eeg::TAG)
    } else if upper.contains("ECG") || upper.contains("EKG") {
        Some(Ecg::TAG)
    } else if upper.contains("EMG") {
        Some(Emg::TAG)
    } else if upper.contains("EOG") {
        Some(Eog::TAG)
    } else if upper.contains("RESPIRATION") || upper.contains("RESP") || upper.contains("AIRFLOW") {
        Some(Resp::TAG)
    } else if upper.contains("ACCEL") || upper.contains("ACC") {
        Some(Accel::TAG)
    } else {
        None
    }
}

/// Infer a recording's modality from its channel labels (+ an optional
/// format-declared hint), for [`crate::atoms::LegacyRecording::with_inferred_modality`]
/// — the S3b born-typed-lowering entry point every reader's `lower_to_abir`
/// calls right after building its `LegacyRecording<Untyped>`.
///
/// Returns `(tag, source)`: the inferred [`Modality::TAG`] (or
/// [`Untyped::TAG`] `= 255` when nothing conclusive can be said) and HOW it
/// was decided. Never silent — even the "give up" case is recorded as
/// `(255, ChannelLabel)`, not left as some implicit default.
///
/// ## Rules (case-insensitive)
///
/// 1. **Format-declared hint** (`format`): if the caller passes a format
///    string that itself carries an explicit modality declaration (e.g. a
///    DICOM `(0008,0060) Modality` attribute value of `"ECG"` — see
///    `source/dicom.rs`), that wins outright with
///    [`ModalitySource::FormatDeclared`]. Most current readers don't have
///    such a field (EDF/BDF/BrainVision/CNT/Raw are modality-agnostic
///    containers) and pass `None`.
/// 2. **Explicit per-channel substring markers**: a label containing
///    `"EEG"` → EEG; `"ECG"`/`"EKG"` → ECG; `"EMG"` → EMG; `"EOG"` → EOG;
///    `"RESP"`/`"RESPIRATION"`/`"AIRFLOW"` → Resp; `"ACC"`/`"ACCEL"` →
///    Accel. A single such channel is enough evidence — these markers are
///    unambiguous on their own.
/// 3. **10-20 EEG electrode names** (`Fp1`, `Fz`, `Cz`, `A1`, …), matched
///    per label-token so a bipolar montage label like `"Fp1-F7"` still
///    reads as EEG (both tokens are 10-20 names).
/// 4. **12-lead ECG names** (`I`, `II`, `aVR`, `V1`, …): matched
///    per-token, but a LONE such channel is ambiguous (too short/generic
///    to trust — e.g. a single `"II"` could be an unrelated numbered
///    channel). Only once **two or more** distinct lead-name channels
///    co-occur are they credited as ECG evidence.
///
/// Every channel casts at most one vote (rules applied in the order
/// above, first match wins — they don't overlap in practice). The
/// modality with the most votes wins ONLY if it is a strict majority
/// (`> 50%`) of all channels; a tie, a plurality that doesn't clear 50%,
/// or no votes at all (unclassifiable / generic channel names) all
/// resolve to `Untyped` — conservative by design (see rule 4's rationale:
/// better to under-classify than to silently guess).
///
/// `labels.is_empty()` → `(Untyped::TAG, ChannelLabel)` immediately (no
/// channels, nothing to infer from).
pub fn infer_modality(labels: &[&str], format: Option<&str>) -> (u8, ModalitySource) {
    if let Some(fmt) = format {
        let upper = fmt.to_ascii_uppercase();
        if let Some(tag) = classify_explicit(&upper) {
            return (tag, ModalitySource::FormatDeclared);
        }
    }

    if labels.is_empty() {
        return (Untyped::TAG, ModalitySource::ChannelLabel);
    }

    let mut eeg_votes = 0usize;
    let mut ecg_votes = 0usize;
    let mut emg_votes = 0usize;
    let mut eog_votes = 0usize;
    let mut resp_votes = 0usize;
    let mut accel_votes = 0usize;
    // Tentative — only promoted to `ecg_votes` if >= 2 (see rule 4).
    let mut bare_lead_votes = 0usize;

    for label in labels {
        let upper = label.to_ascii_uppercase();
        if let Some(tag) = classify_explicit(&upper) {
            if tag == Eeg::TAG {
                eeg_votes += 1;
            } else if tag == Ecg::TAG {
                ecg_votes += 1;
            } else if tag == Emg::TAG {
                emg_votes += 1;
            } else if tag == Eog::TAG {
                eog_votes += 1;
            } else if tag == Resp::TAG {
                resp_votes += 1;
            } else if tag == Accel::TAG {
                accel_votes += 1;
            }
            continue;
        }
        if label_tokens(&upper).any(|t| EEG_1020_ELECTRODES.contains(&t)) {
            eeg_votes += 1;
            continue;
        }
        if label_tokens(&upper).any(|t| ECG_12_LEADS.contains(&t)) {
            bare_lead_votes += 1;
            continue;
        }
        // Unclassified: no vote (e.g. "Status", "ch3", a numbered/generic
        // label with no recognizable modality marker).
    }

    if bare_lead_votes >= 2 {
        ecg_votes += bare_lead_votes;
    }

    let total = labels.len();
    let candidates: [(u8, usize); 6] = [
        (Eeg::TAG, eeg_votes),
        (Ecg::TAG, ecg_votes),
        (Emg::TAG, emg_votes),
        (Eog::TAG, eog_votes),
        (Resp::TAG, resp_votes),
        (Accel::TAG, accel_votes),
    ];
    // `max_by_key` picks the LAST maximum on a tie; harmless here because
    // a genuine tie can never individually clear the `> 50%` bar below, so
    // the tied winner is discarded regardless of which one `max_by_key`
    // happened to return.
    let (best_tag, best_votes) = candidates
        .iter()
        .copied()
        .max_by_key(|&(_, v)| v)
        .unwrap_or((Untyped::TAG, 0));

    if best_votes > 0 && best_votes * 2 > total {
        (best_tag, ModalitySource::ChannelLabel)
    } else {
        (Untyped::TAG, ModalitySource::ChannelLabel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marker_tags_and_names() {
        assert_eq!(Untyped::TAG, 255);
        assert_eq!(Untyped::NAME, "untyped");
        assert_eq!(Eeg::TAG, 0);
        assert_eq!(Eeg::NAME, "eeg");
        assert_eq!(Ieeg::TAG, 1);
        assert_eq!(Ecog::TAG, 2);
        assert_eq!(Seeg::TAG, 3);
        assert_eq!(Ecg::TAG, 4);
        assert_eq!(Emg::TAG, 5);
        assert_eq!(Eog::TAG, 6);
        assert_eq!(Resp::TAG, 7);
        assert_eq!(Accel::TAG, 8);
        assert_eq!(Other::TAG, 9);
    }

    #[test]
    fn name_for_tag_round_trips_every_marker() {
        // Single source of truth: name_for_tag(M::TAG) == M::NAME for every
        // marker; an unknown tag is None (never a stale hard-coded name).
        assert_eq!(name_for_tag(Untyped::TAG), Some(Untyped::NAME));
        assert_eq!(name_for_tag(Eeg::TAG), Some(Eeg::NAME));
        assert_eq!(name_for_tag(Ieeg::TAG), Some(Ieeg::NAME));
        assert_eq!(name_for_tag(Ecog::TAG), Some(Ecog::NAME));
        assert_eq!(name_for_tag(Seeg::TAG), Some(Seeg::NAME));
        assert_eq!(name_for_tag(Ecg::TAG), Some(Ecg::NAME));
        assert_eq!(name_for_tag(Emg::TAG), Some(Emg::NAME));
        assert_eq!(name_for_tag(Eog::TAG), Some(Eog::NAME));
        assert_eq!(name_for_tag(Resp::TAG), Some(Resp::NAME));
        assert_eq!(name_for_tag(Accel::TAG), Some(Accel::NAME));
        assert_eq!(name_for_tag(Other::TAG), Some(Other::NAME));
        assert_eq!(name_for_tag(10), None);
        assert_eq!(name_for_tag(254), None);
    }

    #[test]
    fn modality_source_wire_bytes_pinned() {
        // ADR 0069/0071 L9 BCS1 header offset 7 — these values are ON THE
        // WIRE, not an implementation detail; a silent renumbering would
        // desync old vs new BCS1 readers.
        assert_eq!(ModalitySource::ChannelLabel.to_u8(), 0);
        assert_eq!(ModalitySource::FormatDeclared.to_u8(), 1);
        assert_eq!(ModalitySource::Manual.to_u8(), 2);
    }

    #[test]
    fn modality_source_from_u8_round_trips() {
        for s in [
            ModalitySource::ChannelLabel,
            ModalitySource::FormatDeclared,
            ModalitySource::Manual,
        ] {
            assert_eq!(ModalitySource::from_u8(s.to_u8()), Some(s));
        }
    }

    #[test]
    fn modality_source_from_u8_rejects_unknown() {
        assert_eq!(ModalitySource::from_u8(3), None);
        assert_eq!(ModalitySource::from_u8(255), None);
    }

    #[test]
    fn markers_are_zero_sized() {
        assert_eq!(core::mem::size_of::<Eeg>(), 0);
        assert_eq!(core::mem::size_of::<Untyped>(), 0);
    }

    // ─────────────────── S3b: infer_modality ───────────────────

    #[test]
    fn ten_twenty_labels_infer_eeg() {
        let labels = ["Fp1", "Fp2", "F3", "F4", "Cz", "O1", "O2", "A1"];
        let refs: Vec<&str> = labels.to_vec();
        assert_eq!(
            infer_modality(&refs, None),
            (Eeg::TAG, ModalitySource::ChannelLabel)
        );
    }

    #[test]
    fn twelve_lead_labels_infer_ecg() {
        let labels = [
            "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6",
        ];
        let refs: Vec<&str> = labels.to_vec();
        assert_eq!(
            infer_modality(&refs, None),
            (Ecg::TAG, ModalitySource::ChannelLabel)
        );
    }

    #[test]
    fn mixed_eeg_and_ecg_labels_infer_untyped() {
        // 2 clear EEG + 2 clear ECG — no strict majority either way, so
        // this must NOT silently pick a winner.
        let labels = ["Fp1", "Cz", "ECG1", "ECG2"];
        let refs: Vec<&str> = labels.to_vec();
        assert_eq!(
            infer_modality(&refs, None),
            (Untyped::TAG, ModalitySource::ChannelLabel)
        );
    }

    #[test]
    fn empty_labels_infer_untyped() {
        assert_eq!(
            infer_modality(&[], None),
            (Untyped::TAG, ModalitySource::ChannelLabel)
        );
    }

    #[test]
    fn explicit_eeg_substring_wins_regardless_of_case() {
        let labels = ["eeg Fp1-ref", "eeg fp2-ref"];
        let refs: Vec<&str> = labels.to_vec();
        assert_eq!(
            infer_modality(&refs, None),
            (Eeg::TAG, ModalitySource::ChannelLabel)
        );
    }

    #[test]
    fn bipolar_montage_labels_still_read_as_eeg() {
        let labels = ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"];
        let refs: Vec<&str> = labels.to_vec();
        assert_eq!(
            infer_modality(&refs, None),
            (Eeg::TAG, ModalitySource::ChannelLabel)
        );
    }

    #[test]
    fn lone_lead_name_is_ambiguous_not_ecg() {
        // A single "II" among otherwise-generic channels must NOT be
        // enough to call ECG (rule 4 — only >=2 co-occurring lead names
        // count).
        let labels = ["II", "ch1", "ch2"];
        let refs: Vec<&str> = labels.to_vec();
        assert_eq!(
            infer_modality(&refs, None),
            (Untyped::TAG, ModalitySource::ChannelLabel)
        );
    }

    #[test]
    fn two_co_occurring_lead_names_are_credited_as_ecg() {
        let labels = ["II", "V1"];
        let refs: Vec<&str> = labels.to_vec();
        assert_eq!(
            infer_modality(&refs, None),
            (Ecg::TAG, ModalitySource::ChannelLabel)
        );
    }

    #[test]
    fn generic_unclassifiable_labels_infer_untyped() {
        let labels = ["ch0", "ch1", "ch2", "ch3"];
        let refs: Vec<&str> = labels.to_vec();
        assert_eq!(
            infer_modality(&refs, None),
            (Untyped::TAG, ModalitySource::ChannelLabel)
        );
    }

    #[test]
    fn format_declared_modality_wins_with_format_declared_source() {
        // A reader that genuinely knows the modality (e.g. a DICOM
        // `Modality` attribute) should win outright, source-tagged
        // distinctly from a label-based guess.
        let labels = ["Lead I", "Lead J", "Lead K"]; // generic non-standard names
        let refs: Vec<&str> = labels.to_vec();
        assert_eq!(
            infer_modality(&refs, Some("ECG")),
            (Ecg::TAG, ModalitySource::FormatDeclared)
        );
    }

    #[test]
    fn majority_incidental_channel_does_not_flip_majority_modality() {
        // Common real-world pattern: a 10-20 EEG montage plus one
        // incidental EKG reference channel. The EEG majority (8/9) must
        // still win.
        let labels = ["Fp1", "Fp2", "F3", "F4", "Cz", "O1", "O2", "A1", "EKG1"];
        let refs: Vec<&str> = labels.to_vec();
        assert_eq!(
            infer_modality(&refs, None),
            (Eeg::TAG, ModalitySource::ChannelLabel)
        );
    }
}
