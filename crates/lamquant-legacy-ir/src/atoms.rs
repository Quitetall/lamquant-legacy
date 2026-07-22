//! Retired ADR 0069 columnar recording atoms retained for wire compatibility.
//!
//! An [`LegacyRecording`] is a channel×time recording stored **column-major**: each channel is one
//! contiguous, immutable, `Arc`-shared buffer at its NATIVE width (`i16` / 24-bit-in-`i32` /
//! `i32` / `i64` / `f32`), not always `i64`. A window is an O(1) sub-slice: on the `i64` lane
//! [`Column::window_i64`] returns `Cow::Borrowed` — true zero-copy, the direct cure for the
//! per-window `Vec<Vec<i64>>` memcpy the v1 container did (`legacy container.rs:417`). Narrower
//! lanes widen the window ONCE into an owned buffer at the kernel boundary, byte-exact for the
//! integer lanes, so the kernel's packet output is unchanged.
//!
//! `no_std` + `alloc`. The LML kernels consume `&[i64]`, so `i64` is the codec working
//! currency; narrow storage is a memory/bandwidth win that widens only at the seam.
//!
//! [`LegacyRecording`] is generic over a modality marker `M` (default [`Untyped`] —
//! `LegacyRecording` == `LegacyRecording<Untyped>`, source-compatible with every pre-S3a call
//! site). See [`crate::modality`] for the trust model this typestate
//! implements.

use alloc::borrow::Cow;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::modality::{Modality, ModalityProvenance, ModalitySource, Untyped, VerifyError};

/// Physical storage width of one channel column. Narrow widths cut memory + cache pressure
/// for natively-16/24-bit biosignals; `I64` is the lane the LML kernels read with zero copy.
#[derive(Clone, Debug)]
pub enum Column {
    /// 16-bit integer samples (EDF, most EEG AFEs).
    I16(Arc<[i16]>),
    /// 24-bit integer samples carried in `i32` lanes (BDF / 24-bit ADCs).
    I24(Arc<[i32]>),
    /// 32-bit integer samples.
    I32(Arc<[i32]>),
    /// 64-bit integer samples — the codec working currency; windowed zero-copy.
    I64(Arc<[i64]>),
    /// 32-bit float samples (float iEEG / derived signals). `window_i64` casts toward zero —
    /// meaningful only for integer-valued floats; the lossless LML path is fed via `I64`.
    F32(Arc<[f32]>),
}

impl Column {
    /// Number of samples in the column.
    pub fn len(&self) -> usize {
        match self {
            Column::I16(a) => a.len(),
            Column::I24(a) | Column::I32(a) => a.len(),
            Column::I64(a) => a.len(),
            Column::F32(a) => a.len(),
        }
    }

    /// True if the column has no samples.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A windowed `[start, end)` view as `&[i64]`. **Zero-copy iff the column is already
    /// `I64`** (`Cow::Borrowed` of the shared buffer — the memcpy cure); narrow/float lanes
    /// widen the window once into an owned buffer (byte-exact for the integer lanes).
    ///
    /// Panics on an out-of-range window, matching the v1 encoder's `ch[start..end]`.
    pub fn window_i64(&self, start: usize, end: usize) -> Cow<'_, [i64]> {
        match self {
            Column::I64(a) => Cow::Borrowed(&a[start..end]),
            Column::I16(a) => Cow::Owned(a[start..end].iter().map(|&v| v as i64).collect()),
            Column::I24(a) | Column::I32(a) => {
                Cow::Owned(a[start..end].iter().map(|&v| v as i64).collect())
            }
            Column::F32(a) => Cow::Owned(a[start..end].iter().map(|&v| v as i64).collect()),
        }
    }
}

/// One channel: a width-typed column + its label and physical-range calibration (preserved
/// for reconstruction; NOT part of the LML packet bytes).
#[derive(Clone, Debug)]
pub struct Channel {
    /// Channel label (e.g. `"Fp1"`), shared.
    pub label: Arc<str>,
    /// The sample column at its native width.
    pub data: Column,
    /// Physical minimum (calibration).
    pub phys_min: f64,
    /// Physical maximum (calibration).
    pub phys_max: f64,
}

/// The retired codec recording currency: channel×time, column-major, with `Arc`-shared
/// immutable per-channel buffers so a window is an O(1) sub-slice. Channel ORDER is load-bearing
/// (the codec + wire preserve it).
///
/// Generic over a modality marker `M` (default [`Untyped`], the S2/L6 behavior — `LegacyRecording` ==
/// `LegacyRecording<Untyped>`). This is the ADR 0069 S3a typestate: `LegacyRecording<Eeg>` and `LegacyRecording<Ecg>` are distinct
/// types, so a modality-typed consumer (`fn train(&LegacyRecording<Eeg>)`) cannot be called with the wrong
/// modality — see [`crate::modality`] for the full trust-model rationale.
#[derive(Clone, Debug)]
pub struct LegacyRecording<M: Modality = Untyped> {
    /// Per-channel columns, in wire/channel order.
    pub channels: Vec<Channel>,
    /// Sampling rate in Hz.
    pub sample_rate: f64,
    /// Samples per channel (all channels share this length).
    pub n_samples: usize,
    /// How/whether this `LegacyRecording`'s modality was assigned.
    pub prov: ModalityProvenance,
    _m: PhantomData<M>,
}

impl<M: Modality> LegacyRecording<M> {
    /// Per-window `&[i64]` views, one per channel, for `[start, end)`. On an all-`I64` `LegacyRecording`
    /// every entry is `Cow::Borrowed` → no allocation (the killed per-window memcpy).
    ///
    /// The sanctioned modality-agnostic ENCODER egress — the codec is modality-blind (bytes in,
    /// bytes out). SEMANTIC access (training, cross-modal analysis) goes through modality-typed
    /// consumers (`fn train(&LegacyRecording<Eeg>)`) + [`verify`](LegacyRecording::verify), which is where
    /// Pillar 1 (no cross-modality mixing) is enforced.
    pub fn window_views(&self, start: usize, end: usize) -> Vec<Cow<'_, [i64]>> {
        self.channels
            .iter()
            .map(|c| c.data.window_i64(start, end))
            .collect()
    }

    /// Number of channels.
    pub fn n_channels(&self) -> usize {
        self.channels.len()
    }

    /// Samples per channel (accessor mirroring the public `n_samples` field).
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// This `LegacyRecording<M>`'s modality name (`M::NAME`).
    pub fn modality_name(&self) -> &'static str {
        M::NAME
    }

    /// The recorded modality provenance (how/whether `M` was assigned).
    pub fn provenance(&self) -> &ModalityProvenance {
        &self.prov
    }

    /// Boundary invariant check for a modality-typed consumer: every channel's column length
    /// matches `n_samples`, and the recorded provenance tag agrees with the type-level modality
    /// `M::TAG`. This — not the encoder egress (`window_views`) — is where Pillar 1 is actually
    /// enforced: a typed consumer calls `verify()` before trusting the data as modality `M`.
    pub fn verify(&self) -> Result<(), VerifyError> {
        for (i, ch) in self.channels.iter().enumerate() {
            let actual = ch.data.len();
            if actual != self.n_samples {
                return Err(VerifyError::ChannelLengthMismatch {
                    channel: i,
                    actual,
                    expected: self.n_samples,
                });
            }
        }
        if self.prov.tag != M::TAG {
            return Err(VerifyError::ProvenanceTagMismatch {
                recorded: self.prov.tag,
                expected: M::TAG,
            });
        }
        Ok(())
    }
}

impl LegacyRecording<Untyped> {
    /// Bridge from today's `Vec<Vec<i64>>` currency — each channel becomes an `I64` column via
    /// `Arc::from` (moves the `Vec`, no element copy). Labels/phys are left empty/zero; L7
    /// readers populate them. `n_samples` is taken from channel 0 (0 if no channels).
    ///
    /// Provenance is `{source: Manual, tag: 255}` (untyped) — matches [`Untyped::TAG`].
    pub fn from_channels_i64(signal: Vec<Vec<i64>>, sample_rate: f64) -> Self {
        let n_samples = signal.first().map(|c| c.len()).unwrap_or(0);
        let channels = signal
            .into_iter()
            .map(|ch| Channel {
                label: Arc::from(""),
                data: Column::I64(Arc::from(ch)),
                phys_min: 0.0,
                phys_max: 0.0,
            })
            .collect();
        LegacyRecording {
            channels,
            sample_rate,
            n_samples,
            prov: ModalityProvenance {
                source: ModalitySource::Manual,
                tag: Untyped::TAG,
            },
            _m: PhantomData,
        }
    }

    /// Construct directly from already-built [`Channel`]s — the path format readers that decode
    /// straight into a native-width `Column` use (EDF, BrainVision, CNT, DICOM waveform, raw)
    /// instead of routing through [`from_channels_i64`](LegacyRecording::from_channels_i64). Same provenance
    /// convention: `{source: Manual, tag: 255}` (untyped).
    pub fn from_parts(channels: Vec<Channel>, sample_rate: f64, n_samples: usize) -> Self {
        LegacyRecording {
            channels,
            sample_rate,
            n_samples,
            prov: ModalityProvenance {
                source: ModalitySource::Manual,
                tag: Untyped::TAG,
            },
            _m: PhantomData,
        }
    }

    /// Born-typed lowering (ADR 0069 S3b): infer this `LegacyRecording`'s modality from
    /// its channel labels (+ an optional format-declared hint, see
    /// [`crate::modality::infer_modality`]) and overwrite `prov` with the
    /// result. Every reader's `lower_to_abir` calls this right after
    /// building its `LegacyRecording<Untyped>` (via [`from_channels_i64`](LegacyRecording::from_channels_i64)
    /// or [`from_parts`](LegacyRecording::from_parts), both of which start `prov` at
    /// `{Manual, 255}`) so lowering never leaves a stream silently
    /// "untyped by default" — the provenance always records SOMETHING,
    /// even when that something is "inference gave up: `{ChannelLabel, 255}`".
    ///
    /// `self` stays `LegacyRecording<Untyped>` — this only updates the runtime `prov`
    /// bookkeeping, not the type. Asserting the type is the separate,
    /// explicit [`LegacyRecording::try_into_modality`] step, which CHECKS the recorded
    /// inference rather than overriding it.
    ///
    /// `labels` is taken as an explicit parameter (rather than read back off
    /// `self.channels`) so a reader can pass its native label list (e.g. a
    /// `SignalBundle`'s `channels: Vec<String>`, or a format header's own
    /// channel-name array) without an extra borrow-then-collect round trip;
    /// in every current caller it is exactly `self.channels[..].label`.
    pub fn with_inferred_modality(mut self, labels: &[&str], format: Option<&str>) -> Self {
        let (tag, source) = crate::modality::infer_modality(labels, format);
        self.prov = ModalityProvenance { source, tag };
        self
    }

    /// One-time deliberate narrowing: promote an untyped `LegacyRecording` to a modality-typed one.
    /// Re-wraps the SAME `channels`/`sample_rate`/`n_samples` (a field move — `PhantomData` is
    /// zero-cost) under `M`, recording how the assignment was decided.
    ///
    /// # The compile-time trust guarantee
    ///
    /// A modality-typed consumer only accepts its own modality — passing the wrong one is a
    /// **type error**, caught at compile time, not a runtime bug:
    ///
    /// ```compile_fail
    /// use lamquant_legacy_ir::{LegacyRecording, Ecg, Eeg, ModalitySource};
    ///
    /// fn train(_abir: &LegacyRecording<Eeg>) {}
    ///
    /// let raw = LegacyRecording::from_channels_i64(vec![vec![1i64, 2, 3]], 250.0);
    /// let ecg: LegacyRecording<Ecg> = raw.into_modality(ModalitySource::Manual);
    /// train(&ecg); // ← does NOT compile: expected `&LegacyRecording<Eeg>`, found `&LegacyRecording<Ecg>`
    /// ```
    ///
    /// The matching modality compiles and runs cleanly:
    ///
    /// ```
    /// use lamquant_legacy_ir::{LegacyRecording, Eeg, ModalitySource};
    ///
    /// fn train(_abir: &LegacyRecording<Eeg>) {}
    ///
    /// let raw = LegacyRecording::from_channels_i64(vec![vec![1i64, 2, 3]], 250.0);
    /// let eeg: LegacyRecording<Eeg> = raw.into_modality(ModalitySource::Manual);
    /// train(&eeg);
    /// ```
    pub fn into_modality<M: Modality>(self, source: ModalitySource) -> LegacyRecording<M> {
        LegacyRecording {
            channels: self.channels,
            sample_rate: self.sample_rate,
            n_samples: self.n_samples,
            prov: ModalityProvenance {
                source,
                tag: M::TAG,
            },
            _m: PhantomData,
        }
    }

    /// Stamp an authoritative `Manual` modality TAG on the wire provenance WITHOUT
    /// changing the compile-time modality type (ADR 0074 Track I).
    ///
    /// This is for the modality-blind ENCODER path: the CLI writes bytes, and the
    /// born-typed BCS1 header (byte 6 = this tag, byte 7 = `Manual`) is what a typed
    /// reader (`PyAbir::eeg()`, training) re-verifies at its boundary. The encoder's
    /// own compile-time `M` is irrelevant there — the `LegacyRecording` is serialized, never
    /// consumed as `LegacyRecording<M>` — so a runtime tag is the honest, minimal thing.
    /// Prefer [`into_modality`](LegacyRecording::into_modality) wherever the compile-time TYPE
    /// carries the modality (typed consumers). Byte-neutral: only the
    /// header provenance byte changes (proven by `modality_provenance_snapshot`).
    pub fn with_manual_modality_tag(mut self, tag: u8) -> Self {
        self.prov = ModalityProvenance {
            source: ModalitySource::Manual,
            tag,
        };
        self
    }

    /// VERIFIED promotion (ADR 0069 S3b): promote to `LegacyRecording<M>` only if the
    /// RECORDED inference (`self.prov.tag`, set by
    /// [`with_inferred_modality`](LegacyRecording::with_inferred_modality) or a prior
    /// narrowing) agrees with the asserted type `M`. Preserves the recorded
    /// `prov.source` (it doesn't become "more Manual" just because the
    /// caller asserted a type that matched).
    ///
    /// This is the CHECKED counterpart to [`into_modality`](LegacyRecording::into_modality):
    /// `into_modality` is a deliberate, unconditional override (the caller
    /// is asserting "trust me, this is `M`"); `try_into_modality` is a type
    /// assertion against what the codec itself already inferred, and it
    /// REFUSES to silently mislabel a mismatch — on failure, `self` is
    /// handed back untouched (still `LegacyRecording<Untyped>`, nothing consumed)
    /// alongside a [`VerifyError::ProvenanceTagMismatch`] explaining what
    /// was recorded vs. what was asked for.
    ///
    /// ```
    /// use lamquant_legacy_ir::{LegacyRecording, Ecg, Eeg, ModalitySource};
    ///
    /// let untyped = LegacyRecording::from_channels_i64(vec![vec![1i64, 2, 3]], 250.0)
    ///     .with_inferred_modality(&["Fp1", "Fp2", "Cz"], None);
    ///
    /// // Matches the recorded inference: succeeds.
    /// let eeg: LegacyRecording<Eeg> = untyped.clone().try_into_modality().unwrap();
    /// assert_eq!(eeg.provenance().source, ModalitySource::ChannelLabel);
    ///
    /// // Doesn't match: refused, `self` handed back.
    /// let (back, _err) = untyped.try_into_modality::<Ecg>().unwrap_err();
    /// assert_eq!(back.n_channels(), 1);
    /// ```
    pub fn try_into_modality<M: Modality>(
        self,
    ) -> Result<LegacyRecording<M>, (LegacyRecording<Untyped>, VerifyError)> {
        if self.prov.tag != M::TAG {
            let err = VerifyError::ProvenanceTagMismatch {
                recorded: self.prov.tag,
                expected: M::TAG,
            };
            return Err((self, err));
        }
        let source = self.prov.source;
        Ok(LegacyRecording {
            channels: self.channels,
            sample_rate: self.sample_rate,
            n_samples: self.n_samples,
            prov: ModalityProvenance {
                source,
                tag: M::TAG,
            },
            _m: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i64_window_is_borrowed_zero_copy() {
        let col = Column::I64(Arc::from(vec![10i64, 20, 30, 40, 50]));
        let w = col.window_i64(1, 4);
        assert!(matches!(w, Cow::Borrowed(_)), "i64 lane must be zero-copy");
        assert_eq!(&*w, &[20, 30, 40]);
    }

    #[test]
    fn narrow_lanes_widen_exactly() {
        let i16c = Column::I16(Arc::from(vec![-3i16, 0, 7, 32000]));
        assert!(matches!(i16c.window_i64(0, 4), Cow::Owned(_)));
        assert_eq!(&*i16c.window_i64(0, 4), &[-3i64, 0, 7, 32000]);

        let i32c = Column::I32(Arc::from(vec![-8_000_000i32, 8_388_607]));
        assert_eq!(&*i32c.window_i64(0, 2), &[-8_000_000i64, 8_388_607]);
    }

    #[test]
    fn from_channels_i64_preserves_signal_and_shape() {
        let sig = vec![vec![1i64, 2, 3, 4], vec![5, 6, 7, 8]];
        let abir = LegacyRecording::from_channels_i64(sig.clone(), 250.0);
        assert_eq!(abir.n_channels(), 2);
        assert_eq!(abir.n_samples, 4);
        assert_eq!(abir.sample_rate, 250.0);
        // Full-length views reconstruct the input exactly, channel by channel.
        let views = abir.window_views(0, 4);
        assert!(views.iter().all(|v| matches!(v, Cow::Borrowed(_))));
        for (v, orig) in views.iter().zip(sig.iter()) {
            assert_eq!(v.as_ref(), orig.as_slice());
        }
        // A sub-window slices both channels identically to the source.
        let mid = abir.window_views(1, 3);
        assert_eq!(mid[0].as_ref(), &[2, 3]);
        assert_eq!(mid[1].as_ref(), &[6, 7]);
    }

    #[test]
    fn empty_signal_has_zero_samples() {
        let abir = LegacyRecording::from_channels_i64(Vec::new(), 250.0);
        assert_eq!(abir.n_channels(), 0);
        assert_eq!(abir.n_samples, 0);
    }

    // ─────────────────────── S3a: the modality trust model ───────────────────────

    use crate::modality::{Eeg, Untyped};

    #[test]
    fn into_modality_round_trips_channels_and_records_provenance() {
        let sig = vec![vec![1i64, 2, 3, 4], vec![5, 6, 7, 8]];
        let untyped = LegacyRecording::from_channels_i64(sig.clone(), 250.0);
        let eeg: LegacyRecording<Eeg> = untyped.into_modality(ModalitySource::Manual);

        // Channels/shape carried through the narrowing untouched.
        assert_eq!(eeg.n_channels(), 2);
        assert_eq!(eeg.n_samples(), 4);
        assert_eq!(eeg.sample_rate, 250.0);
        let views = eeg.window_views(0, 4);
        for (v, orig) in views.iter().zip(sig.iter()) {
            assert_eq!(v.as_ref(), orig.as_slice());
        }

        // Provenance recorded correctly.
        assert_eq!(eeg.modality_name(), "eeg");
        assert_eq!(eeg.provenance().tag, Eeg::TAG);
        assert_eq!(eeg.provenance().source, ModalitySource::Manual);

        // A freshly-narrowed LegacyRecording passes its own boundary check.
        assert!(eeg.verify().is_ok());
    }

    #[test]
    fn verify_detects_channel_length_mismatch() {
        let abir = LegacyRecording::from_parts(
            vec![Channel {
                label: Arc::from(""),
                data: Column::I64(Arc::from(vec![1i64, 2, 3])),
                phys_min: 0.0,
                phys_max: 0.0,
            }],
            250.0,
            99, // deliberately wrong vs. the 3-sample column above
        );
        assert!(matches!(
            abir.verify(),
            Err(VerifyError::ChannelLengthMismatch {
                channel: 0,
                actual: 3,
                expected: 99
            })
        ));
    }

    #[test]
    fn verify_detects_provenance_tag_mismatch() {
        let untyped = LegacyRecording::from_channels_i64(vec![vec![1i64, 2, 3]], 250.0);
        let mut eeg: LegacyRecording<Eeg> = untyped.into_modality(ModalitySource::Manual);
        eeg.prov.tag = 123; // corrupt the provenance independent of the type
        assert!(matches!(
            eeg.verify(),
            Err(VerifyError::ProvenanceTagMismatch {
                recorded: 123,
                expected: 0
            })
        ));
    }

    #[test]
    fn abir_typestate_is_zero_cost() {
        // The core proof: PhantomData<M> adds nothing — LegacyRecording<Eeg> and
        // LegacyRecording<Untyped> occupy identical memory, so the trust model is a
        // compile-time-only distinction (Bible: zero-cost abstractions).
        assert_eq!(
            core::mem::size_of::<LegacyRecording<Eeg>>(),
            core::mem::size_of::<LegacyRecording<Untyped>>()
        );
    }

    // ─────────────────── S3b: born-typed lowering ───────────────────

    use crate::modality::Ecg;

    #[test]
    fn with_inferred_modality_records_eeg_from_labels() {
        let untyped =
            LegacyRecording::from_channels_i64(vec![vec![1i64, 2, 3], vec![4, 5, 6]], 250.0)
                .with_inferred_modality(&["Fp1", "Fp2"], None);
        assert_eq!(untyped.provenance().tag, Eeg::TAG);
        assert_eq!(untyped.provenance().source, ModalitySource::ChannelLabel);
    }

    #[test]
    fn with_inferred_modality_never_leaves_manual_default() {
        // Even an inconclusive recording (unrecognizable labels) must be
        // recorded as an explicit ChannelLabel-sourced Untyped verdict,
        // NOT silently left at from_channels_i64's {Manual, 255} default.
        let untyped = LegacyRecording::from_channels_i64(vec![vec![1i64, 2, 3]], 250.0)
            .with_inferred_modality(&["ch0"], None);
        assert_eq!(untyped.provenance().tag, Untyped::TAG);
        assert_eq!(untyped.provenance().source, ModalitySource::ChannelLabel);
    }

    #[test]
    fn try_into_modality_succeeds_when_inference_matches_asserted_type() {
        let untyped = LegacyRecording::from_channels_i64(vec![vec![1i64, 2, 3, 4]], 250.0)
            .with_inferred_modality(&["Fp1"], None);
        let eeg: LegacyRecording<Eeg> = untyped.try_into_modality().expect("Eeg matches inference");
        assert_eq!(eeg.provenance().tag, Eeg::TAG);
        assert_eq!(eeg.provenance().source, ModalitySource::ChannelLabel);
        assert!(eeg.verify().is_ok());
    }

    #[test]
    fn try_into_modality_rejects_mismatched_asserted_type_and_returns_self() {
        let untyped = LegacyRecording::from_channels_i64(vec![vec![1i64, 2, 3, 4]], 250.0)
            .with_inferred_modality(&["Fp1"], None);
        let (back, err) = untyped
            .try_into_modality::<Ecg>()
            .expect_err("Ecg does not match the recorded Eeg inference");
        assert!(matches!(
            err,
            VerifyError::ProvenanceTagMismatch {
                recorded: 0,
                expected: 4,
            }
        ));
        // `self` is handed back untouched — nothing consumed on failure.
        assert_eq!(back.n_channels(), 1);
        assert_eq!(back.provenance().tag, Eeg::TAG);
    }

    #[test]
    fn try_into_modality_preserves_recorded_source_on_success() {
        let untyped = LegacyRecording::from_channels_i64(vec![vec![1i64, 2, 3]], 250.0)
            .with_inferred_modality(&[], Some("ECG"));
        let ecg: LegacyRecording<Ecg> = untyped.try_into_modality().expect("format-declared ECG");
        assert_eq!(ecg.provenance().source, ModalitySource::FormatDeclared);
    }
}
