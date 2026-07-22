#![cfg_attr(not(feature = "std"), no_std)]
//! Retired ADR 0069 in-process recording IR.
//!
//! This no_std-compatible crate preserves the exact LML1/BCS1 codec seam and
//! historical column representation inside the process-isolated legacy owner.
//! It must not acquire new semantic responsibilities.

extern crate alloc;

/// Retired columnar recording atoms for LML1/BCS1 compatibility.
pub mod atoms;
pub use atoms::{Channel, Column, LegacyRecording};

/// The historical modality typestate retained for legacy readers.
pub mod modality;
pub use modality::{
    name_for_tag, Accel, Ecg, Ecog, Eeg, Emg, Eog, Ieeg, Modality, ModalityProvenance,
    ModalitySource, Other, Resp, Seeg, Untyped, VerifyError,
};

/// The reversibility markers — the `Reversible`/`Lossy` typestate (Pillar 3).
/// The no_std vocabulary; the host `Pass`/`LmlPipeline` machinery that gates on
/// it stays in `lamquant-lossless` (ADR 0074).
pub mod reversibility;
pub use reversibility::{Lossy, Reversibility, Reversible};

/// The BCS1 neutral wire header (ADR 0069/0071 L9) — the ONE deliberate byte
/// change: a 40-byte typed header (born-typed modality + codec descriptor +
/// mode + tier) wrapping the byte-unchanged JSON metadata → window index →
/// LML per-window payloads → `LMLFOOT1` footer. `no_std`-clean by
/// construction (pure `to_le_bytes`/`from_le_bytes`, no I/O).
pub mod bcs1;
pub use bcs1::{
    Bcs1Header, Bcs1ParseError, CodecDescriptor, BCS1_FLAG_HAS_FOOTER, BCS1_HEADER_LEN, BCS1_MAGIC,
    BCS1_VERSION_MAJOR, BCS1_VERSION_MINOR, CODEC_LML_53, CODEC_LMO_97, CODEC_LMO_LOSSLESS,
    CODEC_LMQ_FSQ, CODEC_OPTIMUM_V2,
};
