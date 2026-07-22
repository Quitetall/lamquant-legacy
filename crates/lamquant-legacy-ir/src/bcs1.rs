//! BCS1 — the ABIR neutral wire header (ADR 0069/0071, step L9).
//!
//! **The ONE deliberate byte change.** Every prior ABIR step (S1–S3, L1–L8)
//! was byte-identity-preserving by construction — `write_legacy_recording` reproduced the
//! legacy `LML1` 32-byte header verbatim. L9 is the first step that changes
//! the wire on purpose: it wraps the SAME byte-unchanged payload (JSON
//! metadata → window index → per-window `LML1` packets → `LMLFOOT1` footer)
//! in a new 40-byte **typed** header carrying the born-typed modality
//! (Pillar 1, S3a/S3b) and the codec/mode/tier descriptors the old header had
//! no room for.
//!
//! `no_std` + `alloc` — this type is pure `to_le_bytes`/`from_le_bytes`
//! plumbing, no I/O, so it is usable from the firmware decoder as well as the
//! host writer.
//!
//! ## Layout — exactly 40 bytes, little-endian
//!
//! ```text
//! offset  size  field
//! 0       4     magic = b"BCS1"
//! 4       1     version_major (=1)
//! 5       1     version_minor (=0)
//! 6       1     modality_tag        (recording.prov.tag — Modality::TAG)
//! 7       1     modality_source     (ModalitySource::to_u8)
//! 8       1     codec_descriptor    (CodecDescriptor::to_u8 — =0 CODEC_LML_53 today)
//! 9       1     mode                (0=Lossless, 1=BoundedMae, 2=TargetBps)
//! 10      1     tier                (descriptive, non-gating: LosslessMode)
//! 11      1     decode_capability   (the GATE; =0 for the integer floor)
//! 12      2     n_channels          (u16 LE)
//! 14      2     n_windows           (u16 LE)
//! 16      4     total_samples       (u32 LE)
//! 20      2     window_size         (u16 LE)
//! 22      4     sample_rate_mhz     (u32 LE)
//! 26      1     bit_depth (=16)
//! 27      1     flags (FLAG_HAS_FOOTER)
//! 28      4     metadata_length     (u32 LE)
//! 32      4     reserved_0 (=0)
//! 36      4     reserved_1 (=0)
//! ```
//!
//! Everything at offset >= 40 (metadata JSON, window index, per-window `LML1`
//! packets, `LMLFOOT1` footer) is byte-IDENTICAL to the pre-L9 wire — only
//! `first_payload_abs`'s base shifts from 32 to 40 (see
//! `lamquant-lossless::legacy_bcs1_container::write_legacy_recording`).

use core::fmt;

/// The BCS1 magic — 4 ASCII bytes, distinct from every legacy magic
/// (`LML1`/`LMO1`/`LMA1`/`LMA2`/`LMQC`/`LMLCRYPT`) so the facade dispatcher
/// (`legacy_bcs1_container::{read_file,read_bytes}`) can tell them apart from byte 0.
pub const BCS1_MAGIC: &[u8; 4] = b"BCS1";

/// Fixed BCS1 header length in bytes.
pub const BCS1_HEADER_LEN: usize = 40;

/// BCS1 header version (bumped only on a wire-format change to the header
/// itself — independent of the inner `LML1` packet version).
pub const BCS1_VERSION_MAJOR: u8 = 1;
pub const BCS1_VERSION_MINOR: u8 = 0;

/// Header flag bit 0 — the file carries an `LMLFOOT1` seek table at EOF
/// (same semantics as the legacy `FLAG_HAS_FOOTER`, cloned here so this
/// module stays self-contained).
pub const BCS1_FLAG_HAS_FOOTER: u8 = 0b0000_0001;

/// `codec_descriptor` = the inner payload is an LML integer 5/3 stream (the
/// lossless floor). Legacy values 0..=2 mirror the original LMO transform IDs.
pub const CODEC_LML_53: u8 = 0;
/// `codec_descriptor` = an LMO-native 9/7 float PCRD body (lossy). Mirrors
/// LMO's `transform_id=1`. Not wired into the BCS1 read dispatch yet — a
/// deferred, tracked follow-up (LMO/LMQ descriptors are out of L9 scope).
pub const CODEC_LMO_97: u8 = 1;
/// `codec_descriptor` = the Optimum LOSSLESS body (cross-channel + `lml`).
/// Mirrors LMO's `transform_id=2`. Deferred — see [`CODEC_LMO_97`].
pub const CODEC_LMO_LOSSLESS: u8 = 2;
/// `codec_descriptor` = LamQuant Optimum v2 `LMO1` v3 / `BGF1` body.
/// Descriptor 3 is version-qualified and does NOT alias legacy LMO v2
/// `transform_id=3` (bounded MV-RLS); dispatch must inspect the LMO version.
pub const CODEC_OPTIMUM_V2: u8 = 3;
/// `codec_descriptor` = the LMQ neural body: FSQ tokens entropy-coded with rANS
/// (ADR 0074 Track N). The first member of the `0x10..=0xFF` LMQ/neural family.
/// PERMANENTLY LOSSY: a BCS1 file carrying this descriptor is refused fail-closed
/// by every lossless reader (`bcs1_gate_decodable` accepts only `CODEC_LML_53`),
/// so it can never be silently mis-decoded as integer samples.
pub const CODEC_LMQ_FSQ: u8 = 0x10;
// `0x11..=0xFF` stays reserved for the rest of the LMQ/neural descriptor family
// (future body variants). An unrecognized `codec_descriptor` byte in that range
// parses fine (the header is well-formed) but `CodecDescriptor::from_u8` returns
// `None`, and the reader refuses to DECODE it (still no silent mis-decode).

/// Which body format `codec_descriptor` names. Values 0..=2 retain their legacy
/// LMO mapping. New descriptors are independent BCS names, not transform IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecDescriptor {
    /// `=0`. The LML integer 5/3 lifting floor — the only descriptor the L9
    /// writer/reader emits/decodes today.
    Lml53,
    /// `=1`. LMO-native 9/7 float PCRD body (lossy). Parseable, not yet
    /// decodable via the BCS1 dispatch (deferred).
    Lmo97,
    /// `=2`. Optimum lossless body (cross-channel + `lml`). Parseable, not
    /// yet decodable via the BCS1 dispatch (deferred).
    LmoLossless,
    /// `=3`. Optimum v2 `LMO1` v3 / `BGF1`; host-only and not yet wired into
    /// the BCS container reader.
    OptimumV2,
    /// `=0x10`. The LMQ neural body (FSQ tokens → rANS, ADR 0074 Track N).
    /// Recognized so the reader can name + refuse it fail-closed; permanently
    /// lossy, never decoded by a lossless reader.
    LmqFsq,
}

impl CodecDescriptor {
    /// The wire byte for this descriptor.
    pub const fn to_u8(self) -> u8 {
        match self {
            Self::Lml53 => CODEC_LML_53,
            Self::Lmo97 => CODEC_LMO_97,
            Self::LmoLossless => CODEC_LMO_LOSSLESS,
            Self::OptimumV2 => CODEC_OPTIMUM_V2,
            Self::LmqFsq => CODEC_LMQ_FSQ,
        }
    }

    /// Parse a wire byte into a known descriptor. `None` for anything
    /// unrecognized (the still-reserved `0x11+` LMQ range) — callers must treat
    /// `None` as "cannot decode this body", never as CODEC_LML_53. Note a
    /// recognized `LmqFsq` is still refused by `bcs1_gate_decodable` (lossy).
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            CODEC_LML_53 => Some(Self::Lml53),
            CODEC_LMO_97 => Some(Self::Lmo97),
            CODEC_LMO_LOSSLESS => Some(Self::LmoLossless),
            CODEC_OPTIMUM_V2 => Some(Self::OptimumV2),
            CODEC_LMQ_FSQ => Some(Self::LmqFsq),
            _ => None,
        }
    }
}

/// The parsed/pre-built 40-byte BCS1 header. Pure data — construction and
/// `to_bytes`/`parse` never touch I/O, so this is usable from a `no_std`
/// firmware decoder as well as the host writer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bcs1Header {
    pub version_major: u8,
    pub version_minor: u8,
    /// `recording.prov.tag` — the [`Modality::TAG`](crate::Modality::TAG) this
    /// stream was born-typed as (or `Untyped::TAG` = 255).
    pub modality_tag: u8,
    /// [`crate::ModalitySource`] wire byte for how `modality_tag` was
    /// decided.
    pub modality_source: u8,
    /// [`CodecDescriptor`] wire byte for the inner payload body.
    pub codec_descriptor: u8,
    /// Codec operation mode: `0` = Lossless, `1` = BoundedMae, `2` =
    /// TargetBps. Mirrors `metadata_with_codec_mode`'s own precedence
    /// (target_bps wins over delta wins over lossless).
    pub mode: u8,
    /// Descriptive, NON-GATING deployment tier (today: `lossless_mode_for_lpc_mode`
    /// — `0` = Mcu, `1` = Basestation). A reader must never refuse to decode
    /// based on this byte; it is provenance, not a capability gate.
    pub tier: u8,
    /// The decode-capability GATE: the minimum reader capability required to
    /// decode this payload. `0` = the integer floor (every BCS1 reader, MCU
    /// included, can decode it) — the only value the L9 writer emits.
    pub decode_capability: u8,
    pub n_channels: u16,
    pub n_windows: u16,
    pub total_samples: u32,
    pub window_size: u16,
    pub sample_rate_mhz: u32,
    pub bit_depth: u8,
    pub flags: u8,
    pub metadata_length: u32,
}

impl Bcs1Header {
    /// Serialize to the exact 40-byte wire layout (see module docs). Pure
    /// `to_le_bytes` — no I/O, no allocation.
    pub fn to_bytes(&self) -> [u8; BCS1_HEADER_LEN] {
        let mut out = [0u8; BCS1_HEADER_LEN];
        out[0..4].copy_from_slice(BCS1_MAGIC);
        out[4] = self.version_major;
        out[5] = self.version_minor;
        out[6] = self.modality_tag;
        out[7] = self.modality_source;
        out[8] = self.codec_descriptor;
        out[9] = self.mode;
        out[10] = self.tier;
        out[11] = self.decode_capability;
        out[12..14].copy_from_slice(&self.n_channels.to_le_bytes());
        out[14..16].copy_from_slice(&self.n_windows.to_le_bytes());
        out[16..20].copy_from_slice(&self.total_samples.to_le_bytes());
        out[20..22].copy_from_slice(&self.window_size.to_le_bytes());
        out[22..26].copy_from_slice(&self.sample_rate_mhz.to_le_bytes());
        out[26] = self.bit_depth;
        out[27] = self.flags;
        out[28..32].copy_from_slice(&self.metadata_length.to_le_bytes());
        out[32..36].copy_from_slice(&0u32.to_le_bytes()); // reserved_0
        out[36..40].copy_from_slice(&0u32.to_le_bytes()); // reserved_1
        out
    }

    /// Parse a 40-byte BCS1 header off the front of `data`. `data` may be
    /// longer (the full container) — only the first 40 bytes are consulted.
    pub fn parse(data: &[u8]) -> Result<Self, Bcs1ParseError> {
        if data.len() < BCS1_HEADER_LEN {
            return Err(Bcs1ParseError::Truncated {
                expected: BCS1_HEADER_LEN,
                actual: data.len(),
            });
        }
        if &data[0..4] != BCS1_MAGIC {
            return Err(Bcs1ParseError::InvalidMagic([
                data[0], data[1], data[2], data[3],
            ]));
        }
        Ok(Bcs1Header {
            version_major: data[4],
            version_minor: data[5],
            modality_tag: data[6],
            modality_source: data[7],
            codec_descriptor: data[8],
            mode: data[9],
            tier: data[10],
            decode_capability: data[11],
            n_channels: u16::from_le_bytes([data[12], data[13]]),
            n_windows: u16::from_le_bytes([data[14], data[15]]),
            total_samples: u32::from_le_bytes([data[16], data[17], data[18], data[19]]),
            window_size: u16::from_le_bytes([data[20], data[21]]),
            sample_rate_mhz: u32::from_le_bytes([data[22], data[23], data[24], data[25]]),
            bit_depth: data[26],
            flags: data[27],
            metadata_length: u32::from_le_bytes([data[28], data[29], data[30], data[31]]),
        })
    }
}

/// A BCS1 header parse failure. `no_std`-safe (carries only `Copy` fields,
/// no allocation) — mirrors [`crate::modality::VerifyError`]'s shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bcs1ParseError {
    /// Fewer than [`BCS1_HEADER_LEN`] bytes available.
    Truncated { expected: usize, actual: usize },
    /// The leading 4 bytes are not `b"BCS1"`.
    InvalidMagic([u8; 4]),
}

impl fmt::Display for Bcs1ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Bcs1ParseError::Truncated { expected, actual } => write!(
                f,
                "BCS1 header truncated: expected {expected} bytes, got {actual}"
            ),
            Bcs1ParseError::InvalidMagic(m) => {
                write!(f, "BCS1 header: invalid magic {:?}", m)
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Bcs1ParseError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_header() -> Bcs1Header {
        Bcs1Header {
            version_major: BCS1_VERSION_MAJOR,
            version_minor: BCS1_VERSION_MINOR,
            modality_tag: 0,
            modality_source: 0,
            codec_descriptor: CodecDescriptor::Lml53.to_u8(),
            mode: 0,
            tier: 1,
            decode_capability: 0,
            n_channels: 4,
            n_windows: 7,
            total_samples: 1750,
            window_size: 256,
            sample_rate_mhz: 250_000,
            bit_depth: 16,
            flags: BCS1_FLAG_HAS_FOOTER,
            metadata_length: 42,
        }
    }

    #[test]
    fn to_bytes_is_exactly_40_bytes_and_starts_with_magic() {
        let h = sample_header();
        let bytes = h.to_bytes();
        assert_eq!(bytes.len(), BCS1_HEADER_LEN);
        assert_eq!(&bytes[0..4], BCS1_MAGIC);
    }

    #[test]
    fn round_trips_every_field() {
        let h = sample_header();
        let bytes = h.to_bytes();
        let parsed = Bcs1Header::parse(&bytes).expect("parse");
        assert_eq!(parsed, h);
    }

    #[test]
    fn round_trips_with_trailing_payload_bytes() {
        let h = sample_header();
        let mut buf = h.to_bytes().to_vec();
        buf.extend_from_slice(b"{}trailing payload bytes here");
        let parsed = Bcs1Header::parse(&buf).expect("parse with trailing bytes");
        assert_eq!(parsed, h);
    }

    #[test]
    fn reserved_bytes_are_always_zero() {
        let bytes = sample_header().to_bytes();
        assert_eq!(&bytes[32..40], &[0u8; 8]);
    }

    #[test]
    fn rejects_truncated_buffer() {
        let bytes = sample_header().to_bytes();
        for k in 0..BCS1_HEADER_LEN {
            let err = Bcs1Header::parse(&bytes[..k]).expect_err("must reject truncation");
            assert!(matches!(
                err,
                Bcs1ParseError::Truncated { expected: BCS1_HEADER_LEN, actual } if actual == k
            ));
        }
    }

    #[test]
    fn rejects_wrong_magic() {
        let mut bytes = sample_header().to_bytes();
        bytes[0..4].copy_from_slice(b"LML1");
        let err = Bcs1Header::parse(&bytes).expect_err("must reject wrong magic");
        assert!(matches!(err, Bcs1ParseError::InvalidMagic(m) if &m == b"LML1"));
    }

    #[test]
    fn legacy_codec_descriptors_retain_lmo_transform_id_values() {
        // Legacy descriptors remain identical to lamquant-lml-optimum/src/lmo.rs's
        // TRANSFORM_LML_53=0 / TRANSFORM_LMO_97=1 / TRANSFORM_OPTIMUM_LOSSLESS=2
        // (checked by inspection there, since this legacy IR must not depend
        // UP on lamquant-lml-optimum — see the crate dependency graph note
        // in lib.rs).
        assert_eq!(CodecDescriptor::Lml53.to_u8(), 0);
        assert_eq!(CodecDescriptor::Lmo97.to_u8(), 1);
        assert_eq!(CodecDescriptor::LmoLossless.to_u8(), 2);
        assert_eq!(CodecDescriptor::OptimumV2.to_u8(), 3);
    }

    #[test]
    fn codec_descriptor_from_u8_round_trips_known_values() {
        for d in [
            CodecDescriptor::Lml53,
            CodecDescriptor::Lmo97,
            CodecDescriptor::LmoLossless,
            CodecDescriptor::OptimumV2,
        ] {
            assert_eq!(CodecDescriptor::from_u8(d.to_u8()), Some(d));
        }
    }

    #[test]
    fn codec_descriptor_recognizes_lmq_fsq_and_rejects_the_rest() {
        assert_eq!(
            CodecDescriptor::from_u8(3),
            Some(CodecDescriptor::OptimumV2)
        );
        // 0x10 is now the recognized LMQ neural descriptor (ADR 0074 Track N).
        assert_eq!(
            CodecDescriptor::from_u8(CODEC_LMQ_FSQ),
            Some(CodecDescriptor::LmqFsq)
        );
        assert_eq!(CodecDescriptor::LmqFsq.to_u8(), 0x10);
        // 0x11..=0xFF stays reserved (unrecognized).
        assert_eq!(CodecDescriptor::from_u8(0x11), None);
        assert_eq!(CodecDescriptor::from_u8(0xFF), None);
    }

    #[test]
    fn header_is_stack_only_no_alloc_dependency() {
        // Bcs1Header/Bcs1ParseError are Copy — proof they carry no heap
        // allocation, which is the no_std+alloc contract this module
        // promises (the type itself doesn't even need `alloc`, only the
        // crate-level `no_std` cfg).
        fn assert_copy<T: Copy>() {}
        assert_copy::<Bcs1Header>();
        assert_copy::<Bcs1ParseError>();
    }
}
