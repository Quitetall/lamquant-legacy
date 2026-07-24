#![forbid(unsafe_code)]

use abir::{
    canonical_debug_json, logical_content_id, parse_canonical_dataset, payload_content_id,
    verify_payload_content, Atom, AtomTag, ByteOrder, Clock, ClockTag, ConceptId, DatasetDraft,
    DatasetTag, ElementType, Fidelity, FidelityKind, Layout, ObjectId, PayloadDescriptor, Presence,
    Rational, Recording, RecordingTag, SemanticAxis, SemanticRef, SignalBlock, SourceCapsule,
    SourceKey, Stream, StreamTag, Tensor, TimeAxis, TimeSegment, ValidationLimits,
};
use abir_adapter::{MappingDisposition, MappingEntry, MappingReport, ProfileId, SemanticCoverage};
use lamquant_legacy_ir::{Bcs1Header, BCS1_VERSION_MAJOR, CODEC_LML_53};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::borrow::Cow;
use std::fmt;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const RECEIPT_FILE: &str = "receipt.json";
const SOURCE_FILE: &str = "source.bin";
const DATASET_FILE: &str = "dataset.json";
const PAYLOAD_FILE: &str = "payload.i64le";
const LQTP1_PAYLOAD_FILE: &str = "payload.f32le";
const LMQC_PAYLOAD_FILE: &str = "payload.lmqc-latent";
/// Environment variable the retired `lml encrypt`/`lml decrypt` pair documented
/// as the scripted key source: 64 hexadecimal characters = 32 raw bytes.
const KEY_ENV: &str = "LAMQUANT_KEY";
const MAPPING_REPORT_FILE: &str = "mapping-report.json";
const FIDELITY_REPORT_FILE: &str = "fidelity-report.json";
const SEMANTIC_RECEIPT_FILE: &str = "semantic-receipt.json";
const EXPORT_FILE: &str = "legacy-output.bin";
const EXPORT_RECEIPT_FILE: &str = "export-receipt.json";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum LegacyFormat {
    Bcs1,
    Lml1,
    Lma1,
    Lma2,
    Lmqc,
    Lmlcrypt,
    Lqtp1,
    Lqtp2,
    Lqtp3,
}

impl LegacyFormat {
    pub const fn profile(self) -> &'static str {
        match self {
            Self::Bcs1 => "legacy.bcs1.v1",
            Self::Lml1 => "legacy.lml1.v1",
            Self::Lma1 => "legacy.lma.v1",
            Self::Lma2 => "legacy.lma.v2",
            Self::Lmqc => "legacy.lmqc.v1",
            Self::Lmlcrypt => "legacy.lmlcrypt.v1",
            Self::Lqtp1 => "legacy.lqtp.v1",
            Self::Lqtp2 => "legacy.lqtp.v2",
            Self::Lqtp3 => "legacy.lqtp.v3",
        }
    }

    pub const fn supports_semantic_import(self) -> bool {
        matches!(
            self,
            Self::Bcs1
                | Self::Lml1
                | Self::Lqtp1
                | Self::Lma1
                | Self::Lma2
                | Self::Lmqc
                | Self::Lmlcrypt
        )
    }

    pub const fn supports_reverse_export(self) -> bool {
        matches!(
            self,
            Self::Bcs1
                | Self::Lml1
                | Self::Lqtp1
                | Self::Lma1
                | Self::Lma2
                | Self::Lmqc
                | Self::Lmlcrypt
        )
    }
}

#[derive(Clone, Debug)]
struct ContainerFacts {
    channels: u64,
    samples_per_channel: u64,
    decoded_payload_bytes: u64,
    sample_rate_millihz: Option<u32>,
    modality_tag: Option<u8>,
    metadata: String,
}

#[derive(Clone, Debug)]
struct SemanticArtifacts {
    receipt: SemanticImportReceipt,
    dataset_json: Vec<u8>,
    /// One payload per imported recording, as `(file name, bytes)`.
    ///
    /// A single-stream container yields exactly one entry. An archive yields
    /// one entry per contained recording, so a multi-recording archive is never
    /// collapsed into a single synthetic signal — that would report semantics
    /// the source never carried.
    payloads: Vec<(String, Vec<u8>)>,
    mapping_report: Vec<u8>,
    fidelity_report: Vec<u8>,
    semantic_receipt: Vec<u8>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ConvertRequest {
    pub source: PathBuf,
    pub destination: PathBuf,
    pub accept_fidelity: bool,
    pub max_source_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ConvertReceipt {
    pub profile: String,
    pub source_blake3: String,
    pub source_bytes: u64,
    pub source_preserved: bool,
    pub semantic_mapping_claimed: bool,
    pub fidelity: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SemanticImportRequest {
    pub source: PathBuf,
    pub destination: PathBuf,
    pub accept_fidelity: bool,
    pub max_source_bytes: u64,
    pub max_decoded_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SemanticImportReceipt {
    pub profile: String,
    pub source_blake3: String,
    pub source_bytes: u64,
    pub decoded_channels: u64,
    pub decoded_samples_per_channel: u64,
    pub decoded_payload_bytes: u64,
    pub dataset_content_id: String,
    pub payload_content_id: String,
    pub source_preserved: bool,
    pub exact_sample_values: bool,
    pub exact_source_restoration: bool,
    pub semantic_equivalence: bool,
    pub timing: String,
    pub modality: String,
    pub semantic_coverage: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ExportPayload {
    pub content_id: String,
    pub path: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SemanticExportRequest {
    pub format: LegacyFormat,
    pub dataset: PathBuf,
    pub payloads: Vec<ExportPayload>,
    pub destination: PathBuf,
    pub accept_fidelity: bool,
    pub max_dataset_bytes: u64,
    pub max_payload_bytes: u64,
    pub max_output_bytes: u64,
    pub window_size: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SemanticExportReceipt {
    pub profile: String,
    pub dataset_content_id: String,
    pub output_blake3: String,
    pub output_bytes: u64,
    pub decoded_channels: u64,
    pub decoded_samples_per_channel: u64,
    pub exact_sample_values: bool,
    pub semantic_equivalence: bool,
    pub accepted_projection: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SemanticFidelityReport {
    pub schema: String,
    pub source_profile: String,
    pub exact_source_restoration: bool,
    pub exact_sample_values: bool,
    pub sample_values_changed: bool,
    pub timing_equivalence: bool,
    pub modality_equivalence: bool,
    pub semantic_equivalence: bool,
    pub source_capsule_file: String,
    pub caveats: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Inspection {
    pub profile: String,
    pub source_bytes: u64,
    pub source_blake3: String,
    pub semantic_conversion: bool,
    pub forensic_conversion: bool,
    pub decoded_channels: Option<u64>,
    pub decoded_samples_per_channel: Option<u64>,
    pub decoded_payload_bytes: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Capability {
    pub profile: String,
    pub inspect: bool,
    pub forensic_import: bool,
    pub semantic_import: bool,
    pub reverse_export: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CapabilityManifest {
    pub schema: String,
    pub process_protocol: String,
    pub capabilities: Vec<Capability>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "operation", rename_all = "kebab-case")]
pub enum ProcessRequest {
    Manifest,
    Inspect {
        source: PathBuf,
        max_source_bytes: u64,
    },
    ConvertForensic(ConvertRequest),
    ImportSemantic(SemanticImportRequest),
    ExportSemantic(SemanticExportRequest),
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "status", content = "value", rename_all = "kebab-case")]
pub enum ProcessResponse {
    OkManifest(CapabilityManifest),
    OkInspection(Inspection),
    OkConversion(ConvertReceipt),
    OkSemanticImport(SemanticImportReceipt),
    OkSemanticExport(SemanticExportReceipt),
    Error { code: String, message: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LegacyError {
    UnknownMagic,
    SourceTooLarge,
    AcceptanceRequired,
    SemanticImportUnsupported,
    SemanticExportUnsupported,
    /// The AEAD key an encrypted retired blob needs is absent or malformed.
    /// A capability gap, not a broken file.
    KeyUnavailable,
    DecodedTooLarge,
    PayloadIdentityMismatch,
    MalformedContainer(String),
    SemanticValidation(String),
    UnsafeSource,
    DestinationConflict,
    Io(String),
    InvalidProtocol(String),
}

impl LegacyError {
    pub const fn code(&self) -> &'static str {
        match self {
            Self::UnknownMagic => "unknown-magic",
            Self::SourceTooLarge => "source-too-large",
            Self::AcceptanceRequired => "acceptance-required",
            Self::SemanticImportUnsupported => "semantic-import-unsupported",
            Self::SemanticExportUnsupported => "semantic-export-unsupported",
            Self::KeyUnavailable => "key-unavailable",
            Self::DecodedTooLarge => "decoded-output-too-large",
            Self::PayloadIdentityMismatch => "payload-identity-mismatch",
            Self::MalformedContainer(_) => "malformed-container",
            Self::SemanticValidation(_) => "semantic-validation",
            Self::UnsafeSource => "unsafe-source",
            Self::DestinationConflict => "destination-conflict",
            Self::Io(_) => "io",
            Self::InvalidProtocol(_) => "invalid-protocol",
        }
    }
}

impl fmt::Display for LegacyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownMagic => formatter.write_str("unsupported retired wire magic"),
            Self::SourceTooLarge => formatter.write_str("source exceeds declared byte limit"),
            Self::AcceptanceRequired => {
                formatter.write_str("fidelity receipt must be accepted before writing")
            }
            Self::SemanticImportUnsupported => formatter.write_str(
                "retired profile has no current semantic importer; forensic import remains available",
            ),
            Self::SemanticExportUnsupported => formatter.write_str(
                "current ABIR semantics cannot be represented by the requested retired profile",
            ),
            Self::KeyUnavailable => formatter.write_str(
                "encrypted retired blob needs a 64-hex-character LAMQUANT_KEY in the environment",
            ),
            Self::DecodedTooLarge => {
                formatter.write_str("decoded signal exceeds declared byte limit")
            }
            Self::PayloadIdentityMismatch => {
                formatter.write_str("payload bytes do not match their ABIR ContentId")
            }
            Self::MalformedContainer(message) => write!(formatter, "malformed container: {message}"),
            Self::SemanticValidation(message) => {
                write!(formatter, "semantic ABIR validation failed: {message}")
            }
            Self::UnsafeSource => formatter.write_str("source must be a regular non-symlink file"),
            Self::DestinationConflict => {
                formatter.write_str("destination exists with different conversion evidence")
            }
            Self::Io(message) => write!(formatter, "I/O failure: {message}"),
            Self::InvalidProtocol(message) => write!(formatter, "invalid protocol: {message}"),
        }
    }
}

impl std::error::Error for LegacyError {}

pub fn detect_format(bytes: &[u8]) -> Result<LegacyFormat, LegacyError> {
    if bytes.starts_with(b"LMLCRYPT") {
        return Ok(LegacyFormat::Lmlcrypt);
    }
    if bytes.starts_with(b"BCS1") {
        return Ok(LegacyFormat::Bcs1);
    }
    if bytes.starts_with(b"LML1") {
        return Ok(LegacyFormat::Lml1);
    }
    if bytes.starts_with(b"LMA1") {
        return Ok(LegacyFormat::Lma1);
    }
    if bytes.starts_with(b"LMA2") {
        return Ok(LegacyFormat::Lma2);
    }
    if bytes.starts_with(b"LMQC") {
        return Ok(LegacyFormat::Lmqc);
    }
    // The three tensor-pack generations do NOT share one magic with a version
    // byte: LQTP1 is `LQTP` + version 1, while LQTP2 and LQTP3 each took their
    // own four-byte magic (`LQT2` / `LQT3`). Detecting `LQTP\x02` would name a
    // wire no writer ever produced and would refuse the real ones.
    if bytes.starts_with(b"LQT2") {
        return Ok(LegacyFormat::Lqtp2);
    }
    if bytes.starts_with(b"LQT3") {
        return Ok(LegacyFormat::Lqtp3);
    }
    if bytes.starts_with(b"LQTP") && bytes.len() >= 5 {
        return match bytes[4] {
            1 => Ok(LegacyFormat::Lqtp1),
            _ => Err(LegacyError::UnknownMagic),
        };
    }
    Err(LegacyError::UnknownMagic)
}

pub fn capability_manifest() -> CapabilityManifest {
    let formats = [
        LegacyFormat::Bcs1,
        LegacyFormat::Lml1,
        LegacyFormat::Lma1,
        LegacyFormat::Lma2,
        LegacyFormat::Lmqc,
        LegacyFormat::Lmlcrypt,
        LegacyFormat::Lqtp1,
        LegacyFormat::Lqtp2,
        LegacyFormat::Lqtp3,
    ];
    CapabilityManifest {
        schema: "lamquant.legacy-capabilities/v1".to_owned(),
        process_protocol: "abir.adapter-process/v1".to_owned(),
        capabilities: formats
            .into_iter()
            .map(|format| Capability {
                profile: format.profile().to_owned(),
                inspect: true,
                forensic_import: true,
                semantic_import: format.supports_semantic_import(),
                reverse_export: format.supports_reverse_export(),
            })
            .collect(),
    }
}

/// The shape a retired container declares in its own headers: channels,
/// samples per channel, decoded payload bytes.
///
/// Header-only by construction — inspection must not decode a whole recording
/// to answer "what is this". `None` means the wire genuinely cannot say without
/// more than inspection has: a sealed AEAD envelope needs its key, and asking
/// inspect for a key would make simply listing a directory a privileged act.
fn inspect_shape(
    bytes: &[u8],
    format: LegacyFormat,
) -> Result<Option<(u64, u64, u64)>, LegacyError> {
    match format {
        LegacyFormat::Lqtp1 => {
            let header = parse_lqtp1_header(bytes)?;
            let samples = header
                .n_windows
                .checked_mul(header.window_len)
                .and_then(|value| u64::try_from(value).ok())
                .ok_or(LegacyError::DecodedTooLarge)?;
            let channels =
                u64::try_from(header.n_channels).map_err(|_| LegacyError::DecodedTooLarge)?;
            let payload_bytes = channels
                .checked_mul(samples)
                .and_then(|value| value.checked_mul(4))
                .ok_or(LegacyError::DecodedTooLarge)?;
            Ok(Some((channels, samples, payload_bytes)))
        }
        LegacyFormat::Lmqc => {
            let container = lamquant_lml_mcu::lmqc::decode_lmqc(bytes)
                .map_err(|error| LegacyError::MalformedContainer(format!("{error:?}")))?;
            Ok(Some((
                u64::from(container.n_channels),
                u64::from(container.window_samples),
                u64::try_from(container.payload.len()).map_err(|_| LegacyError::DecodedTooLarge)?,
            )))
        }
        LegacyFormat::Lma1 | LegacyFormat::Lma2 => {
            let (channels, samples, payload_bytes) = inspect_archive_shape(bytes)?;
            Ok(Some((channels, samples, payload_bytes)))
        }
        LegacyFormat::Bcs1 | LegacyFormat::Lml1 => {
            let facts = inspect_container(bytes, format, u64::MAX)?;
            Ok(Some((
                facts.channels,
                facts.samples_per_channel,
                facts.decoded_payload_bytes,
            )))
        }
        LegacyFormat::Lmlcrypt => Ok(None),
        _ => Ok(None),
    }
}

/// Verify the 32-byte SHA-256 trailer both archive generations append.
///
/// The codec's reader deliberately does NOT check it: it opens archives by
/// seeking, and hashing a multi-hundred-gigabyte corpus on every open would be
/// ruinous. This adapter already holds the whole blob in memory under
/// `max_source_bytes`, so it can afford the check -- and a retired archive
/// whose own integrity trailer disagrees with its bytes must be refused, not
/// imported with the corruption carried into ABIR semantics.
fn verify_archive_digest(source: &[u8]) -> Result<(), LegacyError> {
    use sha2::Digest;

    let split = source.len().checked_sub(32).ok_or_else(|| {
        LegacyError::MalformedContainer("archive has no digest trailer".to_owned())
    })?;
    let (body, recorded) = source.split_at(split);
    if sha2::Sha256::digest(body).as_slice() != recorded {
        return Err(LegacyError::MalformedContainer(
            "archive SHA-256 trailer does not match its bytes".to_owned(),
        ));
    }
    Ok(())
}

/// Archive-wide sums over the signal entries, read from each entry's header.
fn inspect_archive_shape(bytes: &[u8]) -> Result<(u64, u64, u64), LegacyError> {
    use lamquant_lml_archive::lma;

    verify_archive_digest(bytes)?;
    let mut temporary = tempfile::NamedTempFile::new().map_err(io_error)?;
    temporary.write_all(bytes).map_err(io_error)?;
    temporary.flush().map_err(io_error)?;
    let path = temporary.path();
    let entries = lma::list_archive(path)
        .map_err(|error| LegacyError::MalformedContainer(error.to_string()))?;
    let (mut channels, mut samples, mut payload_bytes) = (0_u64, 0_u64, 0_u64);
    for entry in entries
        .iter()
        .filter(|entry| entry.method == lma::Method::Lml)
    {
        let stored = lma::read_entry(path, &entry.path)
            .map_err(|error| LegacyError::MalformedContainer(error.to_string()))?;
        let encoded = decode_source(&stored, detect_format(&stored)?)?.into_owned();
        let header = lamquant_lml_legacy::container::parse_header(&encoded)
            .map_err(|error| LegacyError::MalformedContainer(error.to_string()))?;
        let entry_channels =
            u64::try_from(header.n_ch).map_err(|_| LegacyError::DecodedTooLarge)?;
        let entry_samples =
            u64::try_from(header.total_samples).map_err(|_| LegacyError::DecodedTooLarge)?;
        channels = channels
            .checked_add(entry_channels)
            .ok_or(LegacyError::DecodedTooLarge)?;
        samples = samples
            .checked_add(entry_samples)
            .ok_or(LegacyError::DecodedTooLarge)?;
        payload_bytes = entry_channels
            .checked_mul(entry_samples)
            .and_then(|value| value.checked_mul(8))
            .and_then(|value| payload_bytes.checked_add(value))
            .ok_or(LegacyError::DecodedTooLarge)?;
    }
    Ok((channels, samples, payload_bytes))
}

pub fn inspect(source: &Path, max_source_bytes: u64) -> Result<Inspection, LegacyError> {
    let bytes = read_bounded(source, max_source_bytes)?;
    let format = detect_format(&bytes)?;
    let shape = if format.supports_semantic_import() {
        inspect_shape(&bytes, format)?
    } else {
        None
    };
    Ok(Inspection {
        profile: format.profile().to_owned(),
        source_bytes: bytes.len() as u64,
        source_blake3: blake3::hash(&bytes).to_hex().to_string(),
        semantic_conversion: format.supports_semantic_import(),
        forensic_conversion: true,
        decoded_channels: shape.map(|(channels, _, _)| channels),
        decoded_samples_per_channel: shape.map(|(_, samples, _)| samples),
        decoded_payload_bytes: shape.map(|(_, _, bytes)| bytes),
    })
}

pub fn convert_forensic(request: &ConvertRequest) -> Result<ConvertReceipt, LegacyError> {
    if !request.accept_fidelity {
        return Err(LegacyError::AcceptanceRequired);
    }
    let bytes = read_bounded(&request.source, request.max_source_bytes)?;
    let format = detect_format(&bytes)?;
    let receipt = ConvertReceipt {
        profile: format.profile().to_owned(),
        source_blake3: blake3::hash(&bytes).to_hex().to_string(),
        source_bytes: bytes.len() as u64,
        source_preserved: true,
        semantic_mapping_claimed: false,
        fidelity: "exact-source-capsule-only".to_owned(),
    };
    if request.destination.exists() {
        return verify_existing(&request.destination, &bytes, &receipt);
    }
    let parent = request
        .destination
        .parent()
        .ok_or_else(|| LegacyError::Io("destination has no parent".to_owned()))?;
    fs::create_dir_all(parent).map_err(io_error)?;
    let temporary = tempfile::Builder::new()
        .prefix(".lamquant-legacy-")
        .tempdir_in(parent)
        .map_err(io_error)?;
    let result = (|| {
        write_new(&temporary.path().join(SOURCE_FILE), &bytes)?;
        let receipt_bytes = serde_json::to_vec_pretty(&receipt)
            .map_err(|error| LegacyError::InvalidProtocol(error.to_string()))?;
        write_new(&temporary.path().join(RECEIPT_FILE), &receipt_bytes)?;
        fs::rename(temporary.path(), &request.destination).map_err(io_error)?;
        Ok(receipt.clone())
    })();
    if result.is_ok() {
        std::mem::forget(temporary);
    }
    result
}

pub fn import_semantic(
    request: &SemanticImportRequest,
) -> Result<SemanticImportReceipt, LegacyError> {
    if !request.accept_fidelity {
        return Err(LegacyError::AcceptanceRequired);
    }
    let source = read_bounded(&request.source, request.max_source_bytes)?;
    let format = detect_format(&source)?;
    if !format.supports_semantic_import() {
        return Err(LegacyError::SemanticImportUnsupported);
    }
    let artifacts = if format == LegacyFormat::Lmlcrypt {
        // The envelope carries no semantics of its own. Open it, then import
        // whatever container it protected, anchored to the CIPHERTEXT so the
        // dataset names the bytes that are actually on disk.
        let (plaintext, envelope) = open_lmlcrypt(&source)?;
        build_artifacts(&plaintext, &envelope, request.max_decoded_bytes)?
    } else {
        build_artifacts(
            &source,
            &SourceAnchor::direct(&source, format),
            request.max_decoded_bytes,
        )?
    };

    if request.destination.exists() {
        return verify_existing_semantic(&request.destination, &source, &artifacts);
    }
    let parent = request
        .destination
        .parent()
        .ok_or_else(|| LegacyError::Io("destination has no parent".to_owned()))?;
    fs::create_dir_all(parent).map_err(io_error)?;
    let temporary = tempfile::Builder::new()
        .prefix(".lamquant-legacy-semantic-")
        .tempdir_in(parent)
        .map_err(io_error)?;
    let result = (|| {
        let root = temporary.path();
        write_new(&root.join(SOURCE_FILE), &source)?;
        write_new(&root.join(DATASET_FILE), &artifacts.dataset_json)?;
        for (name, bytes) in &artifacts.payloads {
            write_new(&root.join(name), bytes)?;
        }
        write_new(&root.join(MAPPING_REPORT_FILE), &artifacts.mapping_report)?;
        write_new(&root.join(FIDELITY_REPORT_FILE), &artifacts.fidelity_report)?;
        write_new(
            &root.join(SEMANTIC_RECEIPT_FILE),
            &artifacts.semantic_receipt,
        )?;
        fs::rename(root, &request.destination).map_err(io_error)?;
        Ok(artifacts.receipt.clone())
    })();
    if result.is_ok() {
        std::mem::forget(temporary);
    }
    result
}

pub fn export_semantic(
    request: &SemanticExportRequest,
) -> Result<SemanticExportReceipt, LegacyError> {
    if !request.accept_fidelity {
        return Err(LegacyError::AcceptanceRequired);
    }
    if !request.format.supports_reverse_export() {
        return Err(LegacyError::SemanticExportUnsupported);
    }
    let dataset_bytes = read_bounded(&request.dataset, request.max_dataset_bytes)?;
    let dataset = parse_canonical_dataset(&dataset_bytes)
        .map_err(|error| LegacyError::SemanticValidation(error.to_string()))?;
    if canonical_debug_json(&dataset)
        .map_err(|error| LegacyError::SemanticValidation(error.to_string()))?
        != dataset_bytes
    {
        return Err(LegacyError::SemanticValidation(
            "dataset input is not canonical ABIR JSON".to_owned(),
        ));
    }
    // Three retired profiles re-emit from the exact source capsule: a tensor
    // pack and a neural container hold no sample-domain semantics this adapter
    // could re-encode, and an AEAD envelope cannot be rebuilt without its key.
    // Capsule-exact re-emission is byte-identical and invents nothing.
    if matches!(
        request.format,
        LegacyFormat::Lqtp1 | LegacyFormat::Lmqc | LegacyFormat::Lmlcrypt
    ) {
        return export_capsule_exact(request, &dataset);
    }
    if matches!(request.format, LegacyFormat::Lma1 | LegacyFormat::Lma2) {
        return export_lma_archive(request, &dataset);
    }
    let (signal, sample_rate, modality_tag) =
        resolve_export_signal(&dataset, &request.payloads, request.max_payload_bytes)?;
    let window_size =
        usize::try_from(request.window_size).map_err(|_| LegacyError::SemanticExportUnsupported)?;
    if window_size == 0 {
        return Err(LegacyError::SemanticExportUnsupported);
    }

    let scratch = tempfile::tempdir().map_err(io_error)?;
    let lml_path = scratch.path().join("encoded.lml");
    lamquant_lml_legacy::container::write_file(
        &lml_path,
        &signal,
        sample_rate,
        window_size,
        0,
        "{}",
    )
    .map_err(|error| LegacyError::MalformedContainer(error.to_string()))?;
    let lml = read_bounded(&lml_path, request.max_output_bytes)?;
    let output = match request.format {
        LegacyFormat::Lml1 => lml,
        LegacyFormat::Bcs1 => lml1_as_bcs1(&lml, modality_tag)?,
        _ => return Err(LegacyError::SemanticExportUnsupported),
    };
    if u64::try_from(output.len()).map_err(|_| LegacyError::SourceTooLarge)?
        > request.max_output_bytes
    {
        return Err(LegacyError::SourceTooLarge);
    }
    let decoded_source = decode_source(&output, request.format)?;
    let decoded = lamquant_lml_legacy::container::read_bytes(&decoded_source)
        .map_err(|error| LegacyError::MalformedContainer(error.to_string()))?
        .0;
    if decoded != signal {
        return Err(LegacyError::MalformedContainer(
            "reverse-export verification changed sample values".to_owned(),
        ));
    }
    let receipt = SemanticExportReceipt {
        profile: request.format.profile().to_owned(),
        dataset_content_id: logical_content_id(&dataset)
            .map_err(|error| LegacyError::SemanticValidation(error.to_string()))?
            .to_string(),
        output_blake3: blake3::hash(&output).to_hex().to_string(),
        output_bytes: output.len() as u64,
        decoded_channels: signal.len() as u64,
        decoded_samples_per_channel: signal.first().map_or(0, Vec::len) as u64,
        exact_sample_values: true,
        semantic_equivalence: false,
        accepted_projection: true,
    };
    let receipt_bytes = serde_json::to_vec_pretty(&receipt)
        .map_err(|error| LegacyError::InvalidProtocol(error.to_string()))?;
    if request.destination.exists() {
        return verify_existing_export(&request.destination, &output, &receipt_bytes, &receipt);
    }
    let parent = request
        .destination
        .parent()
        .ok_or_else(|| LegacyError::Io("destination has no parent".to_owned()))?;
    fs::create_dir_all(parent).map_err(io_error)?;
    let temporary = tempfile::Builder::new()
        .prefix(".lamquant-legacy-export-")
        .tempdir_in(parent)
        .map_err(io_error)?;
    let result = (|| {
        write_new(&temporary.path().join(EXPORT_FILE), &output)?;
        write_new(&temporary.path().join(EXPORT_RECEIPT_FILE), &receipt_bytes)?;
        fs::rename(temporary.path(), &request.destination).map_err(io_error)?;
        Ok(receipt.clone())
    })();
    if result.is_ok() {
        std::mem::forget(temporary);
    }
    result
}

/// Re-emit a multi-recording dataset as an LMA archive, one LML entry per
/// recording.
///
/// The archive writer always produces the v2 layout, so the receipt reports
/// `legacy.lma.v2` whatever generation was requested: claiming an exact v1
/// re-emission would assert a wire this function does not write. Sample values
/// are verified exact by decoding every entry back out of the finished archive.
fn export_lma_archive(
    request: &SemanticExportRequest,
    dataset: &abir::AbirDataset,
) -> Result<SemanticExportReceipt, LegacyError> {
    let signals = resolve_export_signals(dataset, &request.payloads, request.max_payload_bytes)?;
    let window_size =
        usize::try_from(request.window_size).map_err(|_| LegacyError::SemanticExportUnsupported)?;
    if window_size == 0 {
        return Err(LegacyError::SemanticExportUnsupported);
    }
    let scratch = tempfile::tempdir().map_err(io_error)?;
    let mut encoded: Vec<(String, Vec<u8>)> = Vec::with_capacity(signals.len());
    for (index, (signal, sample_rate, _modality)) in signals.iter().enumerate() {
        let lml_path = scratch.path().join(format!("recording-{index:04}.lml"));
        lamquant_lml_legacy::container::write_file(
            &lml_path,
            signal,
            *sample_rate,
            window_size,
            0,
            "{}",
        )
        .map_err(|error| LegacyError::MalformedContainer(error.to_string()))?;
        let bytes = read_bounded(&lml_path, request.max_output_bytes)?;
        encoded.push((format!("recording-{index:04}.lml"), bytes));
    }
    let borrowed: Vec<(&str, &[u8])> = encoded
        .iter()
        .map(|(name, bytes)| (name.as_str(), bytes.as_slice()))
        .collect();
    let archive_path = scratch.path().join("archive.lma");
    lamquant_lml_archive::lma::pack_lml_entries(&borrowed, &archive_path, 3)
        .map_err(|error| LegacyError::MalformedContainer(error.to_string()))?;
    let output = read_bounded(&archive_path, request.max_output_bytes)?;
    if u64::try_from(output.len()).map_err(|_| LegacyError::SourceTooLarge)?
        > request.max_output_bytes
    {
        return Err(LegacyError::SourceTooLarge);
    }

    // Verify from the finished archive, not from the buffers we just wrote, so
    // a framing bug cannot pass unnoticed.
    let written = lamquant_lml_archive::lma::list_archive(&archive_path)
        .map_err(|error| LegacyError::MalformedContainer(error.to_string()))?;
    if written.len() != signals.len() {
        return Err(LegacyError::MalformedContainer(
            "re-emitted archive entry count does not match the dataset".to_owned(),
        ));
    }
    let mut total_channels = 0_u64;
    let mut total_samples = 0_u64;
    for (entry, (signal, _rate, _modality)) in written.iter().zip(signals.iter()) {
        let stored = lamquant_lml_archive::lma::read_entry(&archive_path, &entry.path)
            .map_err(|error| LegacyError::MalformedContainer(error.to_string()))?;
        let decoded = lamquant_lml_legacy::container::read_bytes(&stored)
            .map_err(|error| LegacyError::MalformedContainer(error.to_string()))?
            .0;
        if &decoded != signal {
            return Err(LegacyError::MalformedContainer(
                "reverse-export verification changed sample values".to_owned(),
            ));
        }
        total_channels = total_channels
            .checked_add(signal.len() as u64)
            .ok_or(LegacyError::DecodedTooLarge)?;
        total_samples = total_samples
            .checked_add(signal.first().map_or(0, Vec::len) as u64)
            .ok_or(LegacyError::DecodedTooLarge)?;
    }

    let destination = &request.destination;
    if destination.exists() {
        return Err(LegacyError::DestinationConflict);
    }
    write_new(destination, &output)?;
    Ok(SemanticExportReceipt {
        // The writer emits v2; report what was actually produced.
        profile: LegacyFormat::Lma2.profile().to_owned(),
        dataset_content_id: logical_content_id(dataset)
            .map_err(|error| LegacyError::SemanticValidation(error.to_string()))?
            .to_string(),
        output_blake3: blake3::hash(&output).to_hex().to_string(),
        output_bytes: output.len() as u64,
        decoded_channels: total_channels,
        decoded_samples_per_channel: total_samples,
        exact_sample_values: true,
        semantic_equivalence: false,
        accepted_projection: true,
    })
}

/// Re-emit the exact retired source held in the dataset's source capsule.
///
/// The capsule's media type decides which wire this is, so a capsule of the
/// wrong generation cannot be re-emitted under another profile's name. The
/// bytes are verified against the capsule ContentId AND re-parsed as the
/// requested wire before anything is written.
fn export_capsule_exact(
    request: &SemanticExportRequest,
    dataset: &abir::AbirDataset,
) -> Result<SemanticExportReceipt, LegacyError> {
    if dataset.source_capsules().len() != 1 {
        return Err(LegacyError::SemanticExportUnsupported);
    }
    let capsule = &dataset.source_capsules()[0];
    if capsule.media_type() != Some(legacy_media_type(request.format)) {
        return Err(LegacyError::SemanticExportUnsupported);
    }
    let capsule_id = capsule.content_id().to_string();
    let source = request
        .payloads
        .iter()
        .find(|payload| payload.content_id == capsule_id)
        .ok_or(LegacyError::PayloadIdentityMismatch)?;
    if request
        .payloads
        .iter()
        .filter(|payload| payload.content_id == capsule_id)
        .count()
        != 1
    {
        return Err(LegacyError::InvalidProtocol(
            "duplicate retired source-capsule payload".to_owned(),
        ));
    }
    let output = read_bounded(
        &source.path,
        request.max_payload_bytes.min(request.max_output_bytes),
    )?;
    if payload_content_id(ElementType::Bytes, &output) != capsule.content_id() {
        return Err(LegacyError::PayloadIdentityMismatch);
    }
    // Re-parse as the requested wire: matching the capsule ContentId proves
    // the bytes are unchanged, not that they are the format being claimed.
    if detect_format(&output)? != request.format {
        return Err(LegacyError::SemanticExportUnsupported);
    }
    let (channels, samples) =
        capsule_exact_shape(&output, request.format, request.max_payload_bytes)?;
    let receipt = SemanticExportReceipt {
        profile: request.format.profile().to_owned(),
        dataset_content_id: logical_content_id(dataset)
            .map_err(|error| LegacyError::SemanticValidation(error.to_string()))?
            .to_string(),
        output_blake3: blake3::hash(&output).to_hex().to_string(),
        output_bytes: output.len() as u64,
        decoded_channels: channels,
        decoded_samples_per_channel: samples,
        // The retired bytes are reproduced exactly; for LMQC and LMLCRYPT the
        // sample domain was never entered, so nothing there was altered either.
        exact_sample_values: true,
        semantic_equivalence: false,
        accepted_projection: true,
    };
    let receipt_bytes = serde_json::to_vec_pretty(&receipt)
        .map_err(|error| LegacyError::InvalidProtocol(error.to_string()))?;
    if request.destination.exists() {
        return verify_existing_export(&request.destination, &output, &receipt_bytes, &receipt);
    }
    let parent = request
        .destination
        .parent()
        .ok_or_else(|| LegacyError::Io("destination has no parent".to_owned()))?;
    fs::create_dir_all(parent).map_err(io_error)?;
    let temporary = tempfile::Builder::new()
        .prefix(".lamquant-legacy-export-")
        .tempdir_in(parent)
        .map_err(io_error)?;
    let result = (|| {
        write_new(&temporary.path().join(EXPORT_FILE), &output)?;
        write_new(&temporary.path().join(EXPORT_RECEIPT_FILE), &receipt_bytes)?;
        fs::rename(temporary.path(), &request.destination).map_err(io_error)?;
        Ok(receipt.clone())
    })();
    if result.is_ok() {
        std::mem::forget(temporary);
    }
    result
}

/// Channels and samples-per-channel a capsule-exact re-emission describes.
///
/// Derived by importing the bytes about to be written, so the export receipt
/// reports the same shape the import receipt did, by construction rather than
/// by a second implementation that could drift from it. An LMLCRYPT envelope is
/// opened first, so an export without the key fails closed rather than
/// reporting a made-up shape.
fn capsule_exact_shape(
    output: &[u8],
    format: LegacyFormat,
    max_decoded_bytes: u64,
) -> Result<(u64, u64), LegacyError> {
    let artifacts = if format == LegacyFormat::Lmlcrypt {
        let (plaintext, envelope) = open_lmlcrypt(output)?;
        build_artifacts(&plaintext, &envelope, max_decoded_bytes)?
    } else {
        build_artifacts(
            output,
            &SourceAnchor::direct(output, format),
            max_decoded_bytes,
        )?
    };
    Ok((
        artifacts.receipt.decoded_channels,
        artifacts.receipt.decoded_samples_per_channel,
    ))
}

/// One decoded recording: per-channel samples, sample rate in Hz, modality tag.
type ExportSignal = (Vec<Vec<i64>>, f64, u8);

/// Build the ContentId -> payload path index, rejecting duplicate identities.
fn export_payload_paths(
    payloads: &[ExportPayload],
) -> Result<std::collections::BTreeMap<&str, &Path>, LegacyError> {
    let mut paths = std::collections::BTreeMap::new();
    for payload in payloads {
        if paths
            .insert(payload.content_id.as_str(), payload.path.as_path())
            .is_some()
        {
            return Err(LegacyError::InvalidProtocol(
                "duplicate export payload ContentId".to_owned(),
            ));
        }
    }
    Ok(paths)
}

/// Resolve a single-recording dataset to one signal (the LML/BCS1 export path).
fn resolve_export_signal(
    dataset: &abir::AbirDataset,
    payloads: &[ExportPayload],
    max_payload_bytes: u64,
) -> Result<ExportSignal, LegacyError> {
    if dataset.recordings().len() != 1 || dataset.streams().len() != 1 {
        return Err(LegacyError::SemanticExportUnsupported);
    }
    let recording = &dataset.recordings()[0];
    let stream = &dataset.streams()[0];
    if recording.streams() != [stream.id()]
        || stream.recording_id() != recording.id()
        || stream.atoms().len() != dataset.atoms().len()
    {
        return Err(LegacyError::SemanticExportUnsupported);
    }
    let paths = export_payload_paths(payloads)?;
    let mut used_payloads = std::collections::BTreeSet::new();
    let mut total_payload_bytes = 0_u64;
    let resolved = decode_stream_signal(
        dataset,
        stream,
        &paths,
        &mut used_payloads,
        &mut total_payload_bytes,
        max_payload_bytes,
    )?;
    // Every supplied payload must have been consumed, or the caller handed us
    // bytes this dataset does not describe.
    if used_payloads.len() != paths.len() {
        return Err(LegacyError::SemanticExportUnsupported);
    }
    Ok(resolved)
}

/// Resolve a multi-recording dataset to one signal per recording, in dataset
/// order. Used by the archive export path, where each recording re-emits its
/// own container.
#[allow(dead_code)]
fn resolve_export_signals(
    dataset: &abir::AbirDataset,
    payloads: &[ExportPayload],
    max_payload_bytes: u64,
) -> Result<Vec<ExportSignal>, LegacyError> {
    if dataset.recordings().is_empty() {
        return Err(LegacyError::SemanticExportUnsupported);
    }
    let paths = export_payload_paths(payloads)?;
    let mut used_payloads = std::collections::BTreeSet::new();
    let mut total_payload_bytes = 0_u64;
    let mut resolved = Vec::with_capacity(dataset.recordings().len());
    for recording in dataset.recordings() {
        // Each recording owns exactly one stream in an archive projection.
        let [stream_id] = recording.streams() else {
            return Err(LegacyError::SemanticExportUnsupported);
        };
        let stream = dataset
            .streams()
            .iter()
            .find(|stream| stream.id() == *stream_id)
            .ok_or(LegacyError::SemanticExportUnsupported)?;
        if stream.recording_id() != recording.id() {
            return Err(LegacyError::SemanticExportUnsupported);
        }
        resolved.push(decode_stream_signal(
            dataset,
            stream,
            &paths,
            &mut used_payloads,
            &mut total_payload_bytes,
            max_payload_bytes,
        )?);
    }
    if used_payloads.len() != paths.len() {
        return Err(LegacyError::SemanticExportUnsupported);
    }
    Ok(resolved)
}

/// Bind one stream's atoms to their payloads and decode that stream's signal.
///
/// Shared by the single-recording resolver and the multi-recording archive
/// resolver so both apply exactly the same atom, descriptor, and shape checks.
/// Whether every supplied payload was consumed is a whole-dataset invariant and
/// is therefore asserted by the caller, not here.
fn decode_stream_signal(
    dataset: &abir::AbirDataset,
    stream: &abir::Stream,
    paths: &std::collections::BTreeMap<&str, &Path>,
    used_payloads: &mut std::collections::BTreeSet<String>,
    total_payload_bytes: &mut u64,
    max_payload_bytes: u64,
) -> Result<ExportSignal, LegacyError> {
    if stream.atoms().is_empty() {
        return Err(LegacyError::SemanticExportUnsupported);
    }
    let mut signal = Vec::new();
    let mut rate = None;
    let mut samples = None;
    for atom_id in stream.atoms() {
        let atom = dataset
            .atoms()
            .iter()
            .find(|atom| atom.id() == *atom_id)
            .ok_or(LegacyError::SemanticExportUnsupported)?;
        let Atom::SignalBlock(block) = atom else {
            return Err(LegacyError::SemanticExportUnsupported);
        };
        if atom.presence() != Presence::Present {
            return Err(LegacyError::SemanticExportUnsupported);
        }
        let descriptor = atom
            .payload()
            .ok_or(LegacyError::SemanticExportUnsupported)?;
        if descriptor.element() != ElementType::I64
            || descriptor.byte_order() != ByteOrder::Little
            || !matches!(descriptor.layout(), Layout::DenseRowMajor)
            || descriptor
                .encoding()
                .is_some_and(|encoding| encoding.as_str() != "abir:encoding/raw")
        {
            return Err(LegacyError::SemanticExportUnsupported);
        }
        let TimeAxis::Regular(segment) = block.time_axis() else {
            return Err(LegacyError::SemanticExportUnsupported);
        };
        if segment.start().parts() != (0, 1)
            || rate
                .replace(segment.rate())
                .is_some_and(|prior| prior != segment.rate())
            || samples
                .replace(segment.samples())
                .is_some_and(|prior| prior != segment.samples())
        {
            return Err(LegacyError::SemanticExportUnsupported);
        }
        let descriptor_id = descriptor.content_id().to_string();
        let path = paths
            .get(descriptor_id.as_str())
            .copied()
            .ok_or(LegacyError::PayloadIdentityMismatch)?;
        used_payloads.insert(descriptor_id);
        *total_payload_bytes = total_payload_bytes
            .checked_add(descriptor.logical_bytes())
            .ok_or(LegacyError::DecodedTooLarge)?;
        if *total_payload_bytes > max_payload_bytes {
            return Err(LegacyError::DecodedTooLarge);
        }
        let bytes = read_bounded(path, descriptor.logical_bytes())?;
        verify_payload_content(descriptor, &bytes)
            .map_err(|_| LegacyError::PayloadIdentityMismatch)?;
        let values = bytes
            .chunks_exact(8)
            .map(|sample| i64::from_le_bytes(sample.try_into().expect("exact chunk")))
            .collect::<Vec<_>>();
        let channel_samples =
            usize::try_from(segment.samples()).map_err(|_| LegacyError::DecodedTooLarge)?;
        match descriptor.shape() {
            [extent] if *extent == segment.samples() => signal.push(values),
            [1, extent] if *extent == segment.samples() => signal.push(values),
            [channels, extent]
                if *extent == segment.samples()
                    && usize::try_from(*channels)
                        .ok()
                        .and_then(|channels| channels.checked_mul(channel_samples))
                        == Some(values.len()) =>
            {
                signal.extend(values.chunks_exact(channel_samples).map(<[i64]>::to_vec));
            }
            _ => return Err(LegacyError::SemanticExportUnsupported),
        }
    }
    if signal.is_empty()
        || signal.len() > u16::MAX as usize
        || signal.first().map_or(0, Vec::len) > u32::MAX as usize
    {
        return Err(LegacyError::SemanticExportUnsupported);
    }
    let rate = rate.ok_or(LegacyError::SemanticExportUnsupported)?;
    let (numerator, denominator) = rate.parts();
    let millihz = numerator
        .checked_mul(1000)
        .filter(|value| *value > 0 && *value % denominator == 0)
        .map(|value| value / denominator)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or(LegacyError::SemanticExportUnsupported)?;
    Ok((
        signal,
        f64::from(millihz) / 1000.0,
        modality_tag(stream.modality())?,
    ))
}

fn modality_tag(modality: &ConceptId) -> Result<u8, LegacyError> {
    match modality.as_str() {
        "abir:modality/eeg" => Ok(0),
        "legacy:modality/ieeg" => Ok(1),
        "legacy:modality/ecog" => Ok(2),
        "legacy:modality/seeg" => Ok(3),
        "abir:modality/ecg" => Ok(4),
        "abir:modality/emg" => Ok(5),
        "abir:modality/eog" => Ok(6),
        "legacy:modality/respiration" => Ok(7),
        "legacy:modality/acceleration" => Ok(8),
        "legacy:modality/other" => Ok(9),
        "legacy:modality/untyped" | "legacy:modality/unknown-at-source" => Ok(255),
        _ => Err(LegacyError::SemanticExportUnsupported),
    }
}

fn lml1_as_bcs1(lml: &[u8], modality_tag: u8) -> Result<Vec<u8>, LegacyError> {
    if lml.len() < 32 || &lml[..4] != b"LML1" {
        return Err(LegacyError::MalformedContainer(
            "legacy encoder did not emit LML1".to_owned(),
        ));
    }
    let header = Bcs1Header {
        version_major: BCS1_VERSION_MAJOR,
        version_minor: 0,
        modality_tag,
        modality_source: 0,
        codec_descriptor: CODEC_LML_53,
        mode: 0,
        tier: 0,
        decode_capability: 0,
        n_channels: u16::from_le_bytes([lml[6], lml[7]]),
        n_windows: u16::from_le_bytes([lml[8], lml[9]]),
        total_samples: u32::from_le_bytes([lml[10], lml[11], lml[12], lml[13]]),
        window_size: u16::from_le_bytes([lml[14], lml[15]]),
        sample_rate_mhz: u32::from_le_bytes([lml[16], lml[17], lml[18], lml[19]]),
        bit_depth: lml[20],
        flags: lml[21],
        metadata_length: u32::from_le_bytes([lml[22], lml[23], lml[24], lml[25]]),
    };
    let mut output = Vec::with_capacity(lml.len() + 8);
    output.extend_from_slice(&header.to_bytes());
    output.extend_from_slice(&lml[32..]);
    Ok(output)
}

fn verify_existing_export(
    destination: &Path,
    output: &[u8],
    receipt_bytes: &[u8],
    receipt: &SemanticExportReceipt,
) -> Result<SemanticExportReceipt, LegacyError> {
    let metadata =
        fs::symlink_metadata(destination).map_err(|_| LegacyError::DestinationConflict)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(LegacyError::DestinationConflict);
    }
    if fs::read(destination.join(EXPORT_FILE)).ok().as_deref() == Some(output)
        && fs::read(destination.join(EXPORT_RECEIPT_FILE))
            .ok()
            .as_deref()
            == Some(receipt_bytes)
    {
        Ok(receipt.clone())
    } else {
        Err(LegacyError::DestinationConflict)
    }
}

fn inspect_container(
    bytes: &[u8],
    format: LegacyFormat,
    max_decoded_bytes: u64,
) -> Result<ContainerFacts, LegacyError> {
    if !format.supports_semantic_import() {
        return Err(LegacyError::SemanticImportUnsupported);
    }
    let bcs1_header = if format == LegacyFormat::Bcs1 {
        let header = Bcs1Header::parse(bytes)
            .map_err(|error| LegacyError::MalformedContainer(error.to_string()))?;
        if header.version_major > BCS1_VERSION_MAJOR {
            return Err(LegacyError::MalformedContainer(format!(
                "unsupported BCS1 major version {}",
                header.version_major
            )));
        }
        if header.decode_capability != 0 {
            return Err(LegacyError::MalformedContainer(format!(
                "BCS1 decode capability {} is unsupported",
                header.decode_capability
            )));
        }
        if header.codec_descriptor != CODEC_LML_53 {
            return Err(LegacyError::MalformedContainer(format!(
                "BCS1 codec descriptor {} is not the lossless LML 5/3 profile",
                header.codec_descriptor
            )));
        }
        Some(header)
    } else {
        None
    };
    let decode_source = match bcs1_header.as_ref() {
        Some(header) => Cow::Owned(bcs1_as_lml1(bytes, header)?),
        None => Cow::Borrowed(bytes),
    };
    let header = lamquant_lml_legacy::container::parse_header(&decode_source)
        .map_err(|error| LegacyError::MalformedContainer(error.to_string()))?;
    let channels = u64::try_from(header.n_ch).map_err(|_| LegacyError::DecodedTooLarge)?;
    let samples_per_channel =
        u64::try_from(header.total_samples).map_err(|_| LegacyError::DecodedTooLarge)?;
    let decoded_payload_bytes = channels
        .checked_mul(samples_per_channel)
        .and_then(|value| value.checked_mul(8))
        .ok_or(LegacyError::DecodedTooLarge)?;
    if decoded_payload_bytes > max_decoded_bytes {
        return Err(LegacyError::DecodedTooLarge);
    }
    let sample_rate_millihz = bcs1_header
        .as_ref()
        .map(|value| value.sample_rate_mhz)
        .filter(|value| *value > 0)
        .or_else(|| sample_rate_millihz(bytes, format, &header.metadata));
    let modality_tag = bcs1_header.map(|value| value.modality_tag);
    Ok(ContainerFacts {
        channels,
        samples_per_channel,
        decoded_payload_bytes,
        sample_rate_millihz,
        modality_tag,
        metadata: header.metadata,
    })
}

/// The bytes an imported dataset is anchored to, and how to describe them.
///
/// For a bare retired container the anchor IS the container. For an LMLCRYPT
/// blob it is the ciphertext: the file on disk is the envelope, so the source
/// capsule -- and every identity derived from it -- must be the envelope, or
/// the dataset would claim to have come from bytes nobody holds. The extra
/// mapping entries, caveats and source keys let the envelope state its own
/// contribution inside the inner container's reports rather than beside them.
struct SourceAnchor<'a> {
    bytes: &'a [u8],
    format: LegacyFormat,
    extra_mapping: Vec<MappingEntry>,
    extra_caveats: Vec<String>,
    extra_source_keys: Vec<(String, String)>,
}

impl<'a> SourceAnchor<'a> {
    /// The container is its own anchor: nothing wraps it.
    fn direct(bytes: &'a [u8], format: LegacyFormat) -> Self {
        Self {
            bytes,
            format,
            extra_mapping: Vec::new(),
            extra_caveats: Vec::new(),
            extra_source_keys: Vec::new(),
        }
    }

    fn source_keys(&self) -> Result<Vec<SourceKey>, LegacyError> {
        self.extra_source_keys
            .iter()
            .map(|(namespace, value)| {
                SourceKey::new(namespace, value)
                    .map_err(|error| LegacyError::SemanticValidation(error.to_string()))
            })
            .collect()
    }
}

/// Route one retired container to its builder. `source` is the container being
/// parsed; `anchor` is what the resulting dataset claims to have come from.
fn build_artifacts(
    source: &[u8],
    anchor: &SourceAnchor<'_>,
    max_decoded_bytes: u64,
) -> Result<SemanticArtifacts, LegacyError> {
    let format = detect_format(source)?;
    match format {
        LegacyFormat::Lqtp1 => build_lqtp1_artifacts(source, anchor, max_decoded_bytes),
        LegacyFormat::Lmqc => build_lmqc_artifacts(source, anchor, max_decoded_bytes),
        // An archive is a directory tree, not one LML stream, so it cannot go
        // through the single-container decode path. The archive reader resolves
        // either generation's layout, so v1 and v2 share this path.
        LegacyFormat::Lma1 | LegacyFormat::Lma2 => {
            build_lma_artifacts(source, anchor, max_decoded_bytes)
        }
        LegacyFormat::Bcs1 | LegacyFormat::Lml1 => {
            let facts = inspect_container(source, format, max_decoded_bytes)?;
            let decoded = decode_source(source, format)?;
            let signal = lamquant_lml_legacy::container::read_bytes(&decoded)
                .map_err(|error| LegacyError::MalformedContainer(error.to_string()))?
                .0;
            verify_decoded_shape(&signal, &facts)?;
            build_semantic_artifacts(anchor, &facts, signal, max_decoded_bytes)
        }
        _ => Err(LegacyError::SemanticImportUnsupported),
    }
}

/// The 32-byte AEAD key, from the environment variable the retired
/// `lml encrypt`/`lml decrypt` pair documented.
///
/// Absence is a capability gap, not a malformed file: it gets its own error so
/// a caller can tell "you need to provide the key" from "this blob is broken".
fn lmlcrypt_key() -> Result<[u8; 32], LegacyError> {
    let hex = std::env::var(KEY_ENV).map_err(|_| LegacyError::KeyUnavailable)?;
    if hex.len() != 64 {
        return Err(LegacyError::KeyUnavailable);
    }
    let mut key = [0_u8; 32];
    for (index, byte) in key.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16)
            .map_err(|_| LegacyError::KeyUnavailable)?;
    }
    Ok(key)
}

/// Authenticate and open an LMLCRYPT blob.
///
/// Returns the plaintext plus the anchor that re-attributes the imported
/// dataset to the envelope: the file on disk is the ciphertext, so that is what
/// the source capsule and every derived identity must name. A nested envelope
/// is refused rather than unwrapped, so one import can never be made to peel an
/// unbounded chain.
fn open_lmlcrypt(source: &[u8]) -> Result<(Vec<u8>, SourceAnchor<'_>), LegacyError> {
    use aes_gcm::aead::{Aead, KeyInit};

    const HEADER_LEN: usize = 8 + 1 + 12;
    if source.len() < HEADER_LEN + 16 {
        return Err(LegacyError::MalformedContainer(
            "LMLCRYPT blob is shorter than its header plus authentication tag".to_owned(),
        ));
    }
    if source[8] != 1 {
        return Err(LegacyError::MalformedContainer(format!(
            "unsupported LMLCRYPT version {}",
            source[8]
        )));
    }
    let nonce = &source[9..HEADER_LEN];
    let key = lmlcrypt_key()?;
    let cipher = aes_gcm::Aes256Gcm::new_from_slice(&key)
        .map_err(|_| LegacyError::MalformedContainer("AEAD key is not 32 bytes".to_owned()))?;
    let plaintext = cipher
        .decrypt(aes_gcm::Nonce::from_slice(nonce), &source[HEADER_LEN..])
        .map_err(|_| {
            // A tag mismatch means the wrong key OR a tampered blob. Both are
            // refusals; neither says which, because saying which is an oracle.
            LegacyError::MalformedContainer(
                "LMLCRYPT authentication failed: wrong key or altered ciphertext".to_owned(),
            )
        })?;
    let inner = detect_format(&plaintext)?;
    if inner == LegacyFormat::Lmlcrypt {
        return Err(LegacyError::MalformedContainer(
            "LMLCRYPT envelope contains another envelope".to_owned(),
        ));
    }
    let nonce_hex = nonce
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let anchor = SourceAnchor {
        bytes: source,
        format: LegacyFormat::Lmlcrypt,
        extra_mapping: vec![MappingEntry {
            source_path: "envelope.aead".to_owned(),
            target: "abir.source-capsule".to_owned(),
            disposition: MappingDisposition::Quarantined,
            reason: Some(
                "the AES-256-GCM envelope authenticated the payload; the nonce and version are \
                 recorded, the key never enters the dataset"
                    .to_owned(),
            ),
        }],
        extra_caveats: vec![format!(
            "source bytes are an AES-256-GCM envelope over a {} container; the semantics below \
             are the container's, and reading them again requires the same key",
            inner.profile()
        )],
        extra_source_keys: vec![
            ("lmlcrypt.version".to_owned(), source[8].to_string()),
            ("lmlcrypt.nonce".to_owned(), nonce_hex),
            (
                "lmlcrypt.inner-profile".to_owned(),
                inner.profile().to_owned(),
            ),
        ],
    };
    Ok((plaintext, anchor))
}

/// Import an LMQC neural container.
///
/// The payload is an ENCODED latent: decoding it back to samples needs the
/// neural decoder and its weights, which this adapter does not carry. So the
/// atom is an `EncodedBlock` whose `DecodedSemantics` states the shape the
/// payload decodes to, and the payload file holds the encoded bytes verbatim.
/// The montage the container exists to carry -- channel count, electrode
/// coordinates, channel names, sampling rate -- IS recovered exactly; claiming
/// sample values would be inventing them.
fn build_lmqc_artifacts(
    source: &[u8],
    anchor: &SourceAnchor<'_>,
    max_decoded_bytes: u64,
) -> Result<SemanticArtifacts, LegacyError> {
    let container = lamquant_lml_mcu::lmqc::decode_lmqc(source)
        .map_err(|error| LegacyError::MalformedContainer(format!("{error:?}")))?;
    let payload = container.payload.clone();
    let payload_bytes = u64::try_from(payload.len()).map_err(|_| LegacyError::DecodedTooLarge)?;
    if payload_bytes > max_decoded_bytes {
        return Err(LegacyError::DecodedTooLarge);
    }
    let channels = u64::from(container.n_channels);
    let window_samples = u64::from(container.window_samples);
    if channels == 0 || window_samples == 0 {
        return Err(LegacyError::MalformedContainer(
            "LMQC declares an empty montage or window".to_owned(),
        ));
    }

    let source_hash = blake3::hash(anchor.bytes);
    let source_blake3 = source_hash.to_hex().to_string();
    let payload_id = payload_content_id(ElementType::Bytes, &payload);
    let source_content_id = payload_content_id(ElementType::Bytes, anchor.bytes);
    let dataset_id = derive_id::<DatasetTag>(source_hash.as_bytes(), b"dataset");
    let recording_id = derive_id::<RecordingTag>(source_hash.as_bytes(), b"recording");
    let stream_id = derive_id::<StreamTag>(source_hash.as_bytes(), b"stream");
    let atom_id = derive_id::<AtomTag>(source_hash.as_bytes(), b"latent");
    let clock_id = derive_id::<ClockTag>(source_hash.as_bytes(), b"clock");

    let descriptor = PayloadDescriptor::new(
        payload_id,
        payload_bytes,
        ElementType::Bytes,
        ByteOrder::Little,
        vec![payload_bytes],
        Layout::DenseRowMajor,
        Some(concept("legacy:encoding/lmqc-neural-latent")?),
        Some(legacy_media_type(LegacyFormat::Lmqc).to_owned()),
    );
    let atom = Atom::EncodedBlock(abir::EncodedBlock::new(
        atom_id,
        Presence::Present,
        Some(descriptor),
        abir::DecodedSemantics::new(
            concept("abir:atom/signal-block")?,
            ElementType::F32,
            vec![channels, window_samples],
        ),
    ));

    // The container states its decoded rate in whole hertz, so the clock is
    // exact rather than projected.
    let rate = Rational::new(i128::from(container.sample_rate), 1)
        .map_err(|error| LegacyError::SemanticValidation(error.to_string()))?;

    let mut recording = Recording::new(recording_id, vec![stream_id]);
    for (namespace, value) in [
        ("legacy-source-blake3", source_blake3.clone()),
        ("lmqc.payload-kind", container.payload_kind.to_string()),
        (
            "lmqc.latent-shape",
            format!("{}x{}", container.latent_c, container.latent_t),
        ),
    ] {
        recording.add_source_key(
            SourceKey::new(namespace, &value)
                .map_err(|error| LegacyError::SemanticValidation(error.to_string()))?,
        );
    }
    // The montage is the reason this container exists; carry every name and
    // coordinate it declared rather than summarising them.
    if let Some(names) = container.channels.as_ref() {
        for (index, name) in names.iter().enumerate() {
            recording.add_source_key(
                SourceKey::new(format!("lmqc.channel.{index}"), name)
                    .map_err(|error| LegacyError::SemanticValidation(error.to_string()))?,
            );
        }
    }
    if let Some(coords) = container.coords.as_ref() {
        for (index, position) in coords.chunks_exact(3).enumerate() {
            recording.add_source_key(
                SourceKey::new(
                    format!("lmqc.coordinate.{index}"),
                    format!("{},{},{}", position[0], position[1], position[2]),
                )
                .map_err(|error| LegacyError::SemanticValidation(error.to_string()))?,
            );
        }
    }

    let mut draft = DatasetDraft::new(dataset_id);
    for key in anchor.source_keys()? {
        recording.add_source_key(key);
    }
    draft.add_recording(recording);
    draft.add_stream(Stream::new(
        stream_id,
        recording_id,
        modality_concept(None),
        vec![atom_id],
        Some(clock_id),
        None,
        None,
    ));
    draft.add_atom(atom);
    draft.add_clock(Clock::new(
        clock_id,
        concept("abir:clock/device")?,
        None,
        Rational::new(0, 1).expect("zero rational is valid"),
        rate,
        Rational::new(0, 1).expect("zero uncertainty is valid"),
    ));
    // Exact: the ENCODED payload is carried byte-for-byte. This is not a claim
    // about the samples it would decode to.
    draft.add_fidelity(Fidelity::new(
        SemanticRef::of(atom_id),
        FidelityKind::Exact,
        None,
        None,
    ));
    draft.add_source_capsule(SourceCapsule::new(
        SourceKey::new("legacy-source-blake3", &source_blake3)
            .map_err(|error| LegacyError::SemanticValidation(error.to_string()))?,
        source_content_id,
        Some(legacy_media_type(anchor.format)),
    ));
    let dataset = draft
        .validate(ValidationLimits {
            max_logical_payload_bytes: max_decoded_bytes,
            ..ValidationLimits::default()
        })
        .map_err(|report| LegacyError::SemanticValidation(format!("{report:?}")))?;
    let dataset_json = canonical_debug_json(&dataset)
        .map_err(|error| LegacyError::SemanticValidation(error.to_string()))?;
    let dataset_content_id = logical_content_id(&dataset)
        .map_err(|error| LegacyError::SemanticValidation(error.to_string()))?;

    let mapping = MappingReport {
        source_profile: ProfileId(anchor.format.profile().to_owned()),
        target_profile: ProfileId("abir.semantic-v1".to_owned()),
        semantic_coverage: SemanticCoverage::ProjectedSemantic,
        entries: vec![
            MappingEntry {
                source_path: "wire.montage".to_owned(),
                target: format!("recording:{recording_id}:source-key"),
                disposition: MappingDisposition::Exact,
                reason: None,
            },
            MappingEntry {
                source_path: "wire.sample-rate".to_owned(),
                target: format!("clock:{clock_id}"),
                disposition: MappingDisposition::Exact,
                reason: None,
            },
            MappingEntry {
                source_path: "wire.payload".to_owned(),
                target: format!("atom:{atom_id}"),
                disposition: MappingDisposition::Quarantined,
                reason: Some(
                    "the neural latent stays encoded: decoding it needs the neural decoder and \
                     its weights, which this adapter does not carry"
                        .to_owned(),
                ),
            },
        ],
        preserved_unknowns: 1,
        sample_values_changed: false,
        timing_changed: false,
    };
    let mut mapping = mapping;
    mapping.entries.extend(anchor.extra_mapping.iter().cloned());
    mapping.preserved_unknowns += anchor.extra_mapping.len() as u64;
    let mapping_report = serde_json::to_vec_pretty(&mapping)
        .map_err(|error| LegacyError::InvalidProtocol(error.to_string()))?;
    let fidelity = SemanticFidelityReport {
        schema: "lamquant.legacy-fidelity/v1".to_owned(),
        source_profile: anchor.format.profile().to_owned(),
        exact_source_restoration: true,
        // No sample values were produced, so none can be claimed exact.
        exact_sample_values: false,
        sample_values_changed: false,
        timing_equivalence: true,
        modality_equivalence: false,
        semantic_equivalence: false,
        source_capsule_file: SOURCE_FILE.to_owned(),
        caveats: vec![
            "the payload remains an encoded neural latent; no sample-domain signal is produced"
                .to_owned(),
            "LMQC records no biosignal modality, so the stream modality is unknown-at-source"
                .to_owned(),
        ],
    };
    let mut fidelity = fidelity;
    fidelity
        .caveats
        .extend(anchor.extra_caveats.iter().cloned());
    let fidelity_report = serde_json::to_vec_pretty(&fidelity)
        .map_err(|error| LegacyError::InvalidProtocol(error.to_string()))?;
    let receipt = SemanticImportReceipt {
        profile: anchor.format.profile().to_owned(),
        source_blake3,
        source_bytes: anchor.bytes.len() as u64,
        decoded_channels: channels,
        decoded_samples_per_channel: window_samples,
        decoded_payload_bytes: payload_bytes,
        dataset_content_id: dataset_content_id.to_string(),
        payload_content_id: payload_id.to_string(),
        source_preserved: true,
        exact_sample_values: false,
        exact_source_restoration: true,
        semantic_equivalence: false,
        timing: "regular".to_owned(),
        modality: "legacy:modality/unknown-at-source".to_owned(),
        semantic_coverage: "projected-semantic".to_owned(),
    };
    let semantic_receipt = serde_json::to_vec_pretty(&receipt)
        .map_err(|error| LegacyError::InvalidProtocol(error.to_string()))?;
    Ok(SemanticArtifacts {
        receipt,
        dataset_json,
        payloads: vec![(LMQC_PAYLOAD_FILE.to_owned(), payload)],
        mapping_report,
        fidelity_report,
        semantic_receipt,
    })
}

fn decode_source<'a>(source: &'a [u8], format: LegacyFormat) -> Result<Cow<'a, [u8]>, LegacyError> {
    if format != LegacyFormat::Bcs1 {
        return Ok(Cow::Borrowed(source));
    }
    let header = Bcs1Header::parse(source)
        .map_err(|error| LegacyError::MalformedContainer(error.to_string()))?;
    Ok(Cow::Owned(bcs1_as_lml1(source, &header)?))
}

fn bcs1_as_lml1(source: &[u8], header: &Bcs1Header) -> Result<Vec<u8>, LegacyError> {
    if source.len() < 40 {
        return Err(LegacyError::MalformedContainer(
            "BCS1 source is shorter than its fixed header".to_owned(),
        ));
    }
    let capacity = source
        .len()
        .checked_sub(8)
        .ok_or_else(|| LegacyError::MalformedContainer("BCS1 size underflow".to_owned()))?;
    let mut translated = Vec::with_capacity(capacity);
    translated.extend_from_slice(b"LML1");
    translated.extend_from_slice(&1_u16.to_le_bytes());
    translated.extend_from_slice(&header.n_channels.to_le_bytes());
    translated.extend_from_slice(&header.n_windows.to_le_bytes());
    translated.extend_from_slice(&header.total_samples.to_le_bytes());
    translated.extend_from_slice(&header.window_size.to_le_bytes());
    translated.extend_from_slice(&header.sample_rate_mhz.to_le_bytes());
    translated.push(header.bit_depth);
    translated.push(header.flags);
    translated.extend_from_slice(&header.metadata_length.to_le_bytes());
    translated.extend_from_slice(&[0_u8; 6]);
    translated.extend_from_slice(&source[40..]);
    Ok(translated)
}

fn sample_rate_millihz(bytes: &[u8], format: LegacyFormat, metadata: &str) -> Option<u32> {
    let header_value = match format {
        LegacyFormat::Bcs1 if bytes.len() >= 26 => Some(u32::from_le_bytes([
            bytes[22], bytes[23], bytes[24], bytes[25],
        ])),
        LegacyFormat::Lml1
            if bytes.len() >= 32
                && u16::from_le_bytes([bytes[4], bytes[5]]) == 1
                && matches!(bytes[20], 16 | 24 | 32) =>
        {
            Some(u32::from_le_bytes([
                bytes[16], bytes[17], bytes[18], bytes[19],
            ]))
        }
        _ => None,
    }
    .filter(|value| *value > 0);
    header_value.or_else(|| {
        let value = serde_json::from_str::<Value>(metadata)
            .ok()?
            .get("sample_rate")?
            .as_f64()?;
        if !value.is_finite() || value <= 0.0 || value > f64::from(u32::MAX) / 1000.0 {
            return None;
        }
        let millihz = value * 1000.0;
        let rounded = millihz.round();
        ((millihz - rounded).abs() <= 1e-9).then_some(rounded as u32)
    })
}

fn verify_decoded_shape(signal: &[Vec<i64>], facts: &ContainerFacts) -> Result<(), LegacyError> {
    if u64::try_from(signal.len()).ok() != Some(facts.channels)
        || signal
            .iter()
            .any(|channel| u64::try_from(channel.len()).ok() != Some(facts.samples_per_channel))
    {
        return Err(LegacyError::MalformedContainer(
            "decoder output shape does not match the validated container header".to_owned(),
        ));
    }
    Ok(())
}

/// Project an LMA archive to one ABIR dataset holding one recording per
/// contained LML entry.
///
/// An archive is a directory tree, not a single signal, so it is never
/// flattened into one synthetic recording: each LML entry contributes its own
/// Recording/Stream/Atom and its own payload file. Entries the archive stored
/// or zstd-compressed verbatim (`Method::Zstd` / `Method::Store`) are NOT
/// signals — they are counted and named in the mapping report and remain
/// byte-exact in the source capsule, but they are never promoted to recordings.
fn build_lma_artifacts(
    source: &[u8],
    anchor: &SourceAnchor<'_>,
    max_decoded_bytes: u64,
) -> Result<SemanticArtifacts, LegacyError> {
    use lamquant_lml_archive::lma;

    verify_archive_digest(source)?;
    // The archive readers are path-based; stage the in-memory source once.
    let mut temporary = tempfile::NamedTempFile::new().map_err(io_error)?;
    temporary.write_all(source).map_err(io_error)?;
    temporary.flush().map_err(io_error)?;
    let archive_path = temporary.path();
    let entries = lma::list_archive(archive_path)
        .map_err(|error| LegacyError::MalformedContainer(error.to_string()))?;

    let source_hash = blake3::hash(anchor.bytes);
    let source_blake3 = source_hash.to_hex().to_string();
    let dataset_id = derive_id::<DatasetTag>(source_hash.as_bytes(), b"dataset");
    let source_content_id = payload_content_id(ElementType::Bytes, anchor.bytes);
    let mut draft = DatasetDraft::new(dataset_id);

    let mut payloads: Vec<(String, Vec<u8>)> = Vec::new();
    let mut total_payload_bytes: u64 = 0;
    let mut total_channels: u64 = 0;
    let mut total_samples: u64 = 0;
    let mut signal_entries: u64 = 0;
    let mut sibling_entries: u64 = 0;
    let mut first_payload_id: Option<String> = None;
    let mut timing_known = true;

    for (index, entry) in entries.iter().enumerate() {
        if entry.method != lma::Method::Lml {
            sibling_entries += 1;
            continue;
        }
        let stored = lma::read_entry(archive_path, &entry.path)
            .map_err(|error| LegacyError::MalformedContainer(error.to_string()))?;
        // Archived signal entries may be either container generation (the codec
        // emits BCS1 today, older archives hold LML1), so normalise to the LML1
        // wire the legacy decoder reads rather than assuming one of them.
        let entry_format = detect_format(&stored)?;
        if !matches!(entry_format, LegacyFormat::Bcs1 | LegacyFormat::Lml1) {
            return Err(LegacyError::MalformedContainer(format!(
                "archive signal entry {} is not an LML-family container",
                entry.path
            )));
        }
        let encoded = decode_source(&stored, entry_format)?.into_owned();
        let header = lamquant_lml_legacy::container::parse_header(&encoded)
            .map_err(|error| LegacyError::MalformedContainer(error.to_string()))?;
        let channels = u64::try_from(header.n_ch).map_err(|_| LegacyError::DecodedTooLarge)?;
        let samples =
            u64::try_from(header.total_samples).map_err(|_| LegacyError::DecodedTooLarge)?;
        let entry_bytes = channels
            .checked_mul(samples)
            .and_then(|value| value.checked_mul(8))
            .ok_or(LegacyError::DecodedTooLarge)?;
        total_payload_bytes = total_payload_bytes
            .checked_add(entry_bytes)
            .ok_or(LegacyError::DecodedTooLarge)?;
        if total_payload_bytes > max_decoded_bytes {
            return Err(LegacyError::DecodedTooLarge);
        }

        let signal = lamquant_lml_legacy::container::read_bytes(&encoded)
            .map_err(|error| LegacyError::MalformedContainer(error.to_string()))?
            .0;
        let mut payload = Vec::with_capacity(
            usize::try_from(entry_bytes).map_err(|_| LegacyError::DecodedTooLarge)?,
        );
        for channel in signal {
            for sample in channel {
                payload.extend_from_slice(&sample.to_le_bytes());
            }
        }
        if u64::try_from(payload.len()).ok() != Some(entry_bytes) {
            return Err(LegacyError::MalformedContainer(
                "archive entry decoded to a shape its header did not declare".to_owned(),
            ));
        }

        // Per-entry identities are derived from the entry's own manifest digest
        // so two entries never collide and the ids are stable across runs.
        let mut entry_hasher = blake3::Hasher::new();
        entry_hasher.update(source_hash.as_bytes());
        entry_hasher.update(b"\0lma-entry\0");
        entry_hasher.update(entry.path.as_bytes());
        entry_hasher.update(b"\0");
        entry_hasher.update(entry.sha256.as_bytes());
        let entry_seed = *entry_hasher.finalize().as_bytes();
        let recording_id = derive_id::<RecordingTag>(&entry_seed, b"recording");
        let stream_id = derive_id::<StreamTag>(&entry_seed, b"stream");
        let atom_id = derive_id::<AtomTag>(&entry_seed, b"signal");
        let clock_id = derive_id::<ClockTag>(&entry_seed, b"clock");

        let payload_id = payload_content_id(ElementType::I64, &payload);
        if first_payload_id.is_none() {
            first_payload_id = Some(payload_id.to_string());
        }
        let descriptor = PayloadDescriptor::new(
            payload_id,
            entry_bytes,
            ElementType::I64,
            ByteOrder::Little,
            vec![channels, samples],
            Layout::DenseRowMajor,
            Some(concept("abir:encoding/raw")?),
            Some("application/vnd.abir.i64le".to_owned()),
        );
        let rate_millihz = sample_rate_millihz(&encoded, entry_format, &header.metadata);
        if rate_millihz.is_none() {
            timing_known = false;
        }
        let (atom, stream_clock) = if let Some(millihz) = rate_millihz {
            let rate = Rational::new(i128::from(millihz), 1000)
                .map_err(|error| LegacyError::SemanticValidation(error.to_string()))?;
            let segment = TimeSegment::new(
                Rational::new(0, 1).expect("zero rational is valid"),
                rate,
                samples,
            )
            .map_err(|error| LegacyError::SemanticValidation(error.to_string()))?;
            (
                Atom::SignalBlock(SignalBlock::new(
                    atom_id,
                    Presence::Present,
                    Some(descriptor),
                    TimeAxis::Regular(segment),
                    None,
                )),
                Some((clock_id, rate)),
            )
        } else {
            (
                Atom::Tensor(Tensor::new(
                    atom_id,
                    Presence::Present,
                    Some(descriptor),
                    vec![
                        SemanticAxis::new(concept("abir:axis/channel")?, channels),
                        SemanticAxis::new(concept("abir:axis/sample")?, samples),
                    ],
                )),
                None,
            )
        };

        let mut recording = Recording::new(recording_id, vec![stream_id]);
        recording.add_source_key(
            SourceKey::new("legacy-lma-entry-sha256", &entry.sha256)
                .map_err(|error| LegacyError::SemanticValidation(error.to_string()))?,
        );
        for key in anchor.source_keys()? {
            recording.add_source_key(key);
        }
        draft.add_recording(recording);
        draft.add_stream(Stream::new(
            stream_id,
            recording_id,
            modality_concept(None),
            vec![atom_id],
            stream_clock.map(|(id, _)| id),
            None,
            None,
        ));
        draft.add_atom(atom);
        if let Some((id, rate)) = stream_clock {
            draft.add_clock(Clock::new(
                id,
                concept("abir:clock/device")?,
                None,
                Rational::new(0, 1).expect("zero rational is valid"),
                rate,
                Rational::new(0, 1).expect("zero uncertainty is valid"),
            ));
        }
        draft.add_fidelity(Fidelity::new(
            SemanticRef::of(atom_id),
            FidelityKind::Exact,
            None,
            None,
        ));

        payloads.push((format!("payload-{index:04}.i64le"), payload));
        signal_entries += 1;
        total_channels = total_channels
            .checked_add(channels)
            .ok_or(LegacyError::DecodedTooLarge)?;
        total_samples = total_samples
            .checked_add(samples)
            .ok_or(LegacyError::DecodedTooLarge)?;
    }

    if signal_entries == 0 {
        return Err(LegacyError::MalformedContainer(
            "LMA archive contains no LML entry to import semantically".to_owned(),
        ));
    }

    draft.add_source_capsule(SourceCapsule::new(
        SourceKey::new("legacy-source-blake3", &source_blake3)
            .map_err(|error| LegacyError::SemanticValidation(error.to_string()))?,
        source_content_id,
        Some(legacy_media_type(anchor.format)),
    ));
    let limits = ValidationLimits {
        max_logical_payload_bytes: max_decoded_bytes,
        ..ValidationLimits::default()
    };
    let dataset = draft
        .validate(limits)
        .map_err(|report| LegacyError::SemanticValidation(format!("{report:?}")))?;
    let dataset_json = canonical_debug_json(&dataset)
        .map_err(|error| LegacyError::SemanticValidation(error.to_string()))?;
    let dataset_content_id = logical_content_id(&dataset)
        .map_err(|error| LegacyError::SemanticValidation(error.to_string()))?;

    let mapping = MappingReport {
        source_profile: ProfileId(anchor.format.profile().to_owned()),
        target_profile: ProfileId("abir.semantic-v1".to_owned()),
        semantic_coverage: SemanticCoverage::ProjectedSemantic,
        entries: vec![
            MappingEntry {
                source_path: "archive.lml-entries".to_owned(),
                target: "abir.dataset.recordings".to_owned(),
                disposition: MappingDisposition::Exact,
                reason: None,
            },
            MappingEntry {
                source_path: "archive.entry.sample-rate".to_owned(),
                target: "abir.stream.atom.time-axis".to_owned(),
                disposition: if timing_known {
                    MappingDisposition::Exact
                } else {
                    MappingDisposition::Unsupported
                },
                reason: (!timing_known).then(|| {
                    "at least one archive entry carries no exact positive sample rate".to_owned()
                }),
            },
            MappingEntry {
                source_path: "archive.non-signal-entries".to_owned(),
                target: "abir.source-capsule".to_owned(),
                disposition: MappingDisposition::Quarantined,
                reason: Some(
                    "stored and zstd entries are preserved byte-exact and are never promoted to recordings"
                        .to_owned(),
                ),
            },
        ],
        preserved_unknowns: sibling_entries,
        sample_values_changed: false,
        timing_changed: false,
    };
    let mut mapping = mapping;
    mapping.entries.extend(anchor.extra_mapping.iter().cloned());
    mapping.preserved_unknowns += anchor.extra_mapping.len() as u64;
    let mapping_report = serde_json::to_vec_pretty(&mapping)
        .map_err(|error| LegacyError::InvalidProtocol(error.to_string()))?;
    let fidelity = SemanticFidelityReport {
        schema: "lamquant.legacy-fidelity/v1".to_owned(),
        source_profile: anchor.format.profile().to_owned(),
        exact_source_restoration: true,
        exact_sample_values: true,
        sample_values_changed: false,
        timing_equivalence: timing_known,
        modality_equivalence: false,
        semantic_equivalence: false,
        source_capsule_file: SOURCE_FILE.to_owned(),
        caveats: vec![
            format!(
                "archive holds {signal_entries} LML recording(s) and {sibling_entries} \
                 non-signal entr(y/ies); the flat receipt counts are archive-wide sums, \
                 not one recording's shape"
            ),
            "non-signal entries stay byte-exact in the source capsule and are never \
             promoted to recordings"
                .to_owned(),
            "legacy archive metadata remains quarantined in the exact source capsule".to_owned(),
        ],
    };
    let mut fidelity = fidelity;
    fidelity
        .caveats
        .extend(anchor.extra_caveats.iter().cloned());
    let fidelity_report = serde_json::to_vec_pretty(&fidelity)
        .map_err(|error| LegacyError::InvalidProtocol(error.to_string()))?;
    let receipt = SemanticImportReceipt {
        profile: anchor.format.profile().to_owned(),
        source_blake3,
        source_bytes: anchor.bytes.len() as u64,
        decoded_channels: total_channels,
        decoded_samples_per_channel: total_samples,
        decoded_payload_bytes: total_payload_bytes,
        dataset_content_id: dataset_content_id.to_string(),
        payload_content_id: first_payload_id.ok_or_else(|| {
            LegacyError::MalformedContainer("archive produced no payload".to_owned())
        })?,
        source_preserved: true,
        exact_sample_values: true,
        exact_source_restoration: true,
        semantic_equivalence: false,
        timing: if timing_known {
            "regular".to_owned()
        } else {
            "unknown-at-source".to_owned()
        },
        modality: "legacy:modality/unknown-at-source".to_owned(),
        semantic_coverage: "projected-semantic".to_owned(),
    };
    let semantic_receipt = serde_json::to_vec_pretty(&receipt)
        .map_err(|error| LegacyError::InvalidProtocol(error.to_string()))?;
    Ok(SemanticArtifacts {
        receipt,
        dataset_json,
        payloads,
        mapping_report,
        fidelity_report,
        semantic_receipt,
    })
}

fn build_lqtp1_artifacts(
    source: &[u8],
    anchor: &SourceAnchor<'_>,
    max_decoded_bytes: u64,
) -> Result<SemanticArtifacts, LegacyError> {
    parse_lqtp1_header(source)?;
    let mut temporary = tempfile::NamedTempFile::new().map_err(io_error)?;
    temporary.write_all(source).map_err(io_error)?;
    temporary.flush().map_err(io_error)?;
    let reader = lamquant_lml_archive::tensor_pack::PackReader::open(temporary.path(), None)
        .map_err(|error| LegacyError::MalformedContainer(error.to_string()))?;
    let header = reader.header();
    let decoded_elements = header
        .n_windows
        .checked_mul(header.n_channels)
        .and_then(|value| value.checked_mul(header.window_len))
        .ok_or(LegacyError::DecodedTooLarge)?;
    let decoded_payload_bytes = decoded_elements
        .checked_mul(4)
        .and_then(|value| u64::try_from(value).ok())
        .ok_or(LegacyError::DecodedTooLarge)?;
    if decoded_payload_bytes > max_decoded_bytes {
        return Err(LegacyError::DecodedTooLarge);
    }
    let mut payload = Vec::with_capacity(decoded_payload_bytes as usize);
    for row in 0..reader.n_windows() {
        for value in reader
            .dequantize_window(row)
            .map_err(|error| LegacyError::MalformedContainer(error.to_string()))?
        {
            if !value.is_finite() {
                return Err(LegacyError::MalformedContainer(
                    "LQTP1 contains a non-finite decoded sample".to_owned(),
                ));
            }
            payload.extend_from_slice(&value.to_le_bytes());
        }
    }

    let source_hash = blake3::hash(anchor.bytes);
    let source_blake3 = source_hash.to_hex().to_string();
    let payload_id = payload_content_id(ElementType::F32, &payload);
    let source_content_id = payload_content_id(ElementType::Bytes, anchor.bytes);
    let dataset_id = derive_id::<DatasetTag>(source_hash.as_bytes(), b"dataset");
    let recording_id = derive_id::<RecordingTag>(source_hash.as_bytes(), b"recording");
    let stream_id = derive_id::<StreamTag>(source_hash.as_bytes(), b"stream");
    let atom_id = derive_id::<AtomTag>(source_hash.as_bytes(), b"tensor");
    let windows = u64::try_from(header.n_windows).map_err(|_| LegacyError::DecodedTooLarge)?;
    let channels = u64::try_from(header.n_channels).map_err(|_| LegacyError::DecodedTooLarge)?;
    let samples = u64::try_from(header.window_len).map_err(|_| LegacyError::DecodedTooLarge)?;
    let descriptor = PayloadDescriptor::new(
        payload_id,
        decoded_payload_bytes,
        ElementType::F32,
        ByteOrder::Little,
        vec![windows, channels, samples],
        Layout::DenseRowMajor,
        Some(concept("abir:encoding/raw")?),
        Some("application/vnd.abir.f32le".to_owned()),
    );
    let tensor = Atom::Tensor(Tensor::new(
        atom_id,
        Presence::Present,
        Some(descriptor),
        vec![
            SemanticAxis::new(concept("abir:axis/window")?, windows),
            SemanticAxis::new(concept("abir:axis/channel")?, channels),
            SemanticAxis::new(concept("abir:axis/sample")?, samples),
        ],
    ));
    let manifest_sha256 = header
        .manifest_sha256
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let mut recording = Recording::new(recording_id, vec![stream_id]);
    for (namespace, value) in [
        ("legacy-source-blake3", source_blake3.as_str()),
        ("lqtp1.manifest-sha256", manifest_sha256.as_str()),
    ] {
        recording.add_source_key(
            SourceKey::new(namespace, value)
                .map_err(|error| LegacyError::SemanticValidation(error.to_string()))?,
        );
    }
    recording.add_source_key(
        SourceKey::new("lqtp1.dtype", header.dtype.to_u8().to_string())
            .map_err(|error| LegacyError::SemanticValidation(error.to_string()))?,
    );
    let mut draft = DatasetDraft::new(dataset_id);
    for key in anchor.source_keys()? {
        recording.add_source_key(key);
    }
    draft.add_recording(recording);
    draft.add_stream(Stream::new(
        stream_id,
        recording_id,
        concept("legacy:modality/unknown-at-source")?,
        vec![atom_id],
        None,
        None,
        None,
    ));
    draft.add_atom(tensor);
    draft.add_fidelity(Fidelity::new(
        SemanticRef::of(atom_id),
        FidelityKind::Exact,
        None,
        None,
    ));
    draft.add_source_capsule(SourceCapsule::new(
        SourceKey::new("legacy-source-blake3", &source_blake3)
            .map_err(|error| LegacyError::SemanticValidation(error.to_string()))?,
        source_content_id,
        Some(legacy_media_type(anchor.format)),
    ));
    let dataset = draft
        .validate(ValidationLimits {
            max_logical_payload_bytes: max_decoded_bytes,
            ..ValidationLimits::default()
        })
        .map_err(|report| LegacyError::SemanticValidation(format!("{report:?}")))?;
    let dataset_json = canonical_debug_json(&dataset)
        .map_err(|error| LegacyError::SemanticValidation(error.to_string()))?;
    let dataset_content_id = logical_content_id(&dataset)
        .map_err(|error| LegacyError::SemanticValidation(error.to_string()))?;
    let mapping = MappingReport {
        source_profile: ProfileId(anchor.format.profile().to_owned()),
        target_profile: ProfileId("abir.semantic-v1".to_owned()),
        semantic_coverage: SemanticCoverage::ProjectedSemantic,
        entries: vec![
            MappingEntry {
                source_path: "wire.windows".to_owned(),
                target: format!("atom:{atom_id}"),
                disposition: MappingDisposition::Exact,
                reason: None,
            },
            MappingEntry {
                source_path: "wire.manifest-sha256".to_owned(),
                target: format!("recording:{recording_id}:source-key"),
                disposition: MappingDisposition::Exact,
                reason: None,
            },
            MappingEntry {
                source_path: "wire.window-identities".to_owned(),
                target: "abir.source-capsule".to_owned(),
                disposition: MappingDisposition::Quarantined,
                reason: Some(
                    "LQTP1 binds an external ordered manifest by hash but does not carry row identities"
                        .to_owned(),
                ),
            },
        ],
        preserved_unknowns: 1,
        sample_values_changed: false,
        timing_changed: false,
    };
    let mut mapping = mapping;
    mapping.entries.extend(anchor.extra_mapping.iter().cloned());
    mapping.preserved_unknowns += anchor.extra_mapping.len() as u64;
    let mapping_report = serde_json::to_vec_pretty(&mapping)
        .map_err(|error| LegacyError::InvalidProtocol(error.to_string()))?;
    let fidelity = SemanticFidelityReport {
        schema: "lamquant.legacy-fidelity/v1".to_owned(),
        source_profile: anchor.format.profile().to_owned(),
        exact_source_restoration: true,
        exact_sample_values: true,
        sample_values_changed: false,
        timing_equivalence: false,
        modality_equivalence: false,
        semantic_equivalence: false,
        source_capsule_file: SOURCE_FILE.to_owned(),
        caveats: vec![
            "LQTP1 carries no sampling clock or biosignal modality".to_owned(),
            "row identities remain external and are bound only by manifest SHA-256".to_owned(),
        ],
    };
    let mut fidelity = fidelity;
    fidelity
        .caveats
        .extend(anchor.extra_caveats.iter().cloned());
    let fidelity_report = serde_json::to_vec_pretty(&fidelity)
        .map_err(|error| LegacyError::InvalidProtocol(error.to_string()))?;
    let receipt = SemanticImportReceipt {
        profile: anchor.format.profile().to_owned(),
        source_blake3,
        source_bytes: anchor.bytes.len() as u64,
        decoded_channels: channels,
        decoded_samples_per_channel: windows
            .checked_mul(samples)
            .ok_or(LegacyError::DecodedTooLarge)?,
        decoded_payload_bytes,
        dataset_content_id: dataset_content_id.to_string(),
        payload_content_id: payload_id.to_string(),
        source_preserved: true,
        exact_sample_values: true,
        exact_source_restoration: true,
        semantic_equivalence: false,
        timing: "unknown-at-source".to_owned(),
        modality: "legacy:modality/unknown-at-source".to_owned(),
        semantic_coverage: "projected-semantic".to_owned(),
    };
    let semantic_receipt = serde_json::to_vec_pretty(&receipt)
        .map_err(|error| LegacyError::InvalidProtocol(error.to_string()))?;
    Ok(SemanticArtifacts {
        receipt,
        dataset_json,
        payloads: vec![(LQTP1_PAYLOAD_FILE.to_owned(), payload)],
        mapping_report,
        fidelity_report,
        semantic_receipt,
    })
}

fn parse_lqtp1_header(
    source: &[u8],
) -> Result<lamquant_lml_archive::tensor_pack::PackHeader, LegacyError> {
    let header = lamquant_lml_archive::tensor_pack::PackHeader::parse(source)
        .map_err(|error| LegacyError::MalformedContainer(error.to_string()))?;
    let total_len = header
        .n_windows
        .checked_mul(header.record_stride)
        .and_then(|records| records.checked_add(lamquant_lml_archive::tensor_pack::LQTP_HEADER_LEN))
        .ok_or(LegacyError::DecodedTooLarge)?;
    if total_len != source.len() {
        return Err(LegacyError::MalformedContainer(
            "LQTP1 source has trailing or truncated bytes".to_owned(),
        ));
    }
    Ok(header)
}

fn build_semantic_artifacts(
    anchor: &SourceAnchor<'_>,
    facts: &ContainerFacts,
    signal: Vec<Vec<i64>>,
    max_decoded_bytes: u64,
) -> Result<SemanticArtifacts, LegacyError> {
    let source_hash = blake3::hash(anchor.bytes);
    let source_blake3 = source_hash.to_hex().to_string();
    let mut payload = Vec::with_capacity(
        usize::try_from(facts.decoded_payload_bytes).map_err(|_| LegacyError::DecodedTooLarge)?,
    );
    for channel in signal {
        for sample in channel {
            payload.extend_from_slice(&sample.to_le_bytes());
        }
    }
    if u64::try_from(payload.len()).ok() != Some(facts.decoded_payload_bytes) {
        return Err(LegacyError::DecodedTooLarge);
    }
    let payload_id = payload_content_id(ElementType::I64, &payload);
    let source_content_id = payload_content_id(ElementType::Bytes, anchor.bytes);
    let dataset_id = derive_id::<DatasetTag>(source_hash.as_bytes(), b"dataset");
    let recording_id = derive_id::<RecordingTag>(source_hash.as_bytes(), b"recording");
    let stream_id = derive_id::<StreamTag>(source_hash.as_bytes(), b"stream");
    let atom_id = derive_id::<AtomTag>(source_hash.as_bytes(), b"signal");
    let clock_id = derive_id::<ClockTag>(source_hash.as_bytes(), b"clock");
    let modality = modality_concept(facts.modality_tag);

    let descriptor = PayloadDescriptor::new(
        payload_id,
        facts.decoded_payload_bytes,
        ElementType::I64,
        ByteOrder::Little,
        vec![facts.channels, facts.samples_per_channel],
        Layout::DenseRowMajor,
        Some(concept("abir:encoding/raw")?),
        Some("application/vnd.abir.i64le".to_owned()),
    );
    let (atom, stream_clock) = if let Some(millihz) = facts.sample_rate_millihz {
        let rate = Rational::new(i128::from(millihz), 1000)
            .map_err(|error| LegacyError::SemanticValidation(error.to_string()))?;
        let segment = TimeSegment::new(
            Rational::new(0, 1).expect("zero rational is valid"),
            rate,
            facts.samples_per_channel,
        )
        .map_err(|error| LegacyError::SemanticValidation(error.to_string()))?;
        (
            Atom::SignalBlock(SignalBlock::new(
                atom_id,
                Presence::Present,
                Some(descriptor),
                TimeAxis::Regular(segment),
                None,
            )),
            Some((clock_id, rate)),
        )
    } else {
        (
            Atom::Tensor(Tensor::new(
                atom_id,
                Presence::Present,
                Some(descriptor),
                vec![
                    SemanticAxis::new(concept("abir:axis/channel")?, facts.channels),
                    SemanticAxis::new(concept("abir:axis/sample")?, facts.samples_per_channel),
                ],
            )),
            None,
        )
    };

    let mut recording = Recording::new(recording_id, vec![stream_id]);
    recording.add_source_key(
        SourceKey::new("legacy-source-blake3", &source_blake3)
            .map_err(|error| LegacyError::SemanticValidation(error.to_string()))?,
    );
    let mut draft = DatasetDraft::new(dataset_id);
    for key in anchor.source_keys()? {
        recording.add_source_key(key);
    }
    draft.add_recording(recording);
    draft.add_stream(Stream::new(
        stream_id,
        recording_id,
        modality.clone(),
        vec![atom_id],
        stream_clock.map(|(id, _)| id),
        None,
        None,
    ));
    draft.add_atom(atom);
    if let Some((id, rate)) = stream_clock {
        draft.add_clock(Clock::new(
            id,
            concept("abir:clock/device")?,
            None,
            Rational::new(0, 1).expect("zero rational is valid"),
            rate,
            Rational::new(0, 1).expect("zero uncertainty is valid"),
        ));
    }
    draft.add_fidelity(Fidelity::new(
        SemanticRef::of(atom_id),
        FidelityKind::Exact,
        None,
        None,
    ));
    draft.add_source_capsule(SourceCapsule::new(
        SourceKey::new("legacy-source-blake3", &source_blake3)
            .map_err(|error| LegacyError::SemanticValidation(error.to_string()))?,
        source_content_id,
        Some(legacy_media_type(anchor.format)),
    ));
    let limits = ValidationLimits {
        max_logical_payload_bytes: max_decoded_bytes,
        ..ValidationLimits::default()
    };
    let dataset = draft
        .validate(limits)
        .map_err(|report| LegacyError::SemanticValidation(format!("{report:?}")))?;
    let dataset_json = canonical_debug_json(&dataset)
        .map_err(|error| LegacyError::SemanticValidation(error.to_string()))?;
    let dataset_content_id = logical_content_id(&dataset)
        .map_err(|error| LegacyError::SemanticValidation(error.to_string()))?;

    let timing_equivalence = facts.sample_rate_millihz.is_some();
    let modality_equivalence = facts.modality_tag.is_some();
    let mapping = mapping_report(
        &ProfileId(anchor.format.profile().to_owned()),
        facts,
        timing_equivalence,
        modality_equivalence,
    );
    let mut mapping = mapping;
    mapping.entries.extend(anchor.extra_mapping.iter().cloned());
    mapping.preserved_unknowns += anchor.extra_mapping.len() as u64;
    let mapping_report = serde_json::to_vec_pretty(&mapping)
        .map_err(|error| LegacyError::InvalidProtocol(error.to_string()))?;
    let fidelity = SemanticFidelityReport {
        schema: "lamquant.legacy-fidelity/v1".to_owned(),
        source_profile: anchor.format.profile().to_owned(),
        exact_source_restoration: true,
        exact_sample_values: true,
        sample_values_changed: false,
        timing_equivalence,
        modality_equivalence,
        semantic_equivalence: false,
        source_capsule_file: SOURCE_FILE.to_owned(),
        caveats: vec![
            "legacy metadata remains quarantined in the exact source capsule".to_owned(),
            "semantic coverage is projected until every legacy metadata field has a pinned mapping"
                .to_owned(),
        ],
    };
    let mut fidelity = fidelity;
    fidelity
        .caveats
        .extend(anchor.extra_caveats.iter().cloned());
    let fidelity_report = serde_json::to_vec_pretty(&fidelity)
        .map_err(|error| LegacyError::InvalidProtocol(error.to_string()))?;
    let receipt = SemanticImportReceipt {
        profile: anchor.format.profile().to_owned(),
        source_blake3,
        source_bytes: anchor.bytes.len() as u64,
        decoded_channels: facts.channels,
        decoded_samples_per_channel: facts.samples_per_channel,
        decoded_payload_bytes: facts.decoded_payload_bytes,
        dataset_content_id: dataset_content_id.to_string(),
        payload_content_id: payload_id.to_string(),
        source_preserved: true,
        exact_sample_values: true,
        exact_source_restoration: true,
        semantic_equivalence: false,
        timing: if timing_equivalence {
            "exact"
        } else {
            "unknown-at-source"
        }
        .to_owned(),
        modality: modality.to_string(),
        semantic_coverage: "projected-semantic".to_owned(),
    };
    let semantic_receipt = serde_json::to_vec_pretty(&receipt)
        .map_err(|error| LegacyError::InvalidProtocol(error.to_string()))?;
    Ok(SemanticArtifacts {
        receipt,
        dataset_json,
        payloads: vec![(PAYLOAD_FILE.to_owned(), payload)],
        mapping_report,
        fidelity_report,
        semantic_receipt,
    })
}

fn mapping_report(
    profile: &ProfileId,
    facts: &ContainerFacts,
    timing_equivalence: bool,
    modality_equivalence: bool,
) -> MappingReport {
    let mut entries = vec![MappingEntry {
        source_path: "wire.decoded-samples".to_owned(),
        target: "abir.stream.atom.payload".to_owned(),
        disposition: MappingDisposition::Exact,
        reason: None,
    }];
    entries.push(MappingEntry {
        source_path: "wire.sample-rate".to_owned(),
        target: "abir.stream.atom.time-axis".to_owned(),
        disposition: if timing_equivalence {
            MappingDisposition::Exact
        } else {
            MappingDisposition::Unsupported
        },
        reason: (!timing_equivalence)
            .then(|| "legacy source carries no exact positive sample rate".to_owned()),
    });
    entries.push(MappingEntry {
        source_path: "wire.modality".to_owned(),
        target: "abir.stream.modality".to_owned(),
        disposition: if modality_equivalence {
            MappingDisposition::Exact
        } else {
            MappingDisposition::Unsupported
        },
        reason: (!modality_equivalence).then(|| "LML1 carries no modality field".to_owned()),
    });
    if !facts.metadata.trim().is_empty() {
        entries.push(MappingEntry {
            source_path: "wire.metadata-json".to_owned(),
            target: "abir.source-capsule".to_owned(),
            disposition: MappingDisposition::Quarantined,
            reason: Some(
                "unversioned legacy metadata is preserved byte-exact but not trusted as ABIR semantics"
                    .to_owned(),
            ),
        });
    }
    MappingReport {
        source_profile: profile.clone(),
        target_profile: ProfileId("abir.semantic-v1".to_owned()),
        semantic_coverage: SemanticCoverage::ProjectedSemantic,
        entries,
        preserved_unknowns: u64::from(!facts.metadata.trim().is_empty()),
        sample_values_changed: false,
        timing_changed: false,
    }
}

fn modality_concept(tag: Option<u8>) -> ConceptId {
    let value = match tag {
        Some(0) => "abir:modality/eeg".to_owned(),
        Some(1) => "legacy:modality/ieeg".to_owned(),
        Some(2) => "legacy:modality/ecog".to_owned(),
        Some(3) => "legacy:modality/seeg".to_owned(),
        Some(4) => "abir:modality/ecg".to_owned(),
        Some(5) => "abir:modality/emg".to_owned(),
        Some(6) => "abir:modality/eog".to_owned(),
        Some(7) => "legacy:modality/respiration".to_owned(),
        Some(8) => "legacy:modality/acceleration".to_owned(),
        Some(9) => "legacy:modality/other".to_owned(),
        Some(255) => "legacy:modality/untyped".to_owned(),
        Some(value) => format!("legacy:modality/tag-{value}"),
        None => "legacy:modality/unknown-at-source".to_owned(),
    };
    ConceptId::new(value).expect("static legacy modality concepts are canonical")
}

fn concept(value: &str) -> Result<ConceptId, LegacyError> {
    ConceptId::new(value).map_err(|error| LegacyError::SemanticValidation(error.to_string()))
}

fn derive_id<T>(source_hash: &[u8; 32], label: &[u8]) -> ObjectId<T> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"lamquant.legacy.semantic-import.v1\0");
    hasher.update(label);
    hasher.update(&[0]);
    hasher.update(source_hash);
    let hash = hasher.finalize();
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&hash.as_bytes()[..16]);
    ObjectId::from_bytes(bytes)
}

/// Every retired profile gets its OWN media type. A shared fallback would let
/// the capsule-exact export path accept a capsule of the wrong generation,
/// because that path identifies the source wire by its capsule media type.
fn legacy_media_type(format: LegacyFormat) -> &'static str {
    match format {
        LegacyFormat::Bcs1 => "application/vnd.lamquant.bcs1",
        LegacyFormat::Lml1 => "application/vnd.lamquant.lml1",
        LegacyFormat::Lma1 => "application/vnd.lamquant.lma1",
        LegacyFormat::Lma2 => "application/vnd.lamquant.lma2",
        LegacyFormat::Lmqc => "application/vnd.lamquant.lmqc",
        LegacyFormat::Lmlcrypt => "application/vnd.lamquant.lmlcrypt",
        LegacyFormat::Lqtp1 => "application/vnd.lamquant.lqtp1",
        LegacyFormat::Lqtp2 => "application/vnd.lamquant.lqtp2",
        LegacyFormat::Lqtp3 => "application/vnd.lamquant.lqtp3",
    }
}

fn verify_existing_semantic(
    destination: &Path,
    expected_source: &[u8],
    expected: &SemanticArtifacts,
) -> Result<SemanticImportReceipt, LegacyError> {
    let mut expected_files: Vec<(&str, &[u8])> = vec![
        (SOURCE_FILE, expected_source),
        (DATASET_FILE, &expected.dataset_json),
        (MAPPING_REPORT_FILE, &expected.mapping_report),
        (FIDELITY_REPORT_FILE, &expected.fidelity_report),
        (SEMANTIC_RECEIPT_FILE, &expected.semantic_receipt),
    ];
    // Every recording payload must match too, so a destination that dropped or
    // rewrote one entry of a multi-recording archive is still a conflict.
    expected_files.extend(
        expected
            .payloads
            .iter()
            .map(|(name, bytes)| (name.as_str(), bytes.as_slice())),
    );
    let expected_files = expected_files;
    for (name, expected_bytes) in expected_files {
        let actual =
            fs::read(destination.join(name)).map_err(|_| LegacyError::DestinationConflict)?;
        if actual != *expected_bytes {
            return Err(LegacyError::DestinationConflict);
        }
    }
    Ok(expected.receipt.clone())
}

pub fn handle(request: ProcessRequest) -> ProcessResponse {
    let result = match request {
        ProcessRequest::Manifest => return ProcessResponse::OkManifest(capability_manifest()),
        ProcessRequest::Inspect {
            source,
            max_source_bytes,
        } => inspect(&source, max_source_bytes).map(ProcessResponse::OkInspection),
        ProcessRequest::ConvertForensic(request) => {
            convert_forensic(&request).map(ProcessResponse::OkConversion)
        }
        ProcessRequest::ImportSemantic(request) => {
            import_semantic(&request).map(ProcessResponse::OkSemanticImport)
        }
        ProcessRequest::ExportSemantic(request) => {
            export_semantic(&request).map(ProcessResponse::OkSemanticExport)
        }
    };
    result.unwrap_or_else(|error| ProcessResponse::Error {
        code: error.code().to_owned(),
        message: error.to_string(),
    })
}

fn read_bounded(path: &Path, max_source_bytes: u64) -> Result<Vec<u8>, LegacyError> {
    #[cfg(not(unix))]
    {
        let metadata = fs::symlink_metadata(path).map_err(io_error)?;
        if metadata.file_type().is_symlink() {
            return Err(LegacyError::UnsafeSource);
        }
    }
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    }
    let file = options.open(path).map_err(|error| {
        if is_symlink_error(&error) {
            LegacyError::UnsafeSource
        } else {
            io_error(error)
        }
    })?;
    let metadata = file.metadata().map_err(io_error)?;
    if !metadata.file_type().is_file() {
        return Err(LegacyError::UnsafeSource);
    }
    if metadata.len() > max_source_bytes {
        return Err(LegacyError::SourceTooLarge);
    }
    let capacity = usize::try_from(metadata.len().min(max_source_bytes))
        .map_err(|_| LegacyError::SourceTooLarge)?;
    let mut bytes = Vec::with_capacity(capacity);
    let read_limit = max_source_bytes
        .checked_add(1)
        .ok_or(LegacyError::SourceTooLarge)?;
    file.take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(io_error)?;
    if u64::try_from(bytes.len()).map_err(|_| LegacyError::SourceTooLarge)? > max_source_bytes {
        return Err(LegacyError::SourceTooLarge);
    }
    Ok(bytes)
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), LegacyError> {
    let mut options = fs::OpenOptions::new();
    let mut file = options
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(io_error)?;
    file.write_all(bytes).map_err(io_error)?;
    file.sync_all().map_err(io_error)
}

fn verify_existing(
    destination: &Path,
    expected_source: &[u8],
    expected_receipt: &ConvertReceipt,
) -> Result<ConvertReceipt, LegacyError> {
    let source =
        fs::read(destination.join(SOURCE_FILE)).map_err(|_| LegacyError::DestinationConflict)?;
    let receipt: ConvertReceipt = serde_json::from_slice(
        &fs::read(destination.join(RECEIPT_FILE)).map_err(|_| LegacyError::DestinationConflict)?,
    )
    .map_err(|_| LegacyError::DestinationConflict)?;
    if source == expected_source && &receipt == expected_receipt {
        Ok(receipt)
    } else {
        Err(LegacyError::DestinationConflict)
    }
}

fn io_error(error: std::io::Error) -> LegacyError {
    LegacyError::Io(error.to_string())
}

#[cfg(unix)]
fn is_symlink_error(error: &std::io::Error) -> bool {
    error.raw_os_error() == Some(libc::ELOOP)
}

#[cfg(not(unix))]
fn is_symlink_error(_error: &std::io::Error) -> bool {
    false
}
