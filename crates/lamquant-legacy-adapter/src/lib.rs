#![forbid(unsafe_code)]

use abir::{
    canonical_debug_json, logical_content_id, payload_content_id, Atom, AtomTag, ByteOrder, Clock,
    ClockTag, ConceptId, DatasetDraft, DatasetTag, ElementType, Fidelity, FidelityKind, Layout,
    ObjectId, PayloadDescriptor, Presence, Rational, Recording, RecordingTag, SemanticAxis,
    SemanticRef, SignalBlock, SourceCapsule, SourceKey, Stream, StreamTag, Tensor, TimeAxis,
    TimeSegment, ValidationLimits,
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
const MAPPING_REPORT_FILE: &str = "mapping-report.json";
const FIDELITY_REPORT_FILE: &str = "fidelity-report.json";
const SEMANTIC_RECEIPT_FILE: &str = "semantic-receipt.json";

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
        matches!(self, Self::Bcs1 | Self::Lml1)
    }
}

#[derive(Clone, Debug)]
struct ContainerFacts {
    format: LegacyFormat,
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
    payload: Vec<u8>,
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
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "status", content = "value", rename_all = "kebab-case")]
pub enum ProcessResponse {
    OkManifest(CapabilityManifest),
    OkInspection(Inspection),
    OkConversion(ConvertReceipt),
    OkSemanticImport(SemanticImportReceipt),
    Error { code: String, message: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LegacyError {
    UnknownMagic,
    SourceTooLarge,
    AcceptanceRequired,
    SemanticImportUnsupported,
    DecodedTooLarge,
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
            Self::DecodedTooLarge => "decoded-output-too-large",
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
            Self::DecodedTooLarge => {
                formatter.write_str("decoded signal exceeds declared byte limit")
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
    if bytes.starts_with(b"LQTP") && bytes.len() >= 5 {
        return match bytes[4] {
            1 => Ok(LegacyFormat::Lqtp1),
            2 => Ok(LegacyFormat::Lqtp2),
            3 => Ok(LegacyFormat::Lqtp3),
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
                reverse_export: false,
            })
            .collect(),
    }
}

pub fn inspect(source: &Path, max_source_bytes: u64) -> Result<Inspection, LegacyError> {
    let bytes = read_bounded(source, max_source_bytes)?;
    let format = detect_format(&bytes)?;
    let facts = if format.supports_semantic_import() {
        Some(inspect_container(&bytes, format, u64::MAX)?)
    } else {
        None
    };
    Ok(Inspection {
        profile: format.profile().to_owned(),
        source_bytes: bytes.len() as u64,
        source_blake3: blake3::hash(&bytes).to_hex().to_string(),
        semantic_conversion: format.supports_semantic_import(),
        forensic_conversion: true,
        decoded_channels: facts.as_ref().map(|value| value.channels),
        decoded_samples_per_channel: facts.as_ref().map(|value| value.samples_per_channel),
        decoded_payload_bytes: facts.as_ref().map(|value| value.decoded_payload_bytes),
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
    let facts = inspect_container(&source, format, request.max_decoded_bytes)?;
    let decode_source = decode_source(&source, format)?;
    let signal = lamquant_lml_legacy::container::read_bytes(&decode_source)
        .map_err(|error| LegacyError::MalformedContainer(error.to_string()))?
        .0;
    verify_decoded_shape(&signal, &facts)?;
    let artifacts = build_semantic_artifacts(&source, &facts, signal, request.max_decoded_bytes)?;

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
        write_new(&root.join(PAYLOAD_FILE), &artifacts.payload)?;
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
        format,
        channels,
        samples_per_channel,
        decoded_payload_bytes,
        sample_rate_millihz,
        modality_tag,
        metadata: header.metadata,
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

fn build_semantic_artifacts(
    source: &[u8],
    facts: &ContainerFacts,
    signal: Vec<Vec<i64>>,
    max_decoded_bytes: u64,
) -> Result<SemanticArtifacts, LegacyError> {
    let source_hash = blake3::hash(source);
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
    let source_content_id = payload_content_id(ElementType::Bytes, source);
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
        Some(legacy_media_type(facts.format)),
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
    let mapping = mapping_report(facts, timing_equivalence, modality_equivalence);
    let mapping_report = serde_json::to_vec_pretty(&mapping)
        .map_err(|error| LegacyError::InvalidProtocol(error.to_string()))?;
    let fidelity = SemanticFidelityReport {
        schema: "lamquant.legacy-fidelity/v1".to_owned(),
        source_profile: facts.format.profile().to_owned(),
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
    let fidelity_report = serde_json::to_vec_pretty(&fidelity)
        .map_err(|error| LegacyError::InvalidProtocol(error.to_string()))?;
    let receipt = SemanticImportReceipt {
        profile: facts.format.profile().to_owned(),
        source_blake3,
        source_bytes: source.len() as u64,
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
        payload,
        mapping_report,
        fidelity_report,
        semantic_receipt,
    })
}

fn mapping_report(
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
        source_profile: ProfileId(facts.format.profile().to_owned()),
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

fn legacy_media_type(format: LegacyFormat) -> &'static str {
    match format {
        LegacyFormat::Bcs1 => "application/vnd.lamquant.bcs1",
        LegacyFormat::Lml1 => "application/vnd.lamquant.lml1",
        _ => "application/octet-stream",
    }
}

fn verify_existing_semantic(
    destination: &Path,
    expected_source: &[u8],
    expected: &SemanticArtifacts,
) -> Result<SemanticImportReceipt, LegacyError> {
    let expected_files: &[(&str, &[u8])] = &[
        (SOURCE_FILE, expected_source),
        (DATASET_FILE, &expected.dataset_json),
        (PAYLOAD_FILE, &expected.payload),
        (MAPPING_REPORT_FILE, &expected.mapping_report),
        (FIDELITY_REPORT_FILE, &expected.fidelity_report),
        (SEMANTIC_RECEIPT_FILE, &expected.semantic_receipt),
    ];
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
