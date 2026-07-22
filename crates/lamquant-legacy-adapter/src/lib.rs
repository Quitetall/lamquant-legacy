#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

const RECEIPT_FILE: &str = "receipt.json";
const SOURCE_FILE: &str = "source.bin";

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
pub struct Inspection {
    pub profile: String,
    pub source_bytes: u64,
    pub source_blake3: String,
    pub semantic_conversion: bool,
    pub forensic_conversion: bool,
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
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "status", content = "value", rename_all = "kebab-case")]
pub enum ProcessResponse {
    OkManifest(CapabilityManifest),
    OkInspection(Inspection),
    OkConversion(ConvertReceipt),
    Error { code: String, message: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LegacyError {
    UnknownMagic,
    SourceTooLarge,
    AcceptanceRequired,
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
                formatter.write_str("forensic fidelity receipt must be accepted before writing")
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
                semantic_import: false,
                reverse_export: false,
            })
            .collect(),
    }
}

pub fn inspect(source: &Path, max_source_bytes: u64) -> Result<Inspection, LegacyError> {
    let bytes = read_bounded(source, max_source_bytes)?;
    let format = detect_format(&bytes)?;
    Ok(Inspection {
        profile: format.profile().to_owned(),
        source_bytes: bytes.len() as u64,
        source_blake3: blake3::hash(&bytes).to_hex().to_string(),
        semantic_conversion: false,
        forensic_conversion: true,
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
    let temporary = parent.join(format!(
        ".lamquant-legacy-{}-{}",
        std::process::id(),
        &receipt.source_blake3[..16]
    ));
    if temporary.exists() {
        fs::remove_dir_all(&temporary).map_err(io_error)?;
    }
    fs::create_dir(&temporary).map_err(io_error)?;
    let result = (|| {
        write_new(&temporary.join(SOURCE_FILE), &bytes)?;
        let receipt_bytes = serde_json::to_vec_pretty(&receipt)
            .map_err(|error| LegacyError::InvalidProtocol(error.to_string()))?;
        write_new(&temporary.join(RECEIPT_FILE), &receipt_bytes)?;
        fs::rename(&temporary, &request.destination).map_err(io_error)?;
        Ok(receipt.clone())
    })();
    if result.is_err() && temporary.exists() {
        let _ = fs::remove_dir_all(&temporary);
    }
    result
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
    };
    result.unwrap_or_else(|error| ProcessResponse::Error {
        code: error.code().to_owned(),
        message: error.to_string(),
    })
}

fn read_bounded(path: &Path, max_source_bytes: u64) -> Result<Vec<u8>, LegacyError> {
    let metadata = fs::symlink_metadata(path).map_err(io_error)?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(LegacyError::UnsafeSource);
    }
    if metadata.len() > max_source_bytes {
        return Err(LegacyError::SourceTooLarge);
    }
    let bytes = fs::read(path).map_err(io_error)?;
    if bytes.len() as u64 > max_source_bytes {
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
