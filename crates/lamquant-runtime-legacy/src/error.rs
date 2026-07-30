//! Typed runtime errors. Fail-closed: a bad manifest or an unbuildable
//! source/sink is an error, never a silent default.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, RuntimeError>;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("manifest version {found} unsupported (this build accepts {expected})")]
    ManifestVersion { found: u32, expected: u32 },

    #[error("manifest parse error: {0}")]
    ManifestParse(String),

    #[error("manifest is empty: at least one pipeline is required")]
    EmptyManifest,

    #[error("pipeline {pipeline:?} names no sinks")]
    NoSinks { pipeline: String },

    #[error("source kind {kind:?} is not built into this runtime yet")]
    UnknownSource { kind: String },

    #[error("sink kind {kind:?} is not built into this runtime yet")]
    UnknownSink { kind: String },

    #[error("source {name:?} failed: {msg}")]
    Source { name: String, msg: String },

    #[error("sink {name:?} failed: {msg}")]
    Sink { name: String, msg: String },

    #[error("i/o: {0}")]
    Io(#[from] std::io::Error),

    #[error("status serialize: {0}")]
    Status(#[from] serde_json::Error),
}
