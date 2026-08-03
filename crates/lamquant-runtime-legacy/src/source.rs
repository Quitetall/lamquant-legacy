//! The [`Source`] trait — the ingest half of the runtime. One impl per way a
//! biosignal can arrive (LSL inlet, folder-watch, firmware board, file reader).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::window::WindowBatch;

/// Static description of a source, surfaced to the status/console layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceInfo {
    /// Stable kind tag (e.g. `"lsl"`, `"mem"`, `"edf-watch"`).
    pub kind: String,
    /// Human label (e.g. the LSL stream name or watched dir).
    pub label: String,
    pub n_channels: usize,
    pub sample_rate_hz: f64,
    /// Declared modality tag, when the source is born-typed.
    pub modality: Option<String>,
}

/// A streaming biosignal source. `next_window` yields batches until the stream
/// ends (`Ok(None)`); an error terminates the pipeline for this source (the
/// engine records it and drains the sinks). Implementations are `Send` so the
/// engine can own them in a task; they need not be `Sync`.
#[async_trait]
pub trait Source: Send {
    /// Pull the next window, or `Ok(None)` at end-of-stream. Should be
    /// cancellation-friendly (await points let the engine stop it promptly).
    async fn next_window(&mut self) -> Result<Option<WindowBatch>>;

    /// Static descriptor for status/console.
    fn info(&self) -> SourceInfo;
}
