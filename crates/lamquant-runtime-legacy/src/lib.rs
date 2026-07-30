//! `lamquant-runtime` — the long-lived biosignal ingest→sink daemon (ADR 0135).
//!
//! The product is two halves: **ingest → ABIR**, then **ABIR → output paths**.
//! This crate makes each half a single trait so adding a new biosignal or output
//! is one impl, not a rewrite:
//!
//! - [`Source`] — async, yields channel-major [`WindowBatch`]es (LSL inlet,
//!   folder-watch, firmware board, file reader, ...).
//! - [`Sink`] — async, `consume(&WindowBatch)` (compress-to-`.lml`, LSL relay,
//!   provenance, forward, ...).
//!
//! The [`Engine`] hosts **N sources → per-source bounded channel → M sinks** on
//! one long-lived tokio runtime: bounded-channel backpressure, `spawn_blocking`
//! for CPU-heavy encode (added by concrete sinks), and graceful drain on shutdown.
//! Routing is config-as-data (`RuntimeManifest`); live state is persisted
//! append-only (`status`).

pub const SOURCE_REVISION: &str = "93119e4e25402b2c27a15518f1d2399a98990257";

#[cfg(feature = "auth")]
pub mod auth;
#[cfg(feature = "codec")]
pub mod codec;
#[cfg(feature = "daemon")]
pub mod control;
#[cfg(feature = "daemon")]
pub mod daemon;
pub mod engine;
pub mod error;
#[cfg(feature = "lsl")]
pub mod lsl;
pub mod manifest;
pub mod mem;
#[cfg(feature = "metrics")]
pub mod metrics_sink;
#[cfg(feature = "neg")]
pub mod neg_sink;
pub mod sink;
pub mod source;
pub mod status;
#[cfg(feature = "watch")]
pub mod watch;
pub mod window;

pub use engine::{Engine, EngineConfig, LiveStats, PipelineReport, SinkLive};
pub use error::{Result, RuntimeError};
pub use manifest::{PipelineSpec, RuntimeManifest, SinkSpec, SourceSpec, MANIFEST_VERSION};
pub use sink::{Sink, SinkInfo};
pub use source::{Source, SourceInfo};
pub use status::{RuntimeStatus, SinkStat, SourceStat, StatusWriter};
pub use window::WindowBatch;
