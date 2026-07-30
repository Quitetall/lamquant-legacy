//! The [`Sink`] trait — the output half of the runtime. One impl per output
//! path (compress-to-`.lml`, re-publish to LSL, neural LMQ, provenance, forward).
//! The common `consume(&WindowBatch)` signature is exactly the fan-out
//! abstraction the codebase lacks today (every sink has a bespoke signature).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::window::WindowBatch;

/// Static description of a sink, surfaced to the status/console layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SinkInfo {
    /// Stable kind tag (e.g. `"lml-file"`, `"lsl-outlet"`, `"mem"`).
    pub kind: String,
    /// Human label (e.g. the output path or outlet name).
    pub label: String,
}

/// A biosignal output path. `consume` is called once per window in arrival
/// order; `flush` is a checkpoint (durability); `close` finalizes at end of
/// stream. A sink error is reported by the engine but does NOT tear down the
/// other sinks on the same source (fan-out is independent — one dead output
/// must not kill the recording). Implementations are `Send`.
#[async_trait]
pub trait Sink: Send {
    /// Consume one window. Heavy CPU work (encode) belongs behind
    /// `tokio::task::spawn_blocking` inside the impl.
    async fn consume(&mut self, batch: &WindowBatch) -> Result<()>;

    /// Durability checkpoint (e.g. flush a container's buffered windows). The
    /// default is a no-op for stateless sinks.
    async fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    /// Finalize at end of stream (e.g. write a container footer, drop an outlet).
    /// The default is a no-op.
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }

    /// Static descriptor for status/console.
    fn info(&self) -> SinkInfo;
}
