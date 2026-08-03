//! In-memory reference [`Source`]/[`Sink`] impls — the smallest possible proof
//! of the trait surface, used by the crate's tests and as the template a real
//! impl follows. No I/O, no codec.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Result;
use crate::sink::{Sink, SinkInfo};
use crate::source::{Source, SourceInfo};
use crate::window::WindowBatch;

/// Emits `n_windows` rectangular ramp windows, then ends. An optional per-window
/// pace makes it a long-running synthetic source (for demos + the daemon tests).
pub struct MemSource {
    label: String,
    n_channels: usize,
    n_samples: usize,
    remaining: u64,
    seq: u64,
    sample_rate_hz: f64,
    pace_ms: u64,
}

impl MemSource {
    pub fn ramp(
        label: String,
        n_channels: usize,
        n_samples: usize,
        n_windows: u64,
        sample_rate_hz: f64,
    ) -> Self {
        Self {
            label,
            n_channels,
            n_samples,
            remaining: n_windows,
            seq: 0,
            sample_rate_hz,
            pace_ms: 0,
        }
    }

    /// Wait `ms` before each window (0 = as fast as possible).
    pub fn with_pace(mut self, ms: u64) -> Self {
        self.pace_ms = ms;
        self
    }
}

#[async_trait]
impl Source for MemSource {
    async fn next_window(&mut self) -> Result<Option<WindowBatch>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        if self.pace_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(self.pace_ms)).await;
        }
        self.remaining -= 1;
        let seq = self.seq;
        self.seq += 1;
        // A deterministic ramp so a sink can verify content if it wants.
        let channels: Vec<Vec<i64>> = (0..self.n_channels)
            .map(|c| {
                (0..self.n_samples)
                    .map(|t| (seq as i64) * 1000 + (c as i64) * 100 + t as i64)
                    .collect()
            })
            .collect();
        Ok(Some(WindowBatch::new(seq, channels, self.sample_rate_hz)))
    }

    fn info(&self) -> SourceInfo {
        SourceInfo {
            kind: "mem".into(),
            label: self.label.clone(),
            n_channels: self.n_channels,
            sample_rate_hz: self.sample_rate_hz,
            modality: None,
        }
    }
}

/// Counts consumed windows into a shared atomic the test/console can read.
pub struct MemSink {
    label: String,
    consumed: Arc<AtomicU64>,
    flushed: Arc<AtomicU64>,
    closed: Arc<AtomicU64>,
}

impl MemSink {
    pub fn new(label: String) -> Self {
        Self {
            label,
            consumed: Arc::new(AtomicU64::new(0)),
            flushed: Arc::new(AtomicU64::new(0)),
            closed: Arc::new(AtomicU64::new(0)),
        }
    }

    /// A handle to this sink's consumed-window counter (clone before boxing).
    pub fn counter(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.consumed)
    }

    /// A handle to the flush-count (1 after a clean close).
    pub fn flush_counter(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.flushed)
    }

    /// A handle to the close-count (1 after a clean close).
    pub fn close_counter(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.closed)
    }
}

#[async_trait]
impl Sink for MemSink {
    async fn consume(&mut self, _batch: &WindowBatch) -> Result<()> {
        self.consumed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn flush(&mut self) -> Result<()> {
        self.flushed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        self.closed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn info(&self) -> SinkInfo {
        SinkInfo {
            kind: "mem".into(),
            label: self.label.clone(),
        }
    }
}
