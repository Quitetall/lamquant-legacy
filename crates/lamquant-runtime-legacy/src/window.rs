//! [`WindowBatch`] — the unit of flow through the runtime: one channel-major
//! window of integer samples plus the minimal provenance a sink needs.
//!
//! Phase 0 keeps this self-contained (no ABIR/codec dependency) so the crate
//! builds dependency-light. Concrete source/sink adapters lower each batch into
//! validated ABIR views only at explicit compatibility boundaries.

use serde::{Deserialize, Serialize};

/// One window of biosignal samples, channel-major (`[n_channels][n_samples]`).
///
/// Integer samples are the codec's native currency (ADC counts / `i64`); a live
/// float source quantizes to its declared physical resolution before building a
/// batch. Cheap to fan out — the engine wraps it in an `Arc` so N sinks share one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WindowBatch {
    /// Monotonic per-pipeline window index (0-based). Gap detection for sinks.
    pub seq: u64,
    /// Channel-major samples: `channels[c][t]`.
    pub channels: Vec<Vec<i64>>,
    /// Nominal sample rate (Hz) declared by the source.
    pub sample_rate_millihz: u64,
    /// Source-provided timestamp of the FIRST sample (LSL clock seconds ×1e6),
    /// when the source has one. `None` for clockless sources.
    pub first_ts_micros: Option<i64>,
}

impl WindowBatch {
    pub fn new(seq: u64, channels: Vec<Vec<i64>>, sample_rate_hz: f64) -> Self {
        Self {
            seq,
            channels,
            sample_rate_millihz: (sample_rate_hz * 1000.0).round() as u64,
            first_ts_micros: None,
        }
    }

    pub fn with_timestamp(mut self, first_ts_seconds: f64) -> Self {
        self.first_ts_micros = Some((first_ts_seconds * 1_000_000.0).round() as i64);
        self
    }

    pub fn sample_rate_hz(&self) -> f64 {
        self.sample_rate_millihz as f64 / 1000.0
    }

    pub fn n_channels(&self) -> usize {
        self.channels.len()
    }

    /// Samples in the first channel (windows are rectangular by construction).
    pub fn n_samples(&self) -> usize {
        self.channels.first().map_or(0, Vec::len)
    }

    /// True when every channel has the same, non-zero length — the invariant a
    /// codec sink requires. Sinks should reject a batch that fails this.
    pub fn is_rectangular(&self) -> bool {
        let t = self.n_samples();
        t > 0 && self.channels.iter().all(|c| c.len() == t)
    }
}
