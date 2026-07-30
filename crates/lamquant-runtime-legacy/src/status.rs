//! Append-only status persistence (the BLUT-console precedent). The daemon
//! writes one [`RuntimeStatus`] line per update to `status.jsonl`; a detached
//! console (or a crashed one restarting) tails it. Each record stamps
//! `updated_ms` so a reader can detect a hung daemon by staleness.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::sink::SinkInfo;
use crate::source::SourceInfo;

/// Wall-clock milliseconds since the Unix epoch (monotonic-enough for staleness).
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceStat {
    pub info: SourceInfo,
    pub windows_in: u64,
    pub errored: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SinkStat {
    pub info: SinkInfo,
    pub windows_consumed: u64,
    pub windows_dropped: u64,
    pub errored: bool,
}

/// A snapshot of one pipeline's live state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeStatus {
    pub updated_ms: u64,
    pub pipeline: String,
    pub source: SourceStat,
    pub sinks: Vec<SinkStat>,
}

impl RuntimeStatus {
    /// True when the last update is older than `max_age_ms` — the daemon looks
    /// hung to a reader. (The console uses this to grey out a stale panel.)
    pub fn is_stale(&self, now: u64, max_age_ms: u64) -> bool {
        now.saturating_sub(self.updated_ms) > max_age_ms
    }
}

/// Append-only writer over `status.jsonl`. Line-buffered + flushed per record so
/// a tailing reader sees each update immediately and a crash loses at most the
/// in-flight line.
pub struct StatusWriter {
    file: File,
}

impl StatusWriter {
    /// Open (creating, appending) the status file at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        if let Some(dir) = path.as_ref().parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir)?;
            }
        }
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self { file })
    }

    /// Append one status record as a JSON line and flush.
    pub fn record(&mut self, status: &RuntimeStatus) -> Result<()> {
        let mut line = serde_json::to_vec(status)?;
        line.push(b'\n');
        self.file.write_all(&line)?;
        self.file.flush()?;
        Ok(())
    }
}
