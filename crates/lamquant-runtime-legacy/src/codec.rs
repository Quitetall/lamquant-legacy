//! Codec-backed source + sink (feature `codec`) — the first *real* biosignal
//! path through the runtime, no system library required.
//!
//! - [`LmlReplaySource`] decodes an existing `.lml` and streams its windows, a
//!   stand-in for a live source that exercises the exact Engine path with real
//!   codec data (the live LSL source, feature `lsl`, needs the `liblsl` system
//!   lib and lands later).
//! - [`LmlFileSink`] accumulates windows and compresses them to a (rotating)
//!   `.lml` container via the LML codec — the compress-to-disk output path.
//!
//! Both go through `lamquant_core::container::{read_file,write_file}` (the LML
//! v1 container, byte-faithful lossless), so a window that enters `LmlFileSink`
//! and is read back is sample-for-sample identical to the source.

use std::path::{Path, PathBuf};

use async_trait::async_trait;

use crate::error::{Result, RuntimeError};
use crate::sink::{Sink, SinkInfo};
use crate::source::{Source, SourceInfo};
use crate::window::WindowBatch;

/// Streams the windows of an already-encoded `.lml` file. Real biosignal data,
/// no system library — proves the Source→Engine→Sink path end-to-end here.
pub struct LmlReplaySource {
    label: String,
    signal: Vec<Vec<i64>>,
    sample_rate: f64,
    window: usize,
    pos: usize,
    seq: u64,
}

impl LmlReplaySource {
    /// Decode `path` and prepare to stream it in `window`-sample chunks.
    pub fn open(path: impl AsRef<Path>, window: usize) -> Result<Self> {
        let path = path.as_ref();
        let (signal, _meta) =
            lamquant_core::container::read_file(path).map_err(|e| RuntimeError::Source {
                name: path.display().to_string(),
                msg: e.to_string(),
            })?;
        // Sample rate is not carried on the plain read; the manifest declares the
        // window and the sink re-stamps rate. Default to the codec's canonical
        // 250 Hz for the replay's own descriptor.
        Ok(Self {
            label: path.display().to_string(),
            signal,
            sample_rate: 250.0,
            window: window.max(1),
            pos: 0,
            seq: 0,
        })
    }

    fn n_samples(&self) -> usize {
        self.signal.first().map_or(0, Vec::len)
    }
}

#[async_trait]
impl Source for LmlReplaySource {
    async fn next_window(&mut self) -> Result<Option<WindowBatch>> {
        let total = self.n_samples();
        if self.pos >= total {
            return Ok(None);
        }
        let end = (self.pos + self.window).min(total);
        let channels: Vec<Vec<i64>> = self
            .signal
            .iter()
            .map(|c| c[self.pos..end].to_vec())
            .collect();
        self.pos = end;
        let seq = self.seq;
        self.seq += 1;
        Ok(Some(WindowBatch::new(seq, channels, self.sample_rate)))
    }

    fn info(&self) -> SourceInfo {
        SourceInfo {
            kind: "lml-replay".into(),
            label: self.label.clone(),
            n_channels: self.signal.len(),
            sample_rate_hz: self.sample_rate,
            modality: None,
        }
    }
}

/// Compresses incoming windows to a `.lml` container. Accumulates the
/// (channel-major) signal and writes a complete container on rotation and on
/// close, so each output file is a self-contained, byte-faithful `.lml`.
#[derive(Clone, Copy, PartialEq, Eq)]
struct StreamLayout {
    channels: usize,
    samples: usize,
    sample_rate_millihz: u64,
}

pub struct LmlFileSink {
    base_path: PathBuf,
    rotate_windows: u64,
    noise_bits: u8,
    /// Per-channel accumulation for the current file.
    buffer: Vec<Vec<i64>>,
    windows_buffered: u64,
    layout: Option<StreamLayout>,
    file_idx: u64,
}

impl LmlFileSink {
    /// `rotate_windows == 0` writes a single file (`base_path`) on close;
    /// otherwise a new file is written every `rotate_windows` windows.
    pub fn new(base_path: impl Into<PathBuf>, rotate_windows: u64, noise_bits: u8) -> Self {
        Self {
            base_path: base_path.into(),
            rotate_windows,
            noise_bits,
            buffer: Vec::new(),
            windows_buffered: 0,
            layout: None,
            file_idx: 0,
        }
    }

    /// The path for the current file (unrotated writes use `base_path` verbatim).
    fn current_path(&self) -> PathBuf {
        if self.rotate_windows == 0 && self.file_idx == 0 {
            return self.base_path.clone();
        }
        let stem = self
            .base_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("out");
        let ext = self
            .base_path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("lml");
        let name = format!("{stem}.{:04}.{ext}", self.file_idx);
        self.base_path.with_file_name(name)
    }

    /// Encode + write the buffered signal, then clear for the next file. No-op
    /// when the buffer is empty.
    async fn flush_file(&mut self) -> Result<()> {
        if self.buffer.is_empty() || self.windows_buffered == 0 {
            return Ok(());
        }
        let path = self.current_path();
        if let Some(dir) = path.parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir)?;
            }
        }
        let signal = std::mem::take(&mut self.buffer);
        let layout = self.layout.ok_or_else(|| RuntimeError::Sink {
            name: self.base_path.display().to_string(),
            msg: "buffered signal has no established stream layout".into(),
        })?;
        let sr = layout.sample_rate_millihz as f64 / 1000.0;
        let (window, nb) = (layout.samples, self.noise_bits);
        let path_for_task = path.clone();
        // Encode is CPU-heavy + sync → keep it off the async reactor.
        let stats = tokio::task::spawn_blocking(move || {
            lamquant_core::container::write_file(&path_for_task, &signal, sr, window, nb, "{}")
        })
        .await
        .map_err(|e| RuntimeError::Sink {
            name: path.display().to_string(),
            msg: format!("encode task join failed: {e}"),
        })?
        .map_err(|e| RuntimeError::Sink {
            name: path.display().to_string(),
            msg: e.to_string(),
        })?;
        tracing::info!(path = %path.display(), windows = self.windows_buffered,
            bytes = stats.compressed_size, "lml-file: wrote container");
        self.windows_buffered = 0;
        self.file_idx += 1;
        Ok(())
    }
}

#[async_trait]
impl Sink for LmlFileSink {
    async fn consume(&mut self, batch: &WindowBatch) -> Result<()> {
        if !batch.is_rectangular() {
            return Err(RuntimeError::Sink {
                name: self.base_path.display().to_string(),
                msg: format!("non-rectangular window seq={}", batch.seq),
            });
        }
        let incoming_layout = StreamLayout {
            channels: batch.n_channels(),
            samples: batch.n_samples(),
            sample_rate_millihz: batch.sample_rate_millihz,
        };
        if let Some(expected) = self.layout {
            if incoming_layout != expected {
                return Err(RuntimeError::Sink {
                    name: self.base_path.display().to_string(),
                    msg: format!(
                        "stream layout changed at seq={}: expected {} channels x {} samples at {} mHz, got {} channels x {} samples at {} mHz",
                        batch.seq,
                        expected.channels,
                        expected.samples,
                        expected.sample_rate_millihz,
                        incoming_layout.channels,
                        incoming_layout.samples,
                        incoming_layout.sample_rate_millihz,
                    ),
                });
            }
        } else {
            self.layout = Some(incoming_layout);
        }
        if self.buffer.is_empty() {
            self.buffer = vec![Vec::new(); batch.n_channels()];
        }
        for (buffer, channel) in self.buffer.iter_mut().zip(&batch.channels) {
            buffer.extend_from_slice(channel);
        }
        self.windows_buffered += 1;
        if self.rotate_windows > 0 && self.windows_buffered >= self.rotate_windows {
            self.flush_file().await?;
        }
        Ok(())
    }

    async fn flush(&mut self) -> Result<()> {
        self.flush_file().await
    }

    async fn close(&mut self) -> Result<()> {
        self.flush_file().await
    }

    fn info(&self) -> SinkInfo {
        SinkInfo {
            kind: "lml-file".into(),
            label: self.base_path.display().to_string(),
        }
    }
}
