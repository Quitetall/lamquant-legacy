//! Folder-watch source (feature `watch`) — a [`Source`] that watches a directory
//! and streams the windows of every `.lml`/`.edf`/`.bdf` file dropped into it
//! (ADR 0136). The *different-shaped* source: unlike replay (one file) or LSL
//! (one live stream), this is event-driven and multi-file, and subsumes the
//! `lml watch` folder-daemon by making it one `Source` impl under the Engine.
//!
//! It mirrors the proven `watch_dir` backpressure discipline
//! (`codec-lossless/.../async_io.rs`): a `notify` OS-callback that **never
//! `.await`s** and drops-oldest-with-`warn!` on a full bounded queue; each new
//! path is decoded on `spawn_blocking` (EDF read / LML container read are sync +
//! CPU-heavy) and streamed a window at a time.
//!
//! **Sample rate.** EDF/BDF self-report their rate (`EdfFile::sample_rate`); a
//! `.lml` container does NOT carry the rate through `container::read_from` (the
//! header returned drops it), so the manifest declares an authoritative
//! `sample_rate_hz` for `.lml` files — exactly as `watch_dir` takes `sample_rate`
//! as a parameter. No magic constant.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use notify::{RecursiveMode, Watcher};

use crate::error::{Result, RuntimeError};
use crate::source::{Source, SourceInfo};
use crate::window::WindowBatch;

/// The extensions a dir-watch source ingests when the manifest names none.
pub const DEFAULT_EXTENSIONS: &[&str] = &["lml", "edf", "bdf"];

/// Watches `dir` and streams the windows of each newly-created/modified file
/// whose extension is in `extensions`.
pub struct DirWatchSource {
    label: String,
    /// Held to keep the OS watcher alive for the source's lifetime.
    _watcher: notify::RecommendedWatcher,
    rx: tokio::sync::mpsc::Receiver<PathBuf>,
    window: usize,
    /// Declared rate for `.lml` files (EDF/BDF override with their own).
    declared_rate: f64,
    /// Channel count of the most recent file (for `info`); 0 until first file.
    last_channels: usize,
    /// The file currently being streamed: (signal, sample_rate, cursor).
    cur: Option<(Vec<Vec<i64>>, f64, usize)>,
    seq: u64,
}

impl DirWatchSource {
    /// Watch `dir`, emitting `window`-sample windows. `sample_rate_hz` is the
    /// declared rate for `.lml` files; `extensions` (lowercased, no dot) filter
    /// which files are ingested (defaults to [`DEFAULT_EXTENSIONS`]).
    pub fn open(
        dir: impl Into<PathBuf>,
        window: usize,
        sample_rate_hz: f64,
        extensions: Vec<String>,
    ) -> Result<Self> {
        let dir = dir.into();
        if !dir.is_dir() {
            return Err(RuntimeError::Source {
                name: dir.display().to_string(),
                msg: "watch dir does not exist or is not a directory".into(),
            });
        }
        let exts: Vec<String> = if extensions.is_empty() {
            DEFAULT_EXTENSIONS.iter().map(|s| s.to_string()).collect()
        } else {
            extensions
                .iter()
                .map(|e| e.trim_start_matches('.').to_ascii_lowercase())
                .collect()
        };
        // Bounded queue + drop-oldest-with-WARN: the OS callback must not block.
        let (tx, rx) = tokio::sync::mpsc::channel::<PathBuf>(64);
        let send_tx = tx.clone();
        let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
            if let Ok(ev) = res {
                if matches!(
                    ev.kind,
                    notify::EventKind::Create(_) | notify::EventKind::Modify(_)
                ) {
                    for path in ev.paths {
                        let matches_ext = path
                            .extension()
                            .map(|e| e.to_string_lossy().to_ascii_lowercase())
                            .map(|lower| exts.iter().any(|e| e == &lower))
                            .unwrap_or(false);
                        if matches_ext {
                            // try_send: never .await in the notify callback. Full → drop oldest.
                            if let Err(e) = send_tx.try_send(path.clone()) {
                                tracing::warn!(
                                    "dir-watch: queue full, dropping {} ({e})",
                                    path.display()
                                );
                            }
                        }
                    }
                }
            }
        })
        .map_err(|e| RuntimeError::Source {
            name: dir.display().to_string(),
            msg: format!("notify init: {e}"),
        })?;
        watcher
            .watch(&dir, RecursiveMode::Recursive)
            .map_err(|e| RuntimeError::Source {
                name: dir.display().to_string(),
                msg: format!("watch: {e}"),
            })?;
        drop(tx); // our extra sender: rx closes when the watcher's clone drops.
        Ok(Self {
            label: dir.display().to_string(),
            _watcher: watcher,
            rx,
            window: window.max(1),
            declared_rate: sample_rate_hz,
            last_channels: 0,
            cur: None,
            seq: 0,
        })
    }

    /// Take the next window from the file currently being streamed, if any.
    fn next_from_current(&mut self) -> Option<WindowBatch> {
        let (signal, rate, pos) = self.cur.as_mut()?;
        let total = signal.first().map_or(0, Vec::len);
        if *pos >= total {
            self.cur = None;
            return None;
        }
        let end = (*pos + self.window).min(total);
        let channels: Vec<Vec<i64>> = signal.iter().map(|c| c[*pos..end].to_vec()).collect();
        *pos = end;
        let rate = *rate;
        let seq = self.seq;
        self.seq += 1;
        Some(WindowBatch::new(seq, channels, rate))
    }
}

/// Decode one dropped file into (channel-major signal, sample_rate). EDF/BDF
/// self-report the rate; `.lml` (and anything else) use `declared_rate`.
fn decode_file(path: &Path, declared_rate: f64) -> Result<(Vec<Vec<i64>>, f64)> {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    let err = |msg: String| RuntimeError::Source {
        name: path.display().to_string(),
        msg,
    };
    match ext.as_str() {
        "edf" | "bdf" => {
            let edf = lamquant_core::edf::read_edf(path).map_err(|e| err(e.to_string()))?;
            Ok((edf.signal, edf.sample_rate))
        }
        _ => {
            let (signal, _meta) = crate::codec::read_lml_path(path)?;
            Ok((signal, declared_rate))
        }
    }
}

#[async_trait]
impl Source for DirWatchSource {
    async fn next_window(&mut self) -> Result<Option<WindowBatch>> {
        loop {
            if let Some(batch) = self.next_from_current() {
                return Ok(Some(batch));
            }
            // Current file exhausted (or none yet): wait for the next dropped file.
            let path = match self.rx.recv().await {
                Some(p) => p,
                None => return Ok(None), // watcher gone → end of stream.
            };
            let declared = self.declared_rate;
            let path_for_task = path.clone();
            let decoded =
                tokio::task::spawn_blocking(move || decode_file(&path_for_task, declared))
                    .await
                    .map_err(|e| RuntimeError::Source {
                        name: path.display().to_string(),
                        msg: format!("decode task join failed: {e}"),
                    })?;
            match decoded {
                Ok((signal, rate)) if !signal.is_empty() && !signal[0].is_empty() => {
                    self.last_channels = signal.len();
                    self.cur = Some((signal, rate, 0));
                    tracing::info!(path = %path.display(), channels = self.last_channels, "dir-watch: streaming file");
                    // Loop back: emit its first window.
                }
                Ok(_) => tracing::warn!("dir-watch: {} decoded empty, skipping", path.display()),
                Err(e) => tracing::warn!("dir-watch: decode {} failed: {e}", path.display()),
            }
        }
    }

    fn info(&self) -> SourceInfo {
        SourceInfo {
            kind: "dir-watch".into(),
            label: self.label.clone(),
            n_channels: self.last_channels,
            sample_rate_hz: self.declared_rate,
            modality: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_rejects_missing_dir() {
        // matches! (not unwrap_err) so the source need not be Debug.
        let r = DirWatchSource::open("/no/such/dir/xyz", 250, 250.0, vec![]);
        assert!(matches!(r, Err(RuntimeError::Source { .. })));
    }

    #[tokio::test]
    async fn streams_windows_of_a_dropped_lml() {
        // Write a real `.lml` into a fresh watch dir, then assert the source
        // streams its windows byte-faithfully once the file appears.
        let dir = std::env::temp_dir().join(format!("dirwatch_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mut src = DirWatchSource::open(&dir, 500, 250.0, vec!["lml".into()]).unwrap();

        // 3 channels × 1000 samples of ramp data → a 2-window file at window=500.
        let signal: Vec<Vec<i64>> = (0..3)
            .map(|c| (0..1000).map(|t| (c * 1000 + t) as i64).collect())
            .collect();
        let lml = dir.join("drop.lml");
        lamquant_core::container::write_file(&lml, &signal, 250.0, 500, 0, "{}").unwrap();

        // First two windows come from the dropped file.
        let w0 = tokio::time::timeout(std::time::Duration::from_secs(5), src.next_window())
            .await
            .expect("timed out waiting for window")
            .unwrap()
            .expect("expected a window");
        assert_eq!(w0.seq, 0);
        assert_eq!(w0.n_channels(), 3);
        assert_eq!(w0.n_samples(), 500);
        assert_eq!(w0.sample_rate_hz(), 250.0);
        assert_eq!(w0.channels[0][0], 0); // first ramp sample, byte-faithful.

        let w1 = tokio::time::timeout(std::time::Duration::from_secs(5), src.next_window())
            .await
            .expect("timed out")
            .unwrap()
            .unwrap();
        assert_eq!(w1.seq, 1);
        assert_eq!(w1.channels[0][0], 500); // second window starts at sample 500.

        let _ = std::fs::remove_dir_all(&dir);
    }
}
