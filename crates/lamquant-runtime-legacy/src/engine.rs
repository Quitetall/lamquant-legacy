//! The [`Engine`] — drives sources and fans each window out to its sinks.
//!
//! One **bounded** channel per sink (house rule R33: never block a producer,
//! never silently lose — a full channel drops the window with a WARN, counted in
//! the report). Each sink runs in its own task so one slow/failing output cannot
//! stall the others or the source. On end-of-stream **or** shutdown the source
//! stops, the senders drop, and every sink drains its queue then `flush`+`close`
//! — a graceful drain, not a hard stop.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::warn;

use crate::error::Result;
use crate::manifest::{build_sink, build_source, RuntimeManifest};
use crate::sink::{Sink, SinkInfo};
use crate::source::{Source, SourceInfo};
use crate::window::WindowBatch;

pub(crate) struct PreparedPipeline {
    pub(crate) name: String,
    pub(crate) source: Box<dyn Source>,
    pub(crate) sinks: Vec<Box<dyn Sink>>,
}

pub(crate) fn prepare_manifest(manifest: &RuntimeManifest) -> Result<Vec<PreparedPipeline>> {
    manifest
        .pipelines
        .iter()
        .map(|pipeline| {
            let source = build_source(&pipeline.source)?;
            let sinks = pipeline
                .sinks
                .iter()
                .map(build_sink)
                .collect::<Result<Vec<_>>>()?;
            Ok(PreparedPipeline {
                name: pipeline.name.clone(),
                source,
                sinks,
            })
        })
        .collect()
}

/// Engine tuning. `channel_cap` bounds each per-sink queue.
#[derive(Debug, Clone, Copy)]
pub struct EngineConfig {
    pub channel_cap: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self { channel_cap: 256 }
    }
}

/// What one sink did over a pipeline's lifetime.
#[derive(Debug, Clone, PartialEq)]
pub struct SinkOutcome {
    pub info: SinkInfo,
    pub consumed: u64,
    pub dropped: u64,
    pub errored: bool,
}

/// The tally for one pipeline run.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineReport {
    pub name: String,
    pub source: SourceInfo,
    pub windows_in: u64,
    /// `Some` if the source ended with an error (vs a clean end-of-stream).
    pub source_error: Option<String>,
    pub sinks: Vec<SinkOutcome>,
}

/// Live, atomic per-sink counters a daemon can poll while a pipeline runs.
#[derive(Debug, Default)]
pub struct SinkLive {
    pub consumed: AtomicU64,
    pub dropped: AtomicU64,
}

/// Live snapshot of one running pipeline (shared with the daemon's status loop).
#[derive(Debug)]
pub struct LiveStats {
    pub windows_in: AtomicU64,
    pub source_error: AtomicBool,
    pub ended: AtomicBool,
    pub sinks: Vec<Arc<SinkLive>>,
}

impl LiveStats {
    pub fn new(n_sinks: usize) -> Arc<Self> {
        Arc::new(Self {
            windows_in: AtomicU64::new(0),
            source_error: AtomicBool::new(false),
            ended: AtomicBool::new(false),
            sinks: (0..n_sinks)
                .map(|_| Arc::new(SinkLive::default()))
                .collect(),
        })
    }
}

pub struct Engine {
    cfg: EngineConfig,
}

impl Engine {
    pub fn new(cfg: EngineConfig) -> Self {
        Self { cfg }
    }

    /// Run one pipeline to completion: drive `source`, fan every window out to
    /// each sink over a bounded channel, and drain gracefully when the source
    /// ends or `shutdown` fires. Never returns an error — failures are recorded
    /// in the [`PipelineReport`] (a recording daemon must not unwind on one bad
    /// window or one dead sink).
    pub async fn run_pipeline(
        &self,
        name: impl Into<String>,
        source: Box<dyn Source>,
        sinks: Vec<Box<dyn Sink>>,
        shutdown: CancellationToken,
    ) -> PipelineReport {
        let stats = LiveStats::new(sinks.len());
        self.run_pipeline_tracked(name, source, sinks, shutdown, stats)
            .await
    }

    /// Like [`Engine::run_pipeline`], but shares a [`LiveStats`] the caller can
    /// poll while the pipeline runs — the daemon's live status source.
    pub async fn run_pipeline_tracked(
        &self,
        name: impl Into<String>,
        mut source: Box<dyn Source>,
        sinks: Vec<Box<dyn Sink>>,
        shutdown: CancellationToken,
        stats: Arc<LiveStats>,
    ) -> PipelineReport {
        let name = name.into();
        let source_info = source.info();

        // Spin one task per sink, each owning its receiver.
        let mut senders: Vec<mpsc::Sender<Arc<WindowBatch>>> = Vec::with_capacity(sinks.len());
        let mut infos: Vec<SinkInfo> = Vec::with_capacity(sinks.len());
        let mut dropped: Vec<u64> = vec![0; sinks.len()];
        let mut handles = Vec::with_capacity(sinks.len());
        for (i, mut sink) in sinks.into_iter().enumerate() {
            let info = sink.info();
            infos.push(info.clone());
            let live = Arc::clone(&stats.sinks[i]);
            let (tx, mut rx) = mpsc::channel::<Arc<WindowBatch>>(self.cfg.channel_cap.max(1));
            senders.push(tx);
            handles.push(tokio::spawn(async move {
                let mut consumed = 0u64;
                let mut errored = false;
                while let Some(batch) = rx.recv().await {
                    match sink.consume(&batch).await {
                        Ok(()) => {
                            consumed += 1;
                            live.consumed.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(e) => {
                            errored = true;
                            warn!(sink = %info.label, error = %e, "sink consume failed; continuing");
                        }
                    }
                }
                // Channel closed (source ended / shutdown) → finalize best-effort.
                if let Err(e) = sink.flush().await {
                    errored = true;
                    warn!(sink = %info.label, error = %e, "sink flush failed");
                }
                if let Err(e) = sink.close().await {
                    errored = true;
                    warn!(sink = %info.label, error = %e, "sink close failed");
                }
                (consumed, errored)
            }));
        }

        // Drive the source until end-of-stream, an error, or shutdown.
        let mut windows_in = 0u64;
        let mut source_error = None;
        loop {
            let next = tokio::select! {
                biased;
                _ = shutdown.cancelled() => break,
                r = source.next_window() => r,
            };
            match next {
                Ok(Some(batch)) => {
                    windows_in += 1;
                    stats.windows_in.fetch_add(1, Ordering::Relaxed);
                    let arc = Arc::new(batch);
                    for (i, tx) in senders.iter().enumerate() {
                        match tx.try_send(Arc::clone(&arc)) {
                            Ok(()) => {}
                            Err(mpsc::error::TrySendError::Full(_)) => {
                                dropped[i] += 1;
                                stats.sinks[i].dropped.fetch_add(1, Ordering::Relaxed);
                                warn!(sink = %infos[i].label, "sink channel full; dropping window (R33)");
                            }
                            Err(mpsc::error::TrySendError::Closed(_)) => {
                                // Sink task has exited; nothing more to do for it.
                            }
                        }
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    source_error = Some(e.to_string());
                    stats.source_error.store(true, Ordering::Relaxed);
                    warn!(source = %source_info.label, error = %e, "source failed; draining sinks");
                    break;
                }
            }
        }
        stats.ended.store(true, Ordering::Relaxed);

        // Closing the senders lets each sink drain its queue then finalize.
        drop(senders);
        let mut outcomes = Vec::with_capacity(handles.len());
        for (i, h) in handles.into_iter().enumerate() {
            let (consumed, errored) = h.await.unwrap_or((0, true));
            outcomes.push(SinkOutcome {
                info: infos[i].clone(),
                consumed,
                dropped: dropped[i],
                errored,
            });
        }

        PipelineReport {
            name,
            source: source_info,
            windows_in,
            source_error,
            sinks: outcomes,
        }
    }

    /// Build every pipeline in a manifest and run them concurrently to
    /// completion. Fails CLOSED at build time if any source/sink kind is not
    /// compiled into this build.
    pub async fn run_manifest(
        &self,
        manifest: &RuntimeManifest,
        shutdown: CancellationToken,
    ) -> Result<Vec<PipelineReport>> {
        let cfg = self.cfg; // Copy — each pipeline runs on its own owned Engine.
        let prepared = prepare_manifest(manifest)?;
        let mut handles = Vec::with_capacity(prepared.len());
        for pipeline in prepared {
            let sd = shutdown.clone();
            // Each spawned future OWNS everything (cfg is Copy) so it is 'static;
            // pipelines are independent long-lived loops that must run concurrently.
            handles.push(tokio::spawn(async move {
                let engine = Engine::new(cfg);
                engine
                    .run_pipeline(pipeline.name, pipeline.source, pipeline.sinks, sd)
                    .await
            }));
        }
        let mut reports = Vec::with_capacity(handles.len());
        for h in handles {
            if let Ok(r) = h.await {
                reports.push(r);
            }
        }
        Ok(reports)
    }
}
