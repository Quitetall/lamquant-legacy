//! [`RuntimeManifest`] — the runtime's routing config as data. Mirrors the
//! `IngestManifest` posture: a `version` that is fail-closed, and a list of
//! pipelines that each map one [`SourceSpec`] to one or more [`SinkSpec`]s.
//!
//! The spec enums are stable from the start (so the schema does not churn as
//! impls land); a variant whose concrete builder is not yet compiled in fails
//! CLOSED at build time (`UnknownSource`/`UnknownSink`), never silently.

use serde::{Deserialize, Serialize};

use crate::error::{Result, RuntimeError};
use crate::mem::{MemSink, MemSource};
use crate::sink::Sink;
use crate::source::Source;

/// The only manifest version this build accepts. Bump on a breaking schema change.
pub const MANIFEST_VERSION: u32 = 1;

/// A full runtime configuration: version + N independent pipelines.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeManifest {
    pub version: u32,
    #[serde(default)]
    pub pipelines: Vec<PipelineSpec>,
}

/// One source fanned out to a set of sinks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineSpec {
    /// Stable name (used in status + control).
    pub name: String,
    pub source: SourceSpec,
    pub sinks: Vec<SinkSpec>,
}

/// A source declaration. Internally-tagged by `kind`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum SourceSpec {
    /// In-memory reference source (testing / examples). Emits `n_windows`
    /// rectangular windows of `n_channels`×`n_samples` ramp data.
    Mem {
        label: String,
        n_channels: usize,
        n_samples: usize,
        n_windows: u64,
        sample_rate_hz: f64,
        /// Optional per-window pace in ms (0 = fast; a paced source is long-running).
        #[serde(default)]
        pace_ms: u64,
    },
    /// Replay an existing `.lml` as a window stream (feature `codec`) — a
    /// real-data stand-in for a live source.
    LmlReplay { path: String, window_samples: usize },
    /// Live LSL inlet (feature `lsl`, needs the liblsl system library).
    Lsl {
        stream_name: String,
        window_samples: usize,
        #[serde(default)]
        resolve_timeout_s: f64,
    },
    /// Watch a directory and stream the windows of every `.lml`/`.edf`/`.bdf`
    /// dropped into it (feature `watch`). `sample_rate_hz` is authoritative for
    /// `.lml` (EDF/BDF self-report); `extensions` defaults to lml/edf/bdf.
    DirWatch {
        dir: String,
        window_samples: usize,
        sample_rate_hz: f64,
        #[serde(default)]
        extensions: Vec<String>,
    },
}

/// A sink declaration. Internally-tagged by `kind`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum SinkSpec {
    /// In-memory reference sink (testing / examples) — counts windows.
    Mem { label: String },
    /// Compress each window into a rotating `.lml` file — built in Phase 1.
    LmlFile {
        path: String,
        #[serde(default)]
        rotate_windows: u64,
    },
    /// Re-publish each window to an LSL outlet — built in Phase 1.
    LslOutlet {
        outlet_name: String,
        #[serde(default)]
        source_id: String,
    },
    /// Record a content-addressed NEG evidence graph alongside the signal
    /// (feature `neg`, ADR 0114/0136). `producer` defaults to the pipeline name.
    Neg {
        path: String,
        #[serde(default)]
        producer: String,
        #[serde(default)]
        label: String,
    },
    /// Expose per-pipeline throughput on `GET /metrics` (feature `metrics`,
    /// Prometheus text). `bind_addr` e.g. `"127.0.0.1:9109"`; `pipeline` labels
    /// the counters (defaults to the pipeline name).
    Metrics {
        bind_addr: String,
        #[serde(default)]
        pipeline: String,
    },
}

impl RuntimeManifest {
    /// Parse + validate from TOML. Fail-closed on bad version, empty manifest,
    /// or a pipeline with no sinks.
    pub fn from_toml(text: &str) -> Result<Self> {
        let manifest: RuntimeManifest =
            toml::from_str(text).map_err(|e| RuntimeError::ManifestParse(e.to_string()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<()> {
        if self.version != MANIFEST_VERSION {
            return Err(RuntimeError::ManifestVersion {
                found: self.version,
                expected: MANIFEST_VERSION,
            });
        }
        if self.pipelines.is_empty() {
            return Err(RuntimeError::EmptyManifest);
        }
        for p in &self.pipelines {
            if p.sinks.is_empty() {
                return Err(RuntimeError::NoSinks {
                    pipeline: p.name.clone(),
                });
            }
        }
        Ok(())
    }
}

/// Build a boxed [`Source`] from its spec. Variants without a compiled-in impl
/// fail CLOSED (they parse, but cannot be constructed in this build).
pub fn build_source(spec: &SourceSpec) -> Result<Box<dyn Source>> {
    match spec {
        SourceSpec::Mem {
            label,
            n_channels,
            n_samples,
            n_windows,
            sample_rate_hz,
            pace_ms,
        } => Ok(Box::new(
            MemSource::ramp(
                label.clone(),
                *n_channels,
                *n_samples,
                *n_windows,
                *sample_rate_hz,
            )
            .with_pace(*pace_ms),
        )),
        #[cfg(feature = "codec")]
        SourceSpec::LmlReplay {
            path,
            window_samples,
        } => Ok(Box::new(crate::codec::LmlReplaySource::open(
            path,
            *window_samples,
        )?)),
        #[cfg(not(feature = "codec"))]
        SourceSpec::LmlReplay { .. } => Err(RuntimeError::UnknownSource {
            kind: "lml-replay (build with --features codec)".into(),
        }),
        #[cfg(feature = "lsl")]
        SourceSpec::Lsl {
            stream_name,
            window_samples,
            resolve_timeout_s,
        } => {
            let timeout = if *resolve_timeout_s > 0.0 {
                *resolve_timeout_s
            } else {
                5.0
            };
            Ok(Box::new(crate::lsl::LslInletSource::open(
                stream_name.clone(),
                *window_samples,
                timeout,
            )?))
        }
        #[cfg(not(feature = "lsl"))]
        SourceSpec::Lsl { .. } => Err(RuntimeError::UnknownSource {
            kind: "lsl (needs the liblsl system library; build with --features lsl)".into(),
        }),
        #[cfg(feature = "watch")]
        SourceSpec::DirWatch {
            dir,
            window_samples,
            sample_rate_hz,
            extensions,
        } => Ok(Box::new(crate::watch::DirWatchSource::open(
            dir.clone(),
            *window_samples,
            *sample_rate_hz,
            extensions.clone(),
        )?)),
        #[cfg(not(feature = "watch"))]
        SourceSpec::DirWatch { .. } => Err(RuntimeError::UnknownSource {
            kind: "dir-watch (build with --features watch)".into(),
        }),
    }
}

/// Build a boxed [`Sink`] from its spec. Variants without a compiled-in impl
/// fail CLOSED.
pub fn build_sink(spec: &SinkSpec) -> Result<Box<dyn Sink>> {
    match spec {
        SinkSpec::Mem { label } => Ok(Box::new(MemSink::new(label.clone()))),
        #[cfg(feature = "codec")]
        SinkSpec::LmlFile {
            path,
            rotate_windows,
        } => Ok(Box::new(crate::codec::LmlFileSink::new(
            path.clone(),
            *rotate_windows,
            0,
        ))),
        #[cfg(not(feature = "codec"))]
        SinkSpec::LmlFile { .. } => Err(RuntimeError::UnknownSink {
            kind: "lml-file (build with --features codec)".into(),
        }),
        #[cfg(feature = "lsl")]
        SinkSpec::LslOutlet {
            outlet_name,
            source_id,
        } => Ok(Box::new(crate::lsl::LslOutletSink::new(
            outlet_name.clone(),
            source_id.clone(),
        ))),
        #[cfg(not(feature = "lsl"))]
        SinkSpec::LslOutlet { .. } => Err(RuntimeError::UnknownSink {
            kind: "lsl-outlet (needs the liblsl system library; build with --features lsl)".into(),
        }),
        #[cfg(feature = "neg")]
        SinkSpec::Neg {
            path,
            producer,
            label,
        } => {
            let producer = if producer.is_empty() {
                "lamquant-runtime".to_string()
            } else {
                producer.clone()
            };
            let label = if label.is_empty() {
                path.clone()
            } else {
                label.clone()
            };
            Ok(Box::new(crate::neg_sink::NegProvenanceSink::new(
                path.clone(),
                producer,
                label,
            )))
        }
        #[cfg(not(feature = "neg"))]
        SinkSpec::Neg { .. } => Err(RuntimeError::UnknownSink {
            kind: "neg (build with --features neg)".into(),
        }),
        #[cfg(feature = "metrics")]
        SinkSpec::Metrics {
            bind_addr,
            pipeline,
        } => {
            let pipeline = if pipeline.is_empty() {
                bind_addr.clone()
            } else {
                pipeline.clone()
            };
            Ok(Box::new(crate::metrics_sink::MetricsSink::new(
                bind_addr.clone(),
                pipeline,
            )))
        }
        #[cfg(not(feature = "metrics"))]
        SinkSpec::Metrics { .. } => Err(RuntimeError::UnknownSink {
            kind: "metrics (build with --features metrics)".into(),
        }),
    }
}
