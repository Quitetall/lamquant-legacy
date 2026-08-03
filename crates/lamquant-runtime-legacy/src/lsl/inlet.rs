use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use async_trait::async_trait;
use lamquant_visualization_lsl::{Inlet, SampleBuffer};

use crate::error::{Result, RuntimeError};
use crate::source::{Source, SourceInfo};
use crate::window::WindowBatch;

use super::evidence::{
    record_inlet_evidence, EvidenceLedger, LslInletEvidence, LslInletEvidenceKind,
};
use super::{
    bounded_reason, classify_timestamp_delta, lock_unpoisoned, release_inlet_startup,
    reserve_inlet_startup, retain_unresolved_inlet_startup, sample_period_micros,
    seconds_to_micros, seconds_to_nonnegative_millis, validate_bound, validate_buffered_values,
    LslClockRelation, DEFAULT_EVIDENCE_RECORDS, DEFAULT_INLET_QUEUE_WINDOWS,
    DEFAULT_TIMESTAMP_TOLERANCE_MICROS, MAX_QUEUE_WINDOWS, MAX_RECORDS, MAX_RESOLVE_TIMEOUT_MILLIS,
    MAX_WINDOW_SAMPLES,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LslInletConfig {
    pub window_samples: usize,
    pub resolve_timeout_millis: u64,
    pub queue_windows: usize,
    pub evidence_records: usize,
    pub timestamp_tolerance_micros: u64,
    pub clock_relation: LslClockRelation,
}

impl LslInletConfig {
    pub fn new(stream_name: &str, window_samples: usize, resolve_timeout_s: f64) -> Result<Self> {
        let timeout_millis = seconds_to_nonnegative_millis(resolve_timeout_s).ok_or_else(|| {
            RuntimeError::Source {
                name: stream_name.into(),
                msg: "resolve timeout must be finite and non-negative".into(),
            }
        })?;
        Ok(Self {
            window_samples,
            resolve_timeout_millis: timeout_millis,
            queue_windows: DEFAULT_INLET_QUEUE_WINDOWS,
            evidence_records: DEFAULT_EVIDENCE_RECORDS,
            timestamp_tolerance_micros: DEFAULT_TIMESTAMP_TOLERANCE_MICROS,
            clock_relation: LslClockRelation::unobserved(stream_name)?,
        })
    }

    fn validate(&self, stream_name: &str) -> Result<()> {
        super::validate_identity("stream_name", stream_name).map_err(|msg| {
            RuntimeError::Source {
                name: "lsl".into(),
                msg,
            }
        })?;
        validate_bound("inlet queue_windows", self.queue_windows, MAX_QUEUE_WINDOWS).map_err(
            |msg| RuntimeError::Source {
                name: stream_name.into(),
                msg,
            },
        )?;
        validate_bound("evidence_records", self.evidence_records, MAX_RECORDS).map_err(|msg| {
            RuntimeError::Source {
                name: stream_name.into(),
                msg,
            }
        })?;
        if self.window_samples == 0 {
            return Err(RuntimeError::Source {
                name: stream_name.into(),
                msg: "window_samples must be non-zero".into(),
            });
        }
        if self.window_samples > MAX_WINDOW_SAMPLES {
            return Err(RuntimeError::Source {
                name: stream_name.into(),
                msg: format!("window_samples exceeds {MAX_WINDOW_SAMPLES}"),
            });
        }
        if self.resolve_timeout_millis > MAX_RESOLVE_TIMEOUT_MILLIS {
            return Err(RuntimeError::Source {
                name: stream_name.into(),
                msg: format!("resolve_timeout_millis must be <= {MAX_RESOLVE_TIMEOUT_MILLIS}"),
            });
        }
        self.clock_relation.validate()
    }

    fn resolve_timeout_s(&self) -> f64 {
        self.resolve_timeout_millis as f64 / 1_000.0
    }
}

/// Resolves an LSL stream by name and streams its windows. Completed windows
/// enter a bounded queue; overload drops the complete window, retains its
/// sequence number as a visible gap, and records typed evidence.
pub struct LslInletSource {
    rx: tokio::sync::mpsc::Receiver<WindowBatch>,
    stream_name: String,
    n_channels: usize,
    sample_rate: f64,
    stop: Arc<AtomicBool>,
    evidence: Arc<Mutex<EvidenceLedger>>,
    clock_relation: LslClockRelation,
    _worker: JoinHandle<()>,
}

impl LslInletSource {
    pub fn open(stream_name: String, window: usize, resolve_timeout_s: f64) -> Result<Self> {
        let config = LslInletConfig::new(&stream_name, window, resolve_timeout_s)?;
        Self::open_with_config(stream_name, config)
    }

    pub fn open_with_config(stream_name: String, config: LslInletConfig) -> Result<Self> {
        config.validate(&stream_name)?;
        let stop = Arc::new(AtomicBool::new(false));
        let evidence = Arc::new(Mutex::new(EvidenceLedger {
            records: VecDeque::new(),
            capacity: config.evidence_records,
            suppressed: 0,
        }));
        let (ready_tx, ready_rx) =
            std::sync::mpsc::channel::<std::result::Result<(usize, f64), String>>();
        let (tx, rx) = tokio::sync::mpsc::channel::<WindowBatch>(config.queue_windows);

        let name = stream_name.clone();
        let stop_worker = Arc::clone(&stop);
        let evidence_worker = Arc::clone(&evidence);
        let relation_worker = config.clock_relation.clone();
        let window = config.window_samples;
        let queue_windows = config.queue_windows;
        let resolve_timeout_s = config.resolve_timeout_s();
        let timestamp_tolerance_micros = config.timestamp_tolerance_micros;
        let startup_id = reserve_inlet_startup().map_err(|msg| RuntimeError::Source {
            name: stream_name.clone(),
            msg,
        })?;
        let worker_result = std::thread::Builder::new()
            .name("lqrt-lsl-inlet".into())
            .spawn(move || {
                let inlet = match Inlet::resolve_by_name(&name, resolve_timeout_s) {
                    Ok(inlet) => inlet,
                    Err(error) => {
                        let _ = ready_tx.send(Err(error.to_string()));
                        return;
                    }
                };
                let n_ch = inlet.channel_count();
                let sample_rate = inlet.nominal_srate();
                if !sample_rate.is_finite() || sample_rate <= 0.0 {
                    let _ = ready_tx.send(Err(
                        "sampled LSL inlet requires a finite positive nominal rate".into(),
                    ));
                    return;
                }
                if let Err(reason) = validate_buffered_values(n_ch, window, queue_windows, "inlet")
                {
                    let _ = ready_tx.send(Err(reason));
                    return;
                }
                let mut buffer = match SampleBuffer::new(n_ch, window) {
                    Ok(buffer) => buffer,
                    Err(error) => {
                        let _ = ready_tx.send(Err(error.to_string()));
                        return;
                    }
                };
                let _ = ready_tx.send(Ok((n_ch, sample_rate)));

                let expected_delta_micros = sample_period_micros(sample_rate);
                let mut previous_ts = None;
                let mut first_window_ts = None;
                let mut samples_in_window = 0usize;
                let mut sample_ordinal = 0u64;
                let mut window_seq = 0u64;
                while !stop_worker.load(Ordering::Relaxed) {
                    let (sample, timestamp) = match inlet.pull_sample(1.0) {
                        Ok((sample, _)) if sample.is_empty() => continue,
                        Ok(value) => value,
                        Err(error) => {
                            record_inlet_evidence(
                                &evidence_worker,
                                &relation_worker,
                                sample_ordinal,
                                (samples_in_window != 0).then_some(window_seq),
                                LslInletEvidenceKind::StreamEnded {
                                    reason: bounded_reason(error.to_string()),
                                    partial_samples: samples_in_window,
                                },
                            );
                            break;
                        }
                    };

                    let timestamp_micros = match seconds_to_micros(timestamp) {
                        Some(value) => value,
                        None => {
                            record_inlet_evidence(
                                &evidence_worker,
                                &relation_worker,
                                sample_ordinal,
                                Some(window_seq),
                                LslInletEvidenceKind::InvalidTimestamp,
                            );
                            buffer = match SampleBuffer::new(n_ch, window) {
                                Ok(buffer) => buffer,
                                Err(error) => {
                                    record_inlet_evidence(
                                        &evidence_worker,
                                        &relation_worker,
                                        sample_ordinal,
                                        Some(window_seq),
                                        LslInletEvidenceKind::StreamEnded {
                                            reason: bounded_reason(error.to_string()),
                                            partial_samples: samples_in_window,
                                        },
                                    );
                                    break;
                                }
                            };
                            first_window_ts = None;
                            samples_in_window = 0;
                            previous_ts = None;
                            window_seq = window_seq.saturating_add(1);
                            sample_ordinal = sample_ordinal.saturating_add(1);
                            continue;
                        }
                    };
                    let timestamp_issue = previous_ts.and_then(|previous| {
                        classify_timestamp_delta(
                            previous,
                            timestamp_micros,
                            expected_delta_micros,
                            timestamp_tolerance_micros,
                        )
                    });
                    previous_ts = Some(timestamp_micros);

                    if sample.len() != n_ch {
                        if let Some(issue) = timestamp_issue {
                            record_inlet_evidence(
                                &evidence_worker,
                                &relation_worker,
                                sample_ordinal,
                                Some(window_seq),
                                issue,
                            );
                        }
                        record_inlet_evidence(
                            &evidence_worker,
                            &relation_worker,
                            sample_ordinal,
                            Some(window_seq),
                            LslInletEvidenceKind::SampleWidthMismatch {
                                expected: n_ch,
                                observed: sample.len(),
                            },
                        );
                        buffer = match SampleBuffer::new(n_ch, window) {
                            Ok(buffer) => buffer,
                            Err(error) => {
                                record_inlet_evidence(
                                    &evidence_worker,
                                    &relation_worker,
                                    sample_ordinal,
                                    Some(window_seq),
                                    LslInletEvidenceKind::StreamEnded {
                                        reason: bounded_reason(error.to_string()),
                                        partial_samples: samples_in_window,
                                    },
                                );
                                break;
                            }
                        };
                        first_window_ts = None;
                        samples_in_window = 0;
                        window_seq = window_seq.saturating_add(1);
                        sample_ordinal = sample_ordinal.saturating_add(1);
                        continue;
                    }
                    if let Some(issue) = timestamp_issue {
                        let skip_current =
                            matches!(issue, LslInletEvidenceKind::TimestampRegression { .. });
                        record_inlet_evidence(
                            &evidence_worker,
                            &relation_worker,
                            sample_ordinal,
                            Some(window_seq),
                            issue,
                        );
                        buffer = match SampleBuffer::new(n_ch, window) {
                            Ok(buffer) => buffer,
                            Err(error) => {
                                record_inlet_evidence(
                                    &evidence_worker,
                                    &relation_worker,
                                    sample_ordinal,
                                    Some(window_seq),
                                    LslInletEvidenceKind::StreamEnded {
                                        reason: bounded_reason(error.to_string()),
                                        partial_samples: samples_in_window,
                                    },
                                );
                                break;
                            }
                        };
                        first_window_ts = None;
                        samples_in_window = 0;
                        window_seq = window_seq.saturating_add(1);
                        if skip_current {
                            sample_ordinal = sample_ordinal.saturating_add(1);
                            continue;
                        }
                    }
                    if first_window_ts.is_none() {
                        first_window_ts = Some(timestamp_micros);
                    }
                    if buffer.push_sample(&sample).is_err() {
                        first_window_ts = None;
                        record_inlet_evidence(
                            &evidence_worker,
                            &relation_worker,
                            sample_ordinal,
                            Some(window_seq),
                            LslInletEvidenceKind::SampleWidthMismatch {
                                expected: n_ch,
                                observed: sample.len(),
                            },
                        );
                        buffer = match SampleBuffer::new(n_ch, window) {
                            Ok(buffer) => buffer,
                            Err(error) => {
                                record_inlet_evidence(
                                    &evidence_worker,
                                    &relation_worker,
                                    sample_ordinal,
                                    Some(window_seq),
                                    LslInletEvidenceKind::StreamEnded {
                                        reason: bounded_reason(error.to_string()),
                                        partial_samples: samples_in_window,
                                    },
                                );
                                break;
                            }
                        };
                        window_seq = window_seq.saturating_add(1);
                        samples_in_window = 0;
                        sample_ordinal = sample_ordinal.saturating_add(1);
                        continue;
                    }
                    sample_ordinal = sample_ordinal.saturating_add(1);
                    samples_in_window = samples_in_window.saturating_add(1);

                    if let Some(window) = buffer.flush_if_ready() {
                        let mut batch = WindowBatch::new(window_seq, window, sample_rate);
                        batch.first_ts_micros = first_window_ts.take();
                        samples_in_window = 0;
                        match tx.try_send(batch) {
                            Ok(()) => {}
                            Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                                record_inlet_evidence(
                                    &evidence_worker,
                                    &relation_worker,
                                    sample_ordinal,
                                    Some(window_seq),
                                    LslInletEvidenceKind::WindowQueueOverload,
                                );
                            }
                            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => break,
                        }
                        window_seq = window_seq.saturating_add(1);
                    }
                }
            });
        let worker = match worker_result {
            Ok(worker) => worker,
            Err(error) => {
                release_inlet_startup(startup_id);
                return Err(RuntimeError::Io(error));
            }
        };

        let ready_timeout = config.resolve_timeout_millis.saturating_add(5_000);
        let (n_channels, sample_rate) =
            match ready_rx.recv_timeout(std::time::Duration::from_millis(ready_timeout)) {
                Ok(Ok(ready)) => {
                    release_inlet_startup(startup_id);
                    ready
                }
                Ok(Err(msg)) => {
                    release_inlet_startup(startup_id);
                    let _ = worker.join();
                    return Err(RuntimeError::Source {
                        name: stream_name,
                        msg,
                    });
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    stop.store(true, Ordering::Relaxed);
                    retain_unresolved_inlet_startup(startup_id, worker);
                    return Err(RuntimeError::Source {
                        name: stream_name,
                        msg: format!("inlet worker was not ready within {ready_timeout} ms"),
                    });
                }
                Err(error) => {
                    release_inlet_startup(startup_id);
                    let _ = worker.join();
                    return Err(RuntimeError::Source {
                        name: stream_name,
                        msg: format!("inlet worker exited before ready: {error}"),
                    });
                }
            };

        Ok(Self {
            rx,
            stream_name,
            n_channels,
            sample_rate,
            stop,
            evidence,
            clock_relation: config.clock_relation,
            _worker: worker,
        })
    }

    pub fn clock_relation(&self) -> &LslClockRelation {
        &self.clock_relation
    }

    pub fn drain_evidence(&self) -> Vec<LslInletEvidence> {
        lock_unpoisoned(&self.evidence).drain(&self.clock_relation)
    }
}

impl Drop for LslInletSource {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

#[async_trait]
impl Source for LslInletSource {
    async fn next_window(&mut self) -> Result<Option<WindowBatch>> {
        Ok(self.rx.recv().await)
    }

    fn info(&self) -> SourceInfo {
        SourceInfo {
            kind: "lsl".into(),
            label: self.stream_name.clone(),
            n_channels: self.n_channels,
            sample_rate_hz: self.sample_rate,
            modality: None,
        }
    }
}
