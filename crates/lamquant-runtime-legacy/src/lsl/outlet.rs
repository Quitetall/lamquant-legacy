use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use async_trait::async_trait;
use lamquant_visualization_lsl::Outlet;
use serde::Serialize;

use crate::error::{Result, RuntimeError};
use crate::sink::{Sink, SinkInfo};
use crate::window::WindowBatch;

use super::{
    bounded_reason, lock_unpoisoned, release_outlet_worker, reserve_outlet_worker,
    retain_unresolved_outlet_worker, to_sample_major_i32, validate_bound, validate_buffered_values,
    validate_identity, LslClockId, LslClockRelation, DEFAULT_CLOSE_TIMEOUT_MILLIS,
    DEFAULT_OUTLET_QUEUE_WINDOWS, DEFAULT_RECEIPT_RECORDS, DEFAULT_STARTUP_TIMEOUT_MILLIS,
    MAX_CLOSE_TIMEOUT_MILLIS, MAX_QUEUE_WINDOWS, MAX_RECORDS, MAX_STARTUP_TIMEOUT_MILLIS,
};

/// Honest LSL outlet effect. There is no remote acknowledgement or durable
/// reconciliation in liblsl, so transactional/exactly-once claims are invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LslOutletEffect {
    AtMostOnceEnqueuedProcessLocal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum LslOutletReceiptState {
    Enqueued,
    Attempted {
        samples: usize,
    },
    Failed {
        reason: String,
    },
    Gap {
        previous_enqueued_seq: Option<u64>,
        first_missing_seq: u64,
        last_missing_seq: u64,
    },
}

/// The current visualization wrapper does not expose liblsl's explicit
/// timestamp push API. Input timestamps therefore remain provenance only and
/// liblsl assigns outlet time at the local push.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LslOutletTimestampPolicy {
    LiblslLocalClockAtPush,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LslOutletReceipt {
    pub idempotency_key: String,
    pub window_seq: u64,
    pub input_first_ts_micros_not_transmitted: Option<i64>,
    pub input_to_outlet_clock: LslClockRelation,
    pub outlet_clock: LslClockId,
    pub timestamp_policy: LslOutletTimestampPolicy,
    pub effect: LslOutletEffect,
    pub state: LslOutletReceiptState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LslOutletConfig {
    pub queue_windows: usize,
    pub receipt_records: usize,
    pub outlet_clock: LslClockId,
    pub input_to_outlet_clock: LslClockRelation,
    pub startup_timeout_millis: u64,
    pub close_timeout_millis: u64,
}

impl LslOutletConfig {
    pub fn new(source_id: &str) -> Result<Self> {
        let _ = source_id;
        Ok(Self::process_local())
    }

    pub fn process_local() -> Self {
        let outlet_clock = LslClockId("lsl.local-clock:process-unbound".into());
        Self {
            queue_windows: DEFAULT_OUTLET_QUEUE_WINDOWS,
            receipt_records: DEFAULT_RECEIPT_RECORDS,
            input_to_outlet_clock: LslClockRelation {
                publisher_clock: LslClockId("runtime.window.timestamp:unbound".into()),
                receiver_clock: outlet_clock.clone(),
                offset_micros: None,
                uncertainty_micros: None,
                observed_at_receiver_micros: None,
            },
            outlet_clock,
            startup_timeout_millis: DEFAULT_STARTUP_TIMEOUT_MILLIS,
            close_timeout_millis: DEFAULT_CLOSE_TIMEOUT_MILLIS,
        }
    }

    fn validate(&self, outlet_name: &str) -> Result<()> {
        validate_identity("outlet_name", outlet_name).map_err(|msg| RuntimeError::Sink {
            name: "lsl-outlet".into(),
            msg,
        })?;
        validate_identity("outlet_clock", self.outlet_clock.as_str()).map_err(|msg| {
            RuntimeError::Sink {
                name: outlet_name.into(),
                msg,
            }
        })?;
        self.input_to_outlet_clock
            .validate()
            .map_err(|error| RuntimeError::Sink {
                name: outlet_name.into(),
                msg: error.to_string(),
            })?;
        if self.input_to_outlet_clock.receiver_clock != self.outlet_clock {
            return Err(RuntimeError::Sink {
                name: outlet_name.into(),
                msg: "input clock relation receiver must equal outlet_clock".into(),
            });
        }
        validate_bound(
            "outlet queue_windows",
            self.queue_windows,
            MAX_QUEUE_WINDOWS,
        )
        .map_err(|msg| RuntimeError::Sink {
            name: outlet_name.into(),
            msg,
        })?;
        if self.startup_timeout_millis == 0
            || self.startup_timeout_millis > MAX_STARTUP_TIMEOUT_MILLIS
        {
            return Err(RuntimeError::Sink {
                name: outlet_name.into(),
                msg: format!("startup_timeout_millis must be in 1..={MAX_STARTUP_TIMEOUT_MILLIS}"),
            });
        }
        validate_bound("receipt_records", self.receipt_records, MAX_RECORDS).map_err(|msg| {
            RuntimeError::Sink {
                name: outlet_name.into(),
                msg,
            }
        })?;
        if self.close_timeout_millis == 0 || self.close_timeout_millis > MAX_CLOSE_TIMEOUT_MILLIS {
            return Err(RuntimeError::Sink {
                name: outlet_name.into(),
                msg: format!("close_timeout_millis must be in 1..={MAX_CLOSE_TIMEOUT_MILLIS}"),
            });
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(super) struct OutletCommand {
    pub(super) idempotency_key: String,
    pub(super) samples: Vec<Vec<i32>>,
}

#[derive(Debug)]
pub(super) struct ReceiptLedger {
    pub(super) receipts: VecDeque<LslOutletReceipt>,
    pub(super) capacity: usize,
}

impl ReceiptLedger {
    pub(super) fn prepare_all(
        &mut self,
        receipts: Vec<LslOutletReceipt>,
    ) -> std::result::Result<(), &'static str> {
        if self.receipts.len().saturating_add(receipts.len()) > self.capacity {
            return Err("outlet receipt capacity exhausted; drain completed receipts");
        }
        for (index, receipt) in receipts.iter().enumerate() {
            if self
                .receipts
                .iter()
                .any(|existing| existing.idempotency_key == receipt.idempotency_key)
                || receipts[..index]
                    .iter()
                    .any(|existing| existing.idempotency_key == receipt.idempotency_key)
            {
                return Err("idempotency key already has a retained outlet receipt");
            }
        }
        self.receipts.extend(receipts);
        Ok(())
    }

    pub(super) fn finish(&mut self, key: &str, state: LslOutletReceiptState) {
        if let Some(receipt) = self
            .receipts
            .iter_mut()
            .find(|receipt| receipt.idempotency_key == key)
        {
            receipt.state = state;
        }
    }

    pub(super) fn drain_completed(&mut self) -> Vec<LslOutletReceipt> {
        let mut completed = Vec::new();
        let mut pending = VecDeque::with_capacity(self.receipts.len());
        while let Some(receipt) = self.receipts.pop_front() {
            if matches!(receipt.state, LslOutletReceiptState::Enqueued) {
                pending.push_back(receipt);
            } else {
                completed.push(receipt);
            }
        }
        self.receipts = pending;
        completed
    }
}

/// Re-publishes each window to a bounded LSL outlet worker.
pub struct LslOutletSink {
    outlet_name: String,
    source_id: String,
    pub(super) config: LslOutletConfig,
    pub(super) tx: Option<std::sync::mpsc::SyncSender<OutletCommand>>,
    pub(super) worker: Option<JoinHandle<()>>,
    worker_slot: Option<u64>,
    receipts: Arc<Mutex<ReceiptLedger>>,
    pub(super) last_enqueued_seq: Option<u64>,
    pub(super) layout: Option<(usize, u64)>,
}

impl LslOutletSink {
    pub fn new(outlet_name: String, source_id: String) -> Self {
        let config = LslOutletConfig::process_local();
        Self::with_config(outlet_name, source_id, config)
    }

    pub fn try_new(outlet_name: String, source_id: String) -> Result<Self> {
        let config = LslOutletConfig::new(&source_id)?;
        config.validate(&outlet_name)?;
        validate_identity("source_id", &source_id).map_err(|msg| RuntimeError::Sink {
            name: outlet_name.clone(),
            msg,
        })?;
        Ok(Self::with_config(outlet_name, source_id, config))
    }

    pub fn try_with_config(
        outlet_name: String,
        source_id: String,
        config: LslOutletConfig,
    ) -> Result<Self> {
        config.validate(&outlet_name)?;
        validate_identity("source_id", &source_id).map_err(|msg| RuntimeError::Sink {
            name: outlet_name.clone(),
            msg,
        })?;
        Ok(Self::with_config(outlet_name, source_id, config))
    }

    fn with_config(outlet_name: String, source_id: String, config: LslOutletConfig) -> Self {
        let receipt_records = config.receipt_records;
        Self {
            outlet_name,
            source_id,
            config,
            tx: None,
            worker: None,
            worker_slot: None,
            receipts: Arc::new(Mutex::new(ReceiptLedger {
                receipts: VecDeque::new(),
                capacity: receipt_records,
            })),
            last_enqueued_seq: None,
            layout: None,
        }
    }

    pub const fn effect(&self) -> LslOutletEffect {
        LslOutletEffect::AtMostOnceEnqueuedProcessLocal
    }

    pub fn outlet_clock(&self) -> &LslClockId {
        &self.config.outlet_clock
    }

    pub fn idempotency_key(&self, batch: &WindowBatch) -> String {
        format!("lsl:{}:window:{}", self.source_id, batch.seq)
    }

    /// Returns final local attempt receipts. Enqueued work remains retained so
    /// its eventual result cannot be lost by an early drain.
    pub fn drain_receipts(&self) -> Vec<LslOutletReceipt> {
        lock_unpoisoned(&self.receipts).drain_completed()
    }

    async fn start(&mut self, n_channels: usize, sample_rate: f64) -> Result<()> {
        self.config.validate(&self.outlet_name)?;
        if let Some(existing) = self.worker.as_ref() {
            if existing.is_finished() {
                if let Some(finished) = self.worker.take() {
                    let _ = finished.join();
                }
                if let Some(slot) = self.worker_slot.take() {
                    release_outlet_worker(slot);
                }
            } else {
                return Err(RuntimeError::Sink {
                    name: self.outlet_name.clone(),
                    msg: "previous outlet startup remains unresolved; refusing another worker"
                        .into(),
                });
            }
        }
        validate_identity("source_id", &self.source_id).map_err(|msg| RuntimeError::Sink {
            name: self.outlet_name.clone(),
            msg,
        })?;
        let worker_slot = reserve_outlet_worker().map_err(|msg| RuntimeError::Sink {
            name: self.outlet_name.clone(),
            msg,
        })?;
        let labels: Vec<String> = (0..n_channels).map(|index| format!("ch{index}")).collect();
        let (ready_tx, ready_rx) =
            tokio::sync::oneshot::channel::<std::result::Result<(), String>>();
        let (command_tx, command_rx) =
            std::sync::mpsc::sync_channel::<OutletCommand>(self.config.queue_windows);
        let name = self.outlet_name.clone();
        let source_id = self.source_id.clone();
        let receipts = Arc::clone(&self.receipts);
        let worker_result = std::thread::Builder::new()
            .name("lqrt-lsl-outlet".into())
            .spawn(move || {
                let outlet = match Outlet::create_outlet(&labels, sample_rate, &name, &source_id) {
                    Ok(outlet) => {
                        let _ = ready_tx.send(Ok(()));
                        outlet
                    }
                    Err(error) => {
                        let _ = ready_tx.send(Err(error.to_string()));
                        return;
                    }
                };
                while let Ok(command) = command_rx.recv() {
                    let sample_count = command.samples.len();
                    let state = match outlet.push_chunk(command.samples) {
                        Ok(_) => LslOutletReceiptState::Attempted {
                            samples: sample_count,
                        },
                        Err(error) => LslOutletReceiptState::Failed {
                            reason: bounded_reason(error.to_string()),
                        },
                    };
                    lock_unpoisoned(&receipts).finish(&command.idempotency_key, state);
                }
            });
        let worker = match worker_result {
            Ok(worker) => worker,
            Err(error) => {
                release_outlet_worker(worker_slot);
                return Err(RuntimeError::Io(error));
            }
        };
        self.worker = Some(worker);
        self.worker_slot = Some(worker_slot);
        tokio::time::timeout(
            std::time::Duration::from_millis(self.config.startup_timeout_millis),
            ready_rx,
        )
        .await
        .map_err(|_| RuntimeError::Sink {
            name: self.outlet_name.clone(),
            msg: format!(
                "outlet worker was not ready within {} ms",
                self.config.startup_timeout_millis
            ),
        })?
        .map_err(|_| RuntimeError::Sink {
            name: self.outlet_name.clone(),
            msg: "outlet worker exited before reporting readiness".into(),
        })?
        .map_err(|msg| RuntimeError::Sink {
            name: self.outlet_name.clone(),
            msg,
        })?;
        self.tx = Some(command_tx);
        Ok(())
    }
}

#[async_trait]
impl Sink for LslOutletSink {
    /// Startup is cancellation-friendly. After it completes, there is no
    /// suspension point between accepting the bounded local enqueue and
    /// returning its status. Success is not a publication acknowledgement;
    /// inspect `drain_receipts` for the worker's local attempt result.
    async fn consume(&mut self, batch: &WindowBatch) -> Result<()> {
        if !batch.is_rectangular() {
            return Err(RuntimeError::Sink {
                name: self.outlet_name.clone(),
                msg: format!("non-rectangular window seq={}", batch.seq),
            });
        }
        let layout = (batch.n_channels(), batch.sample_rate_millihz);
        if batch.sample_rate_millihz == 0 {
            return Err(RuntimeError::Sink {
                name: self.outlet_name.clone(),
                msg: "sampled LSL outlet requires a positive nominal rate".into(),
            });
        }
        validate_buffered_values(
            batch.n_channels(),
            batch.n_samples(),
            self.config.queue_windows,
            "outlet",
        )
        .map_err(|msg| RuntimeError::Sink {
            name: self.outlet_name.clone(),
            msg,
        })?;
        if let Some(expected) = self.layout {
            if layout != expected {
                return Err(RuntimeError::Sink {
                    name: self.outlet_name.clone(),
                    msg: format!(
                        "stream layout changed from {expected:?} to {layout:?} at seq={}",
                        batch.seq
                    ),
                });
            }
        }
        if let Some(previous) = self.last_enqueued_seq {
            if batch.seq <= previous {
                return Err(RuntimeError::Sink {
                    name: self.outlet_name.clone(),
                    msg: format!(
                        "duplicate or out-of-order window seq={} after seq={previous}",
                        batch.seq
                    ),
                });
            }
        }
        let key = self.idempotency_key(batch);
        let samples = to_sample_major_i32(batch).map_err(|msg| RuntimeError::Sink {
            name: self.outlet_name.clone(),
            msg,
        })?;
        if self.tx.is_none() {
            self.start(batch.n_channels(), batch.sample_rate_hz())
                .await?;
            self.layout = Some(layout);
        }

        {
            let mut receipts = lock_unpoisoned(&self.receipts);
            let expected_seq = self
                .last_enqueued_seq
                .map_or(0, |previous| previous.saturating_add(1));
            let mut prepared = Vec::with_capacity(2);
            if batch.seq > expected_seq {
                prepared.push(LslOutletReceipt {
                    idempotency_key: format!(
                        "lsl:{}:gap:{}-{}",
                        self.source_id,
                        expected_seq,
                        batch.seq - 1
                    ),
                    window_seq: expected_seq,
                    input_first_ts_micros_not_transmitted: None,
                    input_to_outlet_clock: self.config.input_to_outlet_clock.clone(),
                    outlet_clock: self.config.outlet_clock.clone(),
                    timestamp_policy: LslOutletTimestampPolicy::LiblslLocalClockAtPush,
                    effect: self.effect(),
                    state: LslOutletReceiptState::Gap {
                        previous_enqueued_seq: self.last_enqueued_seq,
                        first_missing_seq: expected_seq,
                        last_missing_seq: batch.seq - 1,
                    },
                });
            }
            prepared.push(LslOutletReceipt {
                idempotency_key: key.clone(),
                window_seq: batch.seq,
                input_first_ts_micros_not_transmitted: batch.first_ts_micros,
                input_to_outlet_clock: self.config.input_to_outlet_clock.clone(),
                outlet_clock: self.config.outlet_clock.clone(),
                timestamp_policy: LslOutletTimestampPolicy::LiblslLocalClockAtPush,
                effect: self.effect(),
                state: LslOutletReceiptState::Enqueued,
            });
            receipts
                .prepare_all(prepared)
                .map_err(|msg| RuntimeError::Sink {
                    name: self.outlet_name.clone(),
                    msg: msg.into(),
                })?;
        }

        let command = OutletCommand {
            idempotency_key: key.clone(),
            samples,
        };
        let sender = self.tx.as_ref().ok_or_else(|| RuntimeError::Sink {
            name: self.outlet_name.clone(),
            msg: "outlet worker is unavailable after startup".into(),
        })?;
        match sender.try_send(command) {
            Ok(()) => {
                self.last_enqueued_seq = Some(batch.seq);
                Ok(())
            }
            Err(error) => {
                lock_unpoisoned(&self.receipts).finish(
                    &key,
                    LslOutletReceiptState::Failed {
                        reason: bounded_reason(error.to_string()),
                    },
                );
                Err(RuntimeError::Sink {
                    name: self.outlet_name.clone(),
                    msg: format!("bounded outlet queue rejected seq={}: {error}", batch.seq),
                })
            }
        }
    }

    async fn flush(&mut self) -> Result<()> {
        if lock_unpoisoned(&self.receipts)
            .receipts
            .iter()
            .any(|receipt| matches!(receipt.state, LslOutletReceiptState::Enqueued))
        {
            return Err(RuntimeError::Sink {
                name: self.outlet_name.clone(),
                msg: "outlet still has process-local enqueued attempts".into(),
            });
        }
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        self.tx = None;
        let deadline = tokio::time::Instant::now()
            + std::time::Duration::from_millis(self.config.close_timeout_millis);
        while let Some(worker) = self.worker.as_ref() {
            if worker.is_finished() {
                let finished = self.worker.take().ok_or_else(|| RuntimeError::Sink {
                    name: self.outlet_name.clone(),
                    msg: "finished outlet worker handle disappeared".into(),
                })?;
                finished.join().map_err(|_| RuntimeError::Sink {
                    name: self.outlet_name.clone(),
                    msg: "outlet worker panicked".into(),
                })?;
                if let Some(slot) = self.worker_slot.take() {
                    release_outlet_worker(slot);
                }
                return Ok(());
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(RuntimeError::Sink {
                    name: self.outlet_name.clone(),
                    msg: format!(
                        "outlet worker did not stop within {} ms; liblsl call or queue drain remains unresolved",
                        self.config.close_timeout_millis
                    ),
                });
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        Ok(())
    }

    fn info(&self) -> SinkInfo {
        SinkInfo {
            kind: "lsl-outlet-at-most-once-enqueued".into(),
            label: self.outlet_name.clone(),
        }
    }
}

impl Drop for LslOutletSink {
    fn drop(&mut self) {
        self.tx = None;
        if self.worker.as_ref().is_some_and(JoinHandle::is_finished) {
            if let Some(worker) = self.worker.take() {
                let _ = worker.join();
            }
            if let Some(slot) = self.worker_slot.take() {
                release_outlet_worker(slot);
            }
        } else if self.worker.is_some() {
            if let (Some(worker), Some(slot)) = (self.worker.take(), self.worker_slot.take()) {
                retain_unresolved_outlet_worker(slot, worker);
            }
            tracing::warn!(
                outlet = %self.outlet_name,
                "dropping LSL outlet with unresolved process-local attempts"
            );
        } else if let Some(slot) = self.worker_slot.take() {
            release_outlet_worker(slot);
        }
    }
}
