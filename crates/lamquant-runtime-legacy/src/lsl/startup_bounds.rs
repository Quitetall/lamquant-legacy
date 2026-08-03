use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::thread::JoinHandle;

use super::LslInletEvidenceKind;
use super::{
    MAX_BUFFERED_SAMPLE_VALUES, MAX_CHANNELS, MAX_IDENTITY_BYTES, MAX_OUTLET_WORKERS,
    MAX_REASON_BYTES, MAX_UNRESOLVED_INLET_STARTUPS, MAX_WINDOW_SAMPLES,
};

#[derive(Debug)]
pub(super) struct InletStartupEntry {
    id: u64,
    worker: Option<JoinHandle<()>>,
}

static INLET_STARTUP_REGISTRY: OnceLock<Mutex<Vec<InletStartupEntry>>> = OnceLock::new();
static NEXT_INLET_STARTUP_ID: AtomicU64 = AtomicU64::new(1);
static OUTLET_WORKER_REGISTRY: OnceLock<Mutex<Vec<InletStartupEntry>>> = OnceLock::new();
static NEXT_OUTLET_WORKER_ID: AtomicU64 = AtomicU64::new(1);

pub(super) fn inlet_startup_registry() -> &'static Mutex<Vec<InletStartupEntry>> {
    INLET_STARTUP_REGISTRY.get_or_init(|| Mutex::new(Vec::new()))
}

fn reap_inlet_startups(entries: &mut Vec<InletStartupEntry>) {
    let mut retained = Vec::with_capacity(entries.len());
    for mut entry in std::mem::take(entries) {
        if entry.worker.as_ref().is_some_and(JoinHandle::is_finished) {
            if let Some(worker) = entry.worker.take() {
                let _ = worker.join();
            }
        } else {
            retained.push(entry);
        }
    }
    *entries = retained;
}

pub(super) fn reserve_inlet_startup() -> std::result::Result<u64, String> {
    let mut entries = lock_unpoisoned(inlet_startup_registry());
    reap_inlet_startups(&mut entries);
    if entries.len() >= MAX_UNRESOLVED_INLET_STARTUPS {
        return Err(format!(
            "{MAX_UNRESOLVED_INLET_STARTUPS} inlet startups remain unresolved"
        ));
    }
    let id = NEXT_INLET_STARTUP_ID.fetch_add(1, Ordering::Relaxed);
    entries.push(InletStartupEntry { id, worker: None });
    Ok(id)
}

pub(super) fn release_inlet_startup(id: u64) {
    lock_unpoisoned(inlet_startup_registry()).retain(|entry| entry.id != id);
}

pub(super) fn retain_unresolved_inlet_startup(id: u64, worker: JoinHandle<()>) {
    let mut entries = lock_unpoisoned(inlet_startup_registry());
    if let Some(entry) = entries.iter_mut().find(|entry| entry.id == id) {
        entry.worker = Some(worker);
    } else {
        // Defensive recovery: admission already bounded this startup, so
        // retaining it remains safer than detaching it.
        entries.push(InletStartupEntry {
            id,
            worker: Some(worker),
        });
    }
}

fn outlet_worker_registry() -> &'static Mutex<Vec<InletStartupEntry>> {
    OUTLET_WORKER_REGISTRY.get_or_init(|| Mutex::new(Vec::new()))
}

pub(super) fn reserve_outlet_worker() -> std::result::Result<u64, String> {
    let mut entries = lock_unpoisoned(outlet_worker_registry());
    reap_inlet_startups(&mut entries);
    if entries.len() >= MAX_OUTLET_WORKERS {
        return Err(format!("outlet worker limit {MAX_OUTLET_WORKERS} reached"));
    }
    let id = NEXT_OUTLET_WORKER_ID.fetch_add(1, Ordering::Relaxed);
    entries.push(InletStartupEntry { id, worker: None });
    Ok(id)
}

pub(super) fn release_outlet_worker(id: u64) {
    lock_unpoisoned(outlet_worker_registry()).retain(|entry| entry.id != id);
}

pub(super) fn retain_unresolved_outlet_worker(id: u64, worker: JoinHandle<()>) {
    let mut entries = lock_unpoisoned(outlet_worker_registry());
    if let Some(entry) = entries.iter_mut().find(|entry| entry.id == id) {
        entry.worker = Some(worker);
    } else {
        entries.push(InletStartupEntry {
            id,
            worker: Some(worker),
        });
    }
}

pub(super) fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

pub(super) fn to_sample_major_i32(
    batch: &crate::window::WindowBatch,
) -> std::result::Result<Vec<Vec<i32>>, String> {
    let samples = batch.n_samples();
    let channels = batch.n_channels();
    (0..samples)
        .map(|sample| {
            (0..channels)
                .map(|channel| {
                    i32::try_from(batch.channels[channel][sample]).map_err(|_| {
                        format!(
                            "sample value outside i32 LSL range at channel={channel} sample={sample}"
                        )
                    })
                })
                .collect()
        })
        .collect()
}

pub(super) fn validate_bound(
    name: &str,
    value: usize,
    max: usize,
) -> std::result::Result<(), String> {
    if value == 0 || value > max {
        Err(format!("{name} must be in 1..={max}"))
    } else {
        Ok(())
    }
}

pub(super) fn validate_identity(name: &str, value: &str) -> std::result::Result<(), String> {
    if value.trim().is_empty() || value.len() > MAX_IDENTITY_BYTES {
        Err(format!(
            "{name} must contain 1..={MAX_IDENTITY_BYTES} bytes"
        ))
    } else {
        Ok(())
    }
}

pub(super) fn validate_buffered_values(
    channels: usize,
    samples_per_window: usize,
    queue_windows: usize,
    direction: &str,
) -> std::result::Result<(), String> {
    if channels == 0 || channels > MAX_CHANNELS {
        return Err(format!(
            "{direction} channel count must be in 1..={MAX_CHANNELS}; requested {channels}"
        ));
    }
    if samples_per_window == 0 || samples_per_window > MAX_WINDOW_SAMPLES {
        return Err(format!(
            "{direction} samples per window must be in 1..={MAX_WINDOW_SAMPLES}; requested {samples_per_window}"
        ));
    }
    // In addition to the bounded channel, account for one in-progress worker
    // command/window and one producer-side command/window.
    let resident_windows = queue_windows
        .checked_add(2)
        .ok_or_else(|| format!("{direction} resident window count overflow"))?;
    let values = channels
        .checked_mul(samples_per_window)
        .and_then(|per_window| per_window.checked_mul(resident_windows))
        .ok_or_else(|| format!("{direction} buffered sample-value count overflow"))?;
    if values > MAX_BUFFERED_SAMPLE_VALUES {
        Err(format!(
            "{direction} buffered sample values must be in 1..={MAX_BUFFERED_SAMPLE_VALUES}; requested {values}"
        ))
    } else {
        Ok(())
    }
}

pub(super) fn bounded_reason(mut reason: String) -> String {
    if reason.len() <= MAX_REASON_BYTES {
        return reason;
    }
    let mut boundary = MAX_REASON_BYTES;
    while !reason.is_char_boundary(boundary) {
        boundary -= 1;
    }
    reason.truncate(boundary);
    reason
}

pub(super) fn seconds_to_nonnegative_millis(value: f64) -> Option<u64> {
    if !value.is_finite() || value < 0.0 || value > u64::MAX as f64 / 1_000.0 {
        return None;
    }
    Some((value * 1_000.0).round() as u64)
}

pub(super) fn seconds_to_micros(value: f64) -> Option<i64> {
    if !value.is_finite()
        || value < i64::MIN as f64 / 1_000_000.0
        || value > i64::MAX as f64 / 1_000_000.0
    {
        return None;
    }
    Some((value * 1_000_000.0).round() as i64)
}

pub(super) fn sample_period_micros(sample_rate: f64) -> Option<u64> {
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return None;
    }
    let period = (1_000_000.0 / sample_rate).round();
    (period >= 1.0 && period <= i64::MAX as f64).then_some(period as u64)
}

pub(super) fn classify_timestamp_delta(
    previous_micros: i64,
    observed_micros: i64,
    expected_delta_micros: Option<u64>,
    tolerance_micros: u64,
) -> Option<LslInletEvidenceKind> {
    let observed_delta = observed_micros.saturating_sub(previous_micros);
    if observed_delta <= 0 {
        return Some(LslInletEvidenceKind::TimestampRegression {
            previous_micros,
            observed_micros,
        });
    }
    let expected = expected_delta_micros?;
    let effective_tolerance = tolerance_micros.min(expected.saturating_sub(1) / 2);
    (observed_delta.abs_diff(expected as i64) > effective_tolerance).then_some(
        LslInletEvidenceKind::TimestampGap {
            expected_delta_micros: expected,
            observed_delta_micros: observed_delta,
            tolerance_micros: effective_tolerance,
        },
    )
}
