//! Live Lab-Streaming-Layer source + sink (feature `lsl`, needs the `liblsl`
//! system library).
//!
//! LSL's `StreamInlet`/`StreamOutlet` are `!Send`, so each lives on a bounded
//! worker thread. This module does not claim that liblsl provides a durable or
//! exactly-once sink: outlet success means only that one process-local,
//! at-most-once attempt was accepted. The worker's eventual attempt result is
//! exposed as a receipt and never relabelled as remote delivery.

mod clock_contracts;
mod evidence;
mod inlet;
mod outlet;
mod startup_bounds;
#[cfg(test)]
mod tests;

use startup_bounds::*;

pub use clock_contracts::{LslClockId, LslClockRelation};
pub use evidence::{LslInletEvidence, LslInletEvidenceKind};
pub use inlet::{LslInletConfig, LslInletSource};
pub use outlet::{
    LslOutletConfig, LslOutletEffect, LslOutletReceipt, LslOutletReceiptState, LslOutletSink,
    LslOutletTimestampPolicy,
};

const DEFAULT_INLET_QUEUE_WINDOWS: usize = 64;
const DEFAULT_OUTLET_QUEUE_WINDOWS: usize = 64;
const DEFAULT_EVIDENCE_RECORDS: usize = 1_024;
const DEFAULT_RECEIPT_RECORDS: usize = 1_024;
const MAX_QUEUE_WINDOWS: usize = 4_096;
const MAX_RECORDS: usize = 65_536;
const MAX_IDENTITY_BYTES: usize = 512;
const MAX_BUFFERED_SAMPLE_VALUES: usize = 16 * 1024 * 1024;
const MAX_REASON_BYTES: usize = 1_024;
const MAX_CHANNELS: usize = 1_024;
const MAX_WINDOW_SAMPLES: usize = 65_536;
const DEFAULT_TIMESTAMP_TOLERANCE_MICROS: u64 = 1_000;
const DEFAULT_STARTUP_TIMEOUT_MILLIS: u64 = 5_000;
const DEFAULT_CLOSE_TIMEOUT_MILLIS: u64 = 1_000;
const MAX_RESOLVE_TIMEOUT_MILLIS: u64 = 60_000;
const MAX_STARTUP_TIMEOUT_MILLIS: u64 = 60_000;
const MAX_CLOSE_TIMEOUT_MILLIS: u64 = 60_000;
const MAX_UNRESOLVED_INLET_STARTUPS: usize = 4;
const MAX_OUTLET_WORKERS: usize = 64;
