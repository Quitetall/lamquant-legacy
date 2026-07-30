use std::collections::VecDeque;
use std::sync::Mutex;

use serde::Serialize;

use super::{lock_unpoisoned, LslClockRelation};

/// Evidence observed by the receiver. `sample_ordinal` is receiver-local; LSL
/// does not expose an upstream publisher sequence number through this wrapper.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LslInletEvidenceKind {
    TimestampRegression {
        previous_micros: i64,
        observed_micros: i64,
    },
    TimestampGap {
        expected_delta_micros: u64,
        observed_delta_micros: i64,
        tolerance_micros: u64,
    },
    InvalidTimestamp,
    SampleWidthMismatch {
        expected: usize,
        observed: usize,
    },
    WindowQueueOverload,
    StreamEnded {
        reason: String,
        partial_samples: usize,
    },
    EvidenceOverflow {
        suppressed_records: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LslInletEvidence {
    pub sample_ordinal: u64,
    pub window_seq: Option<u64>,
    pub clock_relation: LslClockRelation,
    pub issue: LslInletEvidenceKind,
}

#[derive(Debug)]
pub(super) struct EvidenceLedger {
    pub(super) records: VecDeque<LslInletEvidence>,
    pub(super) capacity: usize,
    pub(super) suppressed: u64,
}

impl EvidenceLedger {
    pub(super) fn record(&mut self, evidence: LslInletEvidence) {
        if self.records.len() == self.capacity {
            self.suppressed = self.suppressed.saturating_add(1);
        } else {
            self.records.push_back(evidence);
        }
    }

    pub(super) fn drain(&mut self, relation: &LslClockRelation) -> Vec<LslInletEvidence> {
        let mut records: Vec<_> = self.records.drain(..).collect();
        if self.suppressed != 0 {
            records.push(LslInletEvidence {
                sample_ordinal: 0,
                window_seq: None,
                clock_relation: relation.clone(),
                issue: LslInletEvidenceKind::EvidenceOverflow {
                    suppressed_records: std::mem::take(&mut self.suppressed),
                },
            });
        }
        records
    }
}

pub(super) fn record_inlet_evidence(
    ledger: &Mutex<EvidenceLedger>,
    relation: &LslClockRelation,
    sample_ordinal: u64,
    window_seq: Option<u64>,
    issue: LslInletEvidenceKind,
) {
    lock_unpoisoned(ledger).record(LslInletEvidence {
        sample_ordinal,
        window_seq,
        clock_relation: relation.clone(),
        issue,
    });
}
