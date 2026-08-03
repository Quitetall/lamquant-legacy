use std::collections::VecDeque;

use crate::sink::Sink;
use crate::window::WindowBatch;

use super::evidence::EvidenceLedger;
use super::outlet::OutletCommand;
use super::outlet::ReceiptLedger;
use super::*;

#[test]
fn clock_relation_rejects_partial_measurement() {
    let mut relation = LslClockRelation::unobserved("eeg").unwrap();
    relation.offset_micros = Some(7);
    assert!(relation.validate().is_err());
    relation.uncertainty_micros = Some(3);
    relation.observed_at_receiver_micros = Some(11);
    assert!(relation.validate().is_err());
}

#[test]
fn clock_identity_and_relation_deserialization_fail_closed() {
    assert!(serde_json::from_str::<LslClockId>("\"\"").is_err());
    let measured = serde_json::json!({
        "publisher_clock": "lsl.publisher:one",
        "receiver_clock": "host.monotonic:one",
        "offset_micros": 7,
        "uncertainty_micros": 3,
        "observed_at_receiver_micros": 11
    });
    assert!(serde_json::from_value::<LslClockRelation>(measured).is_err());
}

#[test]
fn evidence_overflow_is_never_silent() {
    let relation = LslClockRelation::unobserved("eeg").unwrap();
    let mut ledger = EvidenceLedger {
        records: VecDeque::new(),
        capacity: 1,
        suppressed: 0,
    };
    for ordinal in 0..3 {
        ledger.record(LslInletEvidence {
            sample_ordinal: ordinal,
            window_seq: None,
            clock_relation: relation.clone(),
            issue: LslInletEvidenceKind::InvalidTimestamp,
        });
    }
    let records = ledger.drain(&relation);
    assert_eq!(records.len(), 2);
    assert!(matches!(
        records[1].issue,
        LslInletEvidenceKind::EvidenceOverflow {
            suppressed_records: 2
        }
    ));
}

#[test]
fn completed_receipts_drain_but_enqueued_receipts_remain() {
    let mut ledger = ReceiptLedger {
        receipts: VecDeque::new(),
        capacity: 2,
    };
    for seq in 0..2 {
        ledger
            .prepare_all(vec![LslOutletReceipt {
                idempotency_key: format!("key-{seq}"),
                window_seq: seq,
                input_first_ts_micros_not_transmitted: None,
                input_to_outlet_clock: LslOutletConfig::process_local().input_to_outlet_clock,
                outlet_clock: LslOutletConfig::process_local().outlet_clock,
                timestamp_policy: LslOutletTimestampPolicy::LiblslLocalClockAtPush,
                effect: LslOutletEffect::AtMostOnceEnqueuedProcessLocal,
                state: LslOutletReceiptState::Enqueued,
            }])
            .unwrap();
    }
    let duplicate = ledger.receipts[0].clone();
    assert!(ledger.prepare_all(vec![duplicate]).is_err());
    ledger.finish("key-0", LslOutletReceiptState::Attempted { samples: 4 });
    let completed = ledger.drain_completed();
    assert_eq!(completed.len(), 1);
    assert_eq!(ledger.receipts.len(), 1);
    assert!(matches!(
        ledger.receipts[0].state,
        LslOutletReceiptState::Enqueued
    ));
}

#[test]
fn time_conversions_reject_nonfinite_values() {
    assert_eq!(startup_bounds::seconds_to_micros(1.25), Some(1_250_000));
    assert_eq!(startup_bounds::seconds_to_micros(f64::NAN), None);
    assert_eq!(startup_bounds::seconds_to_nonnegative_millis(-1.0), None);
    assert_eq!(startup_bounds::sample_period_micros(250.0), Some(4_000));
    assert_eq!(startup_bounds::sample_period_micros(0.0), None);
}

#[test]
fn timestamp_classifier_distinguishes_regression_gap_and_jitter() {
    assert!(matches!(
        startup_bounds::classify_timestamp_delta(10, 9, Some(4), 1),
        Some(LslInletEvidenceKind::TimestampRegression { .. })
    ));
    assert!(matches!(
        startup_bounds::classify_timestamp_delta(10, 30, Some(4), 1),
        Some(LslInletEvidenceKind::TimestampGap {
            observed_delta_micros: 20,
            ..
        })
    ));
    assert_eq!(
        startup_bounds::classify_timestamp_delta(10, 15, Some(4), 1),
        None
    );
    assert_eq!(
        startup_bounds::classify_timestamp_delta(10, 30, None, 1),
        None
    );
    assert!(matches!(
        startup_bounds::classify_timestamp_delta(
            1_000_000,
            1_008_000,
            Some(4_000),
            DEFAULT_TIMESTAMP_TOLERANCE_MICROS,
        ),
        Some(LslInletEvidenceKind::TimestampGap { .. })
    ));
    assert!(matches!(
        startup_bounds::classify_timestamp_delta(
            1_000_000,
            1_002_000,
            Some(1_000),
            DEFAULT_TIMESTAMP_TOLERANCE_MICROS,
        ),
        Some(LslInletEvidenceKind::TimestampGap { .. })
    ));
}

#[test]
fn lsl_i32_projection_fails_closed() {
    let valid = WindowBatch::new(0, vec![vec![i32::MIN as i64, i32::MAX as i64]], 250.0);
    assert!(startup_bounds::to_sample_major_i32(&valid).is_ok());
    let invalid = WindowBatch::new(0, vec![vec![i32::MAX as i64 + 1]], 250.0);
    assert!(startup_bounds::to_sample_major_i32(&invalid).is_err());
}

#[test]
fn resource_bounds_reject_multiplication_overflow_and_excess() {
    assert!(startup_bounds::validate_buffered_values(32, 250, 64, "inlet").is_ok());
    assert!(startup_bounds::validate_buffered_values(usize::MAX, 2, 2, "inlet").is_err());
    assert!(startup_bounds::validate_buffered_values(1_024, 1_024, 64, "outlet").is_err());
    assert!(startup_bounds::validate_buffered_values(MAX_CHANNELS + 1, 1, 1, "outlet").is_err());
    assert!(
        startup_bounds::validate_buffered_values(1, MAX_WINDOW_SAMPLES + 1, 1, "inlet").is_err()
    );
}

#[test]
fn external_reasons_are_utf8_safely_bounded() {
    let reason = "é".repeat(MAX_REASON_BYTES);
    let bounded = startup_bounds::bounded_reason(reason);
    assert!(bounded.len() <= MAX_REASON_BYTES);
    assert!(std::str::from_utf8(bounded.as_bytes()).is_ok());
}

#[tokio::test]
async fn bounded_outlet_overload_returns_failure_receipt_without_advancing_sequence() {
    let mut sink = LslOutletSink::try_new("test-outlet".into(), "test-source".into()).unwrap();
    let (sender, _receiver) = std::sync::mpsc::sync_channel(1);
    sender
        .try_send(OutletCommand {
            idempotency_key: "occupy-queue".into(),
            samples: vec![vec![0]],
        })
        .unwrap();
    sink.tx = Some(sender);
    sink.layout = Some((1, 250_000));

    let mut batch = WindowBatch::new(7, vec![vec![1]], 250.0);
    batch.first_ts_micros = Some(42);
    assert!(sink.consume(&batch).await.is_err());
    assert_eq!(sink.last_enqueued_seq, None);
    let receipts = sink.drain_receipts();
    assert_eq!(receipts.len(), 2);
    let gap = receipts
        .iter()
        .find(|receipt| matches!(receipt.state, LslOutletReceiptState::Gap { .. }))
        .expect("gap receipt");
    assert_eq!(gap.window_seq, 0);
    let failed = receipts
        .iter()
        .find(|receipt| matches!(receipt.state, LslOutletReceiptState::Failed { .. }))
        .expect("failed window receipt");
    assert_eq!(failed.input_first_ts_micros_not_transmitted, Some(42));
    assert_eq!(
        failed.timestamp_policy,
        LslOutletTimestampPolicy::LiblslLocalClockAtPush
    );
    assert_eq!(
        failed.input_to_outlet_clock.receiver_clock,
        failed.outlet_clock
    );
}

#[tokio::test]
async fn outlet_close_timeout_is_bounded_and_retryable() {
    let mut sink = LslOutletSink::try_new("test-outlet".into(), "test-source".into()).unwrap();
    sink.config.close_timeout_millis = 5;
    sink.worker = Some(std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_millis(30));
    }));
    assert!(sink.close().await.is_err());
    tokio::time::sleep(std::time::Duration::from_millis(40)).await;
    assert!(sink.close().await.is_ok());
}
