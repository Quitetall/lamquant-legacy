//! ADR 0135 Phase 1 — live LSL loopback (feature `lsl`, needs the liblsl runtime).
//! `#[ignore]`d because it does a real network round-trip through LSL; run with:
//!   cargo test -p lamquant-runtime-legacy --features lsl --test lsl_loopback -- --ignored
#![cfg(feature = "lsl")]

use std::time::Duration;

use lamquant_runtime::lsl::{
    LslInletSource, LslOutletEffect, LslOutletReceiptState, LslOutletSink,
};
use lamquant_runtime::sink::Sink;
use lamquant_runtime::source::Source;
use lamquant_runtime::status::now_ms;
use lamquant_runtime::window::WindowBatch;

fn ramp(seq: u64, n_ch: usize, n_samp: usize) -> WindowBatch {
    let channels: Vec<Vec<i64>> = (0..n_ch)
        .map(|c| {
            (0..n_samp)
                .map(|t| (seq as i64) * 1000 + (c as i64) * 100 + t as i64)
                .collect()
        })
        .collect();
    WindowBatch::new(seq, channels, 250.0)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 3)]
#[ignore]
async fn lsl_loopback_publishes_and_receives() {
    let name = format!("lqrt-loopback-{}", now_ms());

    // Feeder: publish paced windows to an LSL outlet (created on first consume).
    let feed_name = name.clone();
    let feeder = tokio::spawn(async move {
        let mut sink =
            LslOutletSink::try_new(feed_name, "lqrt-test".into()).expect("valid bounded LSL sink");
        assert_eq!(
            sink.effect(),
            LslOutletEffect::AtMostOnceEnqueuedProcessLocal
        );
        for seq in 0..20u64 {
            if sink.consume(&ramp(seq, 4, 50)).await.is_err() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        sink.close().await.expect("bounded outlet close");
        sink.drain_receipts()
    });

    // Give the outlet a moment to come up, then resolve + pull.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let mut src = tokio::task::spawn_blocking({
        let name = name.clone();
        move || LslInletSource::open(name, 50, 5.0)
    })
    .await
    .unwrap()
    .expect("resolve the loopback stream");

    assert_eq!(src.info().n_channels, 4);
    assert_eq!(src.clock_relation().offset_micros, None);
    assert_eq!(
        src.clock_relation().publisher_clock.as_str(),
        format!("lsl.publisher-name-unverified:{name}")
    );
    let mut received = 0;
    for _ in 0..3 {
        match tokio::time::timeout(Duration::from_secs(5), src.next_window()).await {
            Ok(Ok(Some(w))) => {
                assert_eq!(w.n_channels(), 4);
                assert_eq!(w.n_samples(), 50);
                assert!(w.first_ts_micros.is_some());
                received += 1;
            }
            other => panic!("expected a window, got {other:?}"),
        }
    }
    assert!(received >= 1, "received no windows over LSL");

    let receipts = feeder.await.expect("feeder task");
    assert!(!receipts.is_empty());
    assert!(receipts
        .iter()
        .all(|receipt| matches!(receipt.state, LslOutletReceiptState::Attempted { .. })));
}
