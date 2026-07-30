//! ADR 0135 Phase 0 gate — the trait surface + engine + manifest + status hold.

use std::sync::atomic::Ordering;

use lamquant_runtime::engine::{Engine, EngineConfig};
use lamquant_runtime::manifest::{RuntimeManifest, MANIFEST_VERSION};
use lamquant_runtime::mem::{MemSink, MemSource};
use lamquant_runtime::sink::Sink;
use lamquant_runtime::source::Source;
use lamquant_runtime::status::{now_ms, RuntimeStatus, SinkStat, SourceStat, StatusWriter};
use lamquant_runtime::{RuntimeError, WindowBatch};
use tokio_util::sync::CancellationToken;

const VALID: &str = r#"
version = 1
[[pipelines]]
name = "eeg-desk"
source = { kind = "mem", label = "synthetic", n_channels = 21, n_samples = 250, n_windows = 4, sample_rate_hz = 250.0 }
sinks = [
  { kind = "mem", label = "counter-a" },
  { kind = "mem", label = "counter-b" },
]
"#;

#[test]
fn manifest_parses_valid() {
    let m = RuntimeManifest::from_toml(VALID).expect("valid manifest");
    assert_eq!(m.version, MANIFEST_VERSION);
    assert_eq!(m.pipelines.len(), 1);
    assert_eq!(m.pipelines[0].sinks.len(), 2);
}

#[test]
fn manifest_fails_closed_on_bad_version() {
    let bad = VALID.replace("version = 1", "version = 99");
    match RuntimeManifest::from_toml(&bad) {
        Err(RuntimeError::ManifestVersion {
            found: 99,
            expected,
        }) => {
            assert_eq!(expected, MANIFEST_VERSION)
        }
        other => panic!("expected ManifestVersion, got {other:?}"),
    }
}

#[test]
fn manifest_fails_closed_on_empty_and_sinkless() {
    assert!(matches!(
        RuntimeManifest::from_toml("version = 1"),
        Err(RuntimeError::EmptyManifest)
    ));
    let sinkless = r#"
version = 1
[[pipelines]]
name = "x"
source = { kind = "mem", label = "s", n_channels = 1, n_samples = 1, n_windows = 1, sample_rate_hz = 1.0 }
sinks = []
"#;
    assert!(matches!(
        RuntimeManifest::from_toml(sinkless),
        Err(RuntimeError::NoSinks { .. })
    ));
}

#[test]
fn manifest_rejects_unknown_kind() {
    let bad = VALID.replace(
        r#"kind = "mem", label = "counter-a""#,
        r#"kind = "not-a-sink""#,
    );
    assert!(matches!(
        RuntimeManifest::from_toml(&bad),
        Err(RuntimeError::ManifestParse(_))
    ));
}

#[cfg(not(feature = "lsl"))]
#[test]
fn phase1_sources_fail_closed_until_compiled_in() {
    use lamquant_runtime::manifest::{build_source, SourceSpec};
    let spec = SourceSpec::Lsl {
        stream_name: "EEG".into(),
        window_samples: 250,
        resolve_timeout_s: 1.0,
    };
    assert!(matches!(
        build_source(&spec),
        Err(RuntimeError::UnknownSource { .. })
    ));
}

#[tokio::test]
async fn engine_fans_one_source_to_many_sinks() {
    let src = MemSource::ramp("synthetic".into(), 21, 250, 4, 250.0);
    let sink_a = MemSink::new("a".into());
    let sink_b = MemSink::new("b".into());
    let (ca, fa) = (sink_a.counter(), sink_a.close_counter());
    let (cb, fb) = (sink_b.counter(), sink_b.close_counter());

    let engine = Engine::new(EngineConfig::default());
    let report = engine
        .run_pipeline(
            "eeg-desk",
            Box::new(src),
            vec![Box::new(sink_a), Box::new(sink_b)],
            CancellationToken::new(),
        )
        .await;

    assert_eq!(report.windows_in, 4);
    assert!(report.source_error.is_none());
    assert_eq!(report.sinks.len(), 2);
    for outcome in &report.sinks {
        assert_eq!(outcome.consumed, 4);
        assert_eq!(outcome.dropped, 0);
        assert!(!outcome.errored);
    }
    // Both sinks saw every window and were cleanly finalized.
    assert_eq!(ca.load(Ordering::SeqCst), 4);
    assert_eq!(cb.load(Ordering::SeqCst), 4);
    assert_eq!(fa.load(Ordering::SeqCst), 1);
    assert_eq!(fb.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn engine_drains_gracefully_on_shutdown() {
    // A huge source + a pre-cancelled token → the source never pulls, the sinks
    // still close cleanly (graceful drain, not a hard stop).
    let src = MemSource::ramp("synthetic".into(), 4, 10, 1_000_000, 250.0);
    let sink = MemSink::new("a".into());
    let closed = sink.close_counter();
    let token = CancellationToken::new();
    token.cancel();

    let engine = Engine::new(EngineConfig::default());
    let report = engine
        .run_pipeline("x", Box::new(src), vec![Box::new(sink)], token)
        .await;

    assert_eq!(report.windows_in, 0);
    assert_eq!(closed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn engine_runs_a_mem_manifest() {
    let m = RuntimeManifest::from_toml(VALID).unwrap();
    let engine = Engine::new(EngineConfig::default());
    let reports = engine
        .run_manifest(&m, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].windows_in, 4);
    assert!(reports[0].sinks.iter().all(|s| s.consumed == 4));
}

#[test]
fn status_round_trips_and_detects_staleness() {
    let status = RuntimeStatus {
        updated_ms: now_ms(),
        pipeline: "eeg-desk".into(),
        source: SourceStat {
            info: MemSource::ramp("s".into(), 21, 250, 1, 250.0).info(),
            windows_in: 10,
            errored: false,
        },
        sinks: vec![SinkStat {
            info: MemSink::new("a".into()).info(),
            windows_consumed: 10,
            windows_dropped: 0,
            errored: false,
        }],
    };
    let json = serde_json::to_string(&status).unwrap();
    let back: RuntimeStatus = serde_json::from_str(&json).unwrap();
    assert_eq!(status, back);
    assert!(!status.is_stale(status.updated_ms + 100, 1000));
    assert!(status.is_stale(status.updated_ms + 5000, 1000));

    // The writer appends a readable JSON line.
    let path = std::env::temp_dir().join(format!("lqrt-status-{}.jsonl", now_ms()));
    let mut w = StatusWriter::open(&path).unwrap();
    w.record(&status).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.trim().ends_with('}'));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn window_batch_shape_helpers() {
    let w = WindowBatch::new(3, vec![vec![1, 2, 3], vec![4, 5, 6]], 250.0);
    assert_eq!(w.n_channels(), 2);
    assert_eq!(w.n_samples(), 3);
    assert!(w.is_rectangular());
    assert_eq!(w.sample_rate_hz(), 250.0);
    let ragged = WindowBatch::new(0, vec![vec![1, 2], vec![3]], 1.0);
    assert!(!ragged.is_rectangular());
}
