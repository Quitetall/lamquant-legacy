//! ADR 0135 Phase 1 gate (feature `codec`) — a real biosignal stream flows
//! through the Engine byte-faithfully: replay an `.lml`'s windows into the
//! engine, compress them to a new `.lml`, and prove the decoded signal is
//! sample-for-sample identical. (The live LSL source is feature `lsl` + needs
//! the liblsl system library; it lands on top of this proven path.)
#![cfg(feature = "codec")]

use std::path::{Path, PathBuf};

use lamquant_runtime::codec::{LmlFileSink, LmlReplaySource};
use lamquant_runtime::engine::{Engine, EngineConfig};
use lamquant_runtime::manifest::RuntimeManifest;
use lamquant_runtime::mem::MemSource;
use lamquant_runtime::sink::Sink;
use lamquant_runtime::source::Source;
use lamquant_runtime::status::now_ms;
use tokio_util::sync::CancellationToken;

fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("lqrt-{}-{}.lml", name, now_ms()))
}

/// Drain a source completely into one channel-major signal.
async fn drain(mut src: Box<dyn Source>) -> Vec<Vec<i64>> {
    let mut signal: Vec<Vec<i64>> = Vec::new();
    while let Some(w) = src.next_window().await.unwrap() {
        if signal.is_empty() {
            signal = vec![Vec::new(); w.n_channels()];
        }
        for (c, ch) in w.channels.iter().enumerate() {
            signal[c].extend_from_slice(ch);
        }
    }
    signal
}

/// Write a fixture `.lml` by running a MemSource ramp through an LmlFileSink.
async fn make_fixture(path: &Path, n_ch: usize, n_samp: usize, n_win: u64) {
    let engine = Engine::new(EngineConfig::default());
    let report = engine
        .run_pipeline(
            "fixture",
            Box::new(MemSource::ramp("ramp".into(), n_ch, n_samp, n_win, 250.0)),
            vec![Box::new(LmlFileSink::new(path, 0, 0))],
            CancellationToken::new(),
        )
        .await;
    assert_eq!(report.windows_in, n_win);
    assert!(path.exists(), "fixture .lml not written");
}

#[tokio::test]
async fn lml_sink_rejects_stream_layout_changes_without_panicking() {
    let path = tmp("layout-change");
    let mut sink = LmlFileSink::new(&path, 0, 0);
    sink.consume(&lamquant_runtime::WindowBatch::new(
        0,
        vec![vec![1, 2]],
        250.0,
    ))
    .await
    .unwrap();

    for invalid in [
        lamquant_runtime::WindowBatch::new(1, vec![vec![3, 4], vec![5, 6]], 250.0),
        lamquant_runtime::WindowBatch::new(2, vec![vec![3]], 250.0),
        lamquant_runtime::WindowBatch::new(3, vec![vec![3, 4]], 251.0),
    ] {
        let error = sink
            .consume(&invalid)
            .await
            .expect_err("layout change must fail closed");
        assert!(error.to_string().contains("stream layout changed"));
    }

    sink.close().await.unwrap();
    let signal = drain(Box::new(LmlReplaySource::open(&path, 2).unwrap())).await;
    assert_eq!(signal, vec![vec![1, 2]]);
    let _ = std::fs::remove_file(&path);

    let rotating_path = tmp("layout-change-rotation");
    let mut rotating = LmlFileSink::new(&rotating_path, 1, 0);
    rotating
        .consume(&lamquant_runtime::WindowBatch::new(
            0,
            vec![vec![1, 2]],
            250.0,
        ))
        .await
        .unwrap();
    let error = rotating
        .consume(&lamquant_runtime::WindowBatch::new(
            1,
            vec![vec![3, 4], vec![5, 6]],
            250.0,
        ))
        .await
        .expect_err("layout change after rotation must fail closed");
    assert!(error.to_string().contains("stream layout changed"));

    let stem = rotating_path.file_stem().unwrap().to_str().unwrap();
    let first_rotated = rotating_path.with_file_name(format!("{stem}.0000.lml"));
    let _ = std::fs::remove_file(first_rotated);
}

#[tokio::test]
async fn replay_source_to_lml_file_is_byte_faithful() {
    let fixture = tmp("fixture");
    let out = tmp("out");
    make_fixture(&fixture, 21, 250, 4).await;

    // The slice: LmlReplaySource → Engine → LmlFileSink.
    let engine = Engine::new(EngineConfig::default());
    let report = engine
        .run_pipeline(
            "eeg-desk",
            Box::new(LmlReplaySource::open(&fixture, 250).unwrap()),
            vec![Box::new(LmlFileSink::new(out.clone(), 0, 0))],
            CancellationToken::new(),
        )
        .await;
    assert_eq!(report.windows_in, 4);
    assert!(report.source_error.is_none());
    assert_eq!(report.sinks[0].consumed, 4);
    assert_eq!(report.sinks[0].dropped, 0);
    assert!(!report.sinks[0].errored);
    assert!(out.exists());

    // Decoding both files yields identical samples — lossless through the runtime.
    let a = drain(Box::new(LmlReplaySource::open(&fixture, 250).unwrap())).await;
    let b = drain(Box::new(LmlReplaySource::open(&out, 250).unwrap())).await;
    assert_eq!(a, b, "runtime slice was not byte-faithful");
    assert_eq!(a.len(), 21);
    assert_eq!(a[0].len(), 1000); // 4 windows × 250 samples

    let _ = std::fs::remove_file(&fixture);
    let _ = std::fs::remove_file(&out);
}

#[tokio::test]
async fn rotating_lml_file_writes_multiple_containers() {
    let fixture = tmp("rot-fixture");
    make_fixture(&fixture, 4, 50, 6).await;
    let base = tmp("rot-out");

    let engine = Engine::new(EngineConfig::default());
    let report = engine
        .run_pipeline(
            "rot",
            Box::new(LmlReplaySource::open(&fixture, 50).unwrap()),
            vec![Box::new(LmlFileSink::new(base.clone(), 2, 0))], // rotate every 2 windows
            CancellationToken::new(),
        )
        .await;
    assert_eq!(report.windows_in, 6);
    // 6 windows / 2 = 3 rotated files: base.0000.lml, base.0001.lml, base.0002.lml.
    let stem = base.file_stem().unwrap().to_str().unwrap().to_string();
    for i in 0..3 {
        let p = base.with_file_name(format!("{stem}.{:04}.lml", i));
        assert!(p.exists(), "rotated file {i} missing");
        let _ = std::fs::remove_file(&p);
    }
    let _ = std::fs::remove_file(&fixture);
}

#[tokio::test]
async fn manifest_drives_the_codec_slice() {
    let fixture = tmp("man-fixture");
    make_fixture(&fixture, 8, 100, 3).await;
    let out = tmp("man-out");

    let toml = format!(
        r#"
version = 1
[[pipelines]]
name = "eeg"
source = {{ kind = "lml-replay", path = "{}", window_samples = 100 }}
sinks = [ {{ kind = "lml-file", path = "{}", rotate_windows = 0 }} ]
"#,
        fixture.display(),
        out.display()
    );
    let manifest = RuntimeManifest::from_toml(&toml).unwrap();
    let engine = Engine::new(EngineConfig::default());
    let reports = engine
        .run_manifest(&manifest, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].windows_in, 3);
    assert!(out.exists());

    let a = drain(Box::new(LmlReplaySource::open(&fixture, 100).unwrap())).await;
    let b = drain(Box::new(LmlReplaySource::open(&out, 100).unwrap())).await;
    assert_eq!(a, b);

    let _ = std::fs::remove_file(&fixture);
    let _ = std::fs::remove_file(&out);
}
