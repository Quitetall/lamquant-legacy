//! ADR 0135 Phase 2 gate (feature `daemon`) — the daemon runs a manifest, serves
//! live status + a `status.jsonl`, and stops cleanly over its control socket.
#![cfg(feature = "daemon")]

use std::path::Path;
use std::time::Duration;

use lamquant_runtime::control::{read_frame, write_frame, ControlRequest, ControlResponse};
use lamquant_runtime::daemon::{self, DaemonConfig};
use lamquant_runtime::manifest::RuntimeManifest;
use lamquant_runtime::status::now_ms;
use tokio::net::UnixStream;
use tokio_util::sync::CancellationToken;

async fn request(socket: &Path, req: ControlRequest) -> ControlResponse {
    let mut stream = UnixStream::connect(socket).await.expect("connect socket");
    write_frame(&mut stream, &req).await.expect("write");
    read_frame(&mut stream).await.expect("read")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 3)]
async fn daemon_serves_status_and_stops_cleanly() {
    let dir = std::env::temp_dir().join(format!("lqrt-daemon-{}", now_ms()));
    let cfg = DaemonConfig::under(&dir);
    let manifest = RuntimeManifest::from_toml(
        r#"
version = 1
[[pipelines]]
name = "synthetic"
source = { kind = "mem", label = "s", n_channels = 4, n_samples = 10, n_windows = 60, sample_rate_hz = 250.0, pace_ms = 40 }
sinks = [ { kind = "mem", label = "counter" } ]
"#,
    )
    .unwrap();

    let shutdown = CancellationToken::new();
    let handle = tokio::spawn(daemon::run(manifest, cfg.clone(), shutdown.clone()));

    // Wait for the control socket to come up.
    let socket = cfg.socket_path.clone();
    let mut up = false;
    for _ in 0..60 {
        if UnixStream::connect(&socket).await.is_ok() {
            up = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(up, "daemon control socket never appeared");

    // Handshake.
    assert!(matches!(
        request(&socket, ControlRequest::Ping).await,
        ControlResponse::Pong { .. }
    ));

    // Let some windows flow, then read live status.
    tokio::time::sleep(Duration::from_millis(300)).await;
    match request(&socket, ControlRequest::Status).await {
        ControlResponse::Status { pipelines } => {
            assert_eq!(pipelines.len(), 1);
            assert_eq!(pipelines[0].pipeline, "synthetic");
            assert!(pipelines[0].source.windows_in >= 1, "no windows counted");
            assert_eq!(pipelines[0].sinks.len(), 1);
            assert!(pipelines[0].sinks[0].windows_consumed >= 1);
        }
        other => panic!("expected Status, got {other:?}"),
    }
    assert!(cfg.status_path.exists(), "status.jsonl not written");

    // Stop → the daemon drains + exits, and cleans up its socket.
    assert!(matches!(
        request(&socket, ControlRequest::Stop).await,
        ControlResponse::Stopping
    ));
    let res = tokio::time::timeout(Duration::from_secs(5), handle)
        .await
        .expect("daemon did not stop within 5s")
        .expect("daemon task panicked");
    assert!(res.is_ok(), "daemon returned an error: {res:?}");
    assert!(!socket.exists(), "socket not cleaned up");

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn daemon_refuses_to_replace_regular_control_path() {
    let dir = std::env::temp_dir().join(format!("lqrt-daemon-path-{}", now_ms()));
    std::fs::create_dir_all(&dir).unwrap();
    let cfg = DaemonConfig::under(&dir);
    let sentinel = b"preserve this file";
    std::fs::write(&cfg.socket_path, sentinel).unwrap();
    let manifest = RuntimeManifest::from_toml(
        r#"
version = 1
[[pipelines]]
name = "synthetic"
source = { kind = "mem", label = "s", n_channels = 1, n_samples = 1, n_windows = 1, sample_rate_hz = 1.0 }
sinks = [ { kind = "mem", label = "counter" } ]
"#,
    )
    .unwrap();

    let error = daemon::run(manifest, cfg.clone(), CancellationToken::new())
        .await
        .expect_err("existing regular control path must fail closed");
    assert!(
        error.to_string().contains("refusing to replace"),
        "unexpected error: {error}"
    );
    assert_eq!(std::fs::read(&cfg.socket_path).unwrap(), sentinel);

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn daemon_cleanup_preserves_replacement_path() {
    let dir = std::env::temp_dir().join(format!("lqrt-daemon-cleanup-{}", now_ms()));
    let cfg = DaemonConfig::under(&dir);
    let manifest = RuntimeManifest::from_toml(
        r#"
version = 1
[[pipelines]]
name = "synthetic"
source = { kind = "mem", label = "s", n_channels = 1, n_samples = 1, n_windows = 1000, sample_rate_hz = 1.0, pace_ms = 10 }
sinks = [ { kind = "mem", label = "counter" } ]
"#,
    )
    .unwrap();
    let shutdown = CancellationToken::new();
    let handle = tokio::spawn(daemon::run(manifest, cfg.clone(), shutdown.clone()));

    for _ in 0..100 {
        if cfg.socket_path.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(cfg.socket_path.exists(), "daemon socket never appeared");
    std::fs::remove_file(&cfg.socket_path).unwrap();
    let sentinel = b"replacement survives cleanup";
    std::fs::write(&cfg.socket_path, sentinel).unwrap();

    shutdown.cancel();
    let result = tokio::time::timeout(Duration::from_secs(5), handle)
        .await
        .expect("daemon did not stop")
        .expect("daemon task panicked");
    assert!(result.is_ok());
    assert_eq!(std::fs::read(&cfg.socket_path).unwrap(), sentinel);

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn daemon_accepts_near_limit_socket_path() {
    let dir = std::env::temp_dir().join(format!("lq-{:0>80}", now_ms()));
    let cfg = DaemonConfig::under(&dir);
    assert_eq!(
        cfg.socket_path.as_os_str().as_encoded_bytes().len(),
        101,
        "regression fixture must remain near Linux SUN_LEN"
    );
    let manifest = RuntimeManifest::from_toml(
        r#"
version = 1
[[pipelines]]
name = "synthetic"
source = { kind = "mem", label = "s", n_channels = 1, n_samples = 1, n_windows = 1, sample_rate_hz = 1.0 }
sinks = [ { kind = "mem", label = "counter" } ]
"#,
    )
    .unwrap();

    daemon::run(manifest, cfg, CancellationToken::new())
        .await
        .expect("valid near-limit socket path must bind");

    let _ = std::fs::remove_dir_all(&dir);
}
