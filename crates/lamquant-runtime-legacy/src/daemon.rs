//! The `lamquant-runtimed` engine (feature `daemon`) — hosts a manifest's
//! pipelines on one runtime, **persists** append-only `status.jsonl`, **and**
//! serves a Unix-socket control channel (hybrid control, ADR 0135). Graceful
//! shutdown on SIGINT/SIGTERM, a `Stop` control request, or all pipelines ending.

use std::os::unix::fs::{FileTypeExt, MetadataExt};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use tokio::net::UnixListener;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::control::{read_frame, write_frame, ControlRequest, ControlResponse, PROTOCOL_VERSION};
use crate::engine::{prepare_manifest, Engine, EngineConfig, LiveStats};
use crate::manifest::RuntimeManifest;
use crate::sink::SinkInfo;
use crate::source::SourceInfo;
use crate::status::{now_ms, RuntimeStatus, SinkStat, SourceStat, StatusWriter};

/// Where the daemon writes its socket + status, and how often it snapshots.
#[derive(Debug, Clone)]
pub struct DaemonConfig {
    pub socket_path: PathBuf,
    pub status_path: PathBuf,
    pub status_interval: Duration,
}

impl DaemonConfig {
    /// Default layout under `state_dir`: `control.sock` + `status.jsonl`, 500 ms.
    pub fn under(state_dir: impl Into<PathBuf>) -> Self {
        let dir = state_dir.into();
        Self {
            socket_path: dir.join("control.sock"),
            status_path: dir.join("status.jsonl"),
            status_interval: Duration::from_millis(500),
        }
    }
}

/// Per-pipeline handle the status loop reads live counters from.
struct PipelineHandle {
    name: String,
    source: SourceInfo,
    sinks: Vec<SinkInfo>,
    stats: Arc<LiveStats>,
}

struct SocketPathGuard {
    path: PathBuf,
    device: u64,
    inode: u64,
}

impl Drop for SocketPathGuard {
    fn drop(&mut self) {
        let Ok(metadata) = std::fs::symlink_metadata(&self.path) else {
            return;
        };
        if metadata.file_type().is_socket()
            && metadata.dev() == self.device
            && metadata.ino() == self.inode
        {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

fn bind_control_socket(
    socket_path: &std::path::Path,
) -> crate::error::Result<(UnixListener, SocketPathGuard)> {
    if let Some(dir) = socket_path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)?;
        }
    }
    match std::fs::symlink_metadata(socket_path) {
        Ok(_) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!(
                    "refusing to replace existing control-socket path {}",
                    socket_path.display()
                ),
            )
            .into());
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }

    let parent = socket_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    const STAGING_NAMES: &[u8] =
        b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_-";
    let mut bound = None;
    for name in STAGING_NAMES {
        let staging_path = parent.join(char::from(*name).to_string());
        if staging_path == socket_path {
            continue;
        }
        match UnixListener::bind(&staging_path) {
            Ok(listener) => {
                bound = Some((listener, staging_path));
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AddrInUse => continue,
            Err(error) => return Err(error.into()),
        }
    }
    let (listener, staging_path) = bound.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "cannot allocate a one-byte control-socket staging path",
        )
    })?;
    let staging_metadata = std::fs::symlink_metadata(&staging_path)?;
    if !staging_metadata.file_type().is_socket() {
        return Err(std::io::Error::other(format!(
            "staging control path {} is not a Unix socket after bind",
            staging_path.display()
        ))
        .into());
    }
    let staging_guard = SocketPathGuard {
        path: staging_path,
        device: staging_metadata.dev(),
        inode: staging_metadata.ino(),
    };
    std::fs::hard_link(&staging_guard.path, socket_path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::AlreadyExists {
            std::io::Error::new(
                error.kind(),
                format!(
                    "refusing to replace existing control-socket path {}",
                    socket_path.display()
                ),
            )
        } else {
            error
        }
    })?;
    let published_metadata = std::fs::symlink_metadata(socket_path)?;
    if !published_metadata.file_type().is_socket()
        || published_metadata.dev() != staging_guard.device
        || published_metadata.ino() != staging_guard.inode
    {
        return Err(std::io::Error::other(format!(
            "control path {} changed during publication",
            socket_path.display()
        ))
        .into());
    }
    let published_guard = SocketPathGuard {
        path: socket_path.to_path_buf(),
        device: published_metadata.dev(),
        inode: published_metadata.ino(),
    };
    drop(staging_guard);
    Ok((listener, published_guard))
}

fn snapshot(handles: &[PipelineHandle]) -> Vec<RuntimeStatus> {
    let now = now_ms();
    handles
        .iter()
        .map(|h| RuntimeStatus {
            updated_ms: now,
            pipeline: h.name.clone(),
            source: SourceStat {
                info: h.source.clone(),
                windows_in: h.stats.windows_in.load(Ordering::Relaxed),
                errored: h.stats.source_error.load(Ordering::Relaxed),
            },
            sinks: h
                .sinks
                .iter()
                .enumerate()
                .map(|(i, info)| SinkStat {
                    info: info.clone(),
                    windows_consumed: h.stats.sinks[i].consumed.load(Ordering::Relaxed),
                    windows_dropped: h.stats.sinks[i].dropped.load(Ordering::Relaxed),
                    errored: false,
                })
                .collect(),
        })
        .collect()
}

/// Run the daemon until all pipelines end, `shutdown` fires, or a `Stop` arrives.
pub async fn run(
    manifest: RuntimeManifest,
    cfg: DaemonConfig,
    shutdown: CancellationToken,
) -> crate::error::Result<()> {
    let engine_cfg = EngineConfig::default();
    let prepared = prepare_manifest(&manifest)?;
    let (control_listener, socket_guard) = bind_control_socket(&cfg.socket_path)?;
    info!(path = %cfg.socket_path.display(), "control: listening");

    let mut handles = Vec::with_capacity(prepared.len());
    let mut joins = Vec::with_capacity(prepared.len());
    for pipeline in prepared {
        let source_info = pipeline.source.info();
        let sink_infos: Vec<SinkInfo> = pipeline.sinks.iter().map(|sink| sink.info()).collect();
        let stats = LiveStats::new(pipeline.sinks.len());
        handles.push(PipelineHandle {
            name: pipeline.name.clone(),
            source: source_info,
            sinks: sink_infos,
            stats: Arc::clone(&stats),
        });
        let sd = shutdown.clone();
        joins.push(tokio::spawn(async move {
            Engine::new(engine_cfg)
                .run_pipeline_tracked(pipeline.name, pipeline.source, pipeline.sinks, sd, stats)
                .await
        }));
    }
    let handles = Arc::new(handles);

    // Status loop: periodically PERSIST live counters to status.jsonl (the
    // crash-safe / headless-readable half of hybrid control).
    let status_task = tokio::spawn(status_loop(
        Arc::clone(&handles),
        cfg.status_path.clone(),
        cfg.status_interval,
        shutdown.clone(),
    ));

    // Control server: serve LIVE status straight from the counters (the socket
    // half — always current, unlike the periodic file).
    let control_task = tokio::spawn(control_server(
        control_listener,
        Arc::clone(&handles),
        shutdown.clone(),
    ));

    // All pipelines finished (source EOF or shutdown) → tear everything down.
    for j in joins {
        let _ = j.await;
    }
    shutdown.cancel();
    let _ = status_task.await;
    control_task.abort();
    let _ = control_task.await;
    drop(socket_guard);
    info!("daemon: all pipelines ended; shut down");
    Ok(())
}

async fn status_loop(
    handles: Arc<Vec<PipelineHandle>>,
    status_path: PathBuf,
    interval: Duration,
    shutdown: CancellationToken,
) {
    let mut writer = StatusWriter::open(&status_path).ok();
    let mut ticker = tokio::time::interval(interval);
    loop {
        let done = tokio::select! {
            _ = shutdown.cancelled() => true,
            _ = ticker.tick() => false,
        };
        if let Some(w) = writer.as_mut() {
            for s in &snapshot(&handles) {
                if let Err(e) = w.record(s) {
                    warn!(error = %e, "status: record failed");
                }
            }
        }
        if done {
            break;
        }
    }
}

async fn control_server(
    listener: UnixListener,
    handles: Arc<Vec<PipelineHandle>>,
    shutdown: CancellationToken,
) {
    loop {
        tokio::select! {
            _ = shutdown.cancelled() => break,
            accepted = listener.accept() => match accepted {
                Ok((stream, _)) => {
                    // SO_PEERCRED gate (feature `auth`): only our own uid may
                    // control the daemon. When `auth` is off this is absent and
                    // behavior is byte-identical to ADR 0135.
                    #[cfg(feature = "auth")]
                    {
                        match stream.peer_cred() {
                            Ok(cred) if crate::auth::authorize_peer_uid(cred.uid()).is_ok() => {}
                            Ok(cred) => {
                                warn!(peer_uid = cred.uid(), "control: refused foreign uid");
                                continue;
                            }
                            Err(e) => {
                                warn!(error = %e, "control: cannot read peer cred; refusing");
                                continue;
                            }
                        }
                    }
                    let handles = Arc::clone(&handles);
                    let sd = shutdown.clone();
                    tokio::spawn(async move { handle_conn(stream, handles, sd).await; });
                }
                Err(e) => warn!(error = %e, "control: accept failed"),
            },
        }
    }
}

async fn handle_conn(
    mut stream: tokio::net::UnixStream,
    handles: Arc<Vec<PipelineHandle>>,
    shutdown: CancellationToken,
) {
    // Serve requests until the client closes the connection.
    loop {
        let req: ControlRequest = match read_frame(&mut stream).await {
            Ok(r) => r,
            Err(_) => break, // EOF / closed
        };
        let resp = match req {
            ControlRequest::Ping => ControlResponse::Pong {
                version: PROTOCOL_VERSION,
            },
            ControlRequest::Status => ControlResponse::Status {
                pipelines: snapshot(&handles),
            },
            ControlRequest::Stop => {
                shutdown.cancel();
                ControlResponse::Stopping
            }
        };
        if write_frame(&mut stream, &resp).await.is_err() {
            break;
        }
    }
}
