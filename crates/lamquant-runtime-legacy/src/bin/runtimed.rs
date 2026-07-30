//! `lamquant-runtimed` (feature `daemon`) — the standalone runtime daemon + its
//! control client. `start` runs a manifest headless (persist + socket); `status`
//! / `stop` / `ping` talk to a running daemon over its Unix socket (ADR 0135).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use clap::{Parser, Subcommand};
use lamquant_runtime::control::{read_frame, write_frame, ControlRequest, ControlResponse};
use lamquant_runtime::daemon::{self, DaemonConfig};
use lamquant_runtime::manifest::RuntimeManifest;
use lamquant_runtime::status::RuntimeStatus;
use tokio::net::UnixStream;
use tokio_util::sync::CancellationToken;

#[derive(Parser)]
#[command(
    name = "lamquant-runtimed",
    about = "LamQuant biosignal ingest→sink daemon (ADR 0135)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run a manifest headless: ingest → sinks, persist status.jsonl, serve the socket.
    Start {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long, default_value = ".lamquant-runtime")]
        state_dir: PathBuf,
    },
    /// Print the running daemon's live pipeline status.
    Status {
        #[arg(long, default_value = ".lamquant-runtime")]
        state_dir: PathBuf,
    },
    /// Live console: refresh the daemon's status until Ctrl-C (prefers the
    /// socket; falls back to tailing status.jsonl).
    Watch {
        #[arg(long, default_value = ".lamquant-runtime")]
        state_dir: PathBuf,
        #[arg(long, default_value = "500")]
        interval_ms: u64,
    },
    /// Ask the running daemon to drain + shut down.
    Stop {
        #[arg(long, default_value = ".lamquant-runtime")]
        state_dir: PathBuf,
    },
    /// Liveness + version handshake.
    Ping {
        #[arg(long, default_value = ".lamquant-runtime")]
        state_dir: PathBuf,
    },
}

async fn request(socket: &PathBuf, req: ControlRequest) -> std::io::Result<ControlResponse> {
    let mut stream = UnixStream::connect(socket).await?;
    write_frame(&mut stream, &req).await?;
    read_frame(&mut stream).await
}

async fn wait_for_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut term = signal(SignalKind::terminate()).expect("SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {},
            _ = term.recv() => {},
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber_init();
    match Cli::parse().cmd {
        Cmd::Start {
            manifest,
            state_dir,
        } => {
            let text = match std::fs::read_to_string(&manifest) {
                Ok(t) => t,
                Err(e) => return fail(format!("read {}: {e}", manifest.display())),
            };
            let m = match RuntimeManifest::from_toml(&text) {
                Ok(m) => m,
                Err(e) => return fail(format!("manifest: {e}")),
            };
            let cfg = DaemonConfig::under(&state_dir);
            println!(
                "lamquant-runtimed: {} pipeline(s) → socket {}",
                m.pipelines.len(),
                cfg.socket_path.display()
            );
            let shutdown = CancellationToken::new();
            let sig = shutdown.clone();
            tokio::spawn(async move {
                wait_for_signal().await;
                eprintln!("lamquant-runtimed: signal → draining");
                sig.cancel();
            });
            match daemon::run(m, cfg, shutdown).await {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => fail(format!("daemon: {e}")),
            }
        }
        Cmd::Status { state_dir } => {
            let cfg = DaemonConfig::under(&state_dir);
            match request(&cfg.socket_path, ControlRequest::Status).await {
                Ok(ControlResponse::Status { pipelines }) => {
                    print_pipelines(&pipelines);
                    ExitCode::SUCCESS
                }
                Ok(other) => fail(format!("unexpected reply: {other:?}")),
                // Socket down → the hybrid fallback: read the persisted file.
                Err(e) => match read_status_file(&cfg.status_path) {
                    Some(pipelines) => {
                        eprintln!("(socket down: {e}) — showing persisted status.jsonl");
                        print_pipelines(&pipelines);
                        ExitCode::SUCCESS
                    }
                    None => fail(format!(
                        "no daemon at {} and no status.jsonl",
                        cfg.socket_path.display()
                    )),
                },
            }
        }
        Cmd::Watch {
            state_dir,
            interval_ms,
        } => {
            let cfg = DaemonConfig::under(&state_dir);
            let mut ticker = tokio::time::interval(Duration::from_millis(interval_ms.max(50)));
            loop {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {
                        println!();
                        break;
                    }
                    _ = ticker.tick() => {}
                }
                print!("\x1b[2J\x1b[H"); // clear + home
                println!(
                    "lamquant runtime — {} (Ctrl-C to exit)\n",
                    cfg.socket_path.display()
                );
                match request(&cfg.socket_path, ControlRequest::Status).await {
                    Ok(ControlResponse::Status { pipelines }) => print_pipelines(&pipelines),
                    Ok(other) => println!("unexpected reply: {other:?}"),
                    Err(_) => match read_status_file(&cfg.status_path) {
                        Some(pipelines) => {
                            println!("(socket down — persisted status.jsonl)\n");
                            print_pipelines(&pipelines);
                        }
                        None => {
                            println!("(no daemon; run `lamquant runtime start --manifest ...`)")
                        }
                    },
                }
            }
            ExitCode::SUCCESS
        }
        Cmd::Stop { state_dir } => {
            let socket = DaemonConfig::under(&state_dir).socket_path;
            match request(&socket, ControlRequest::Stop).await {
                Ok(ControlResponse::Stopping) => {
                    println!("stopping");
                    ExitCode::SUCCESS
                }
                Ok(other) => fail(format!("unexpected reply: {other:?}")),
                Err(e) => fail(format!("connect {}: {e}", socket.display())),
            }
        }
        Cmd::Ping { state_dir } => {
            let socket = DaemonConfig::under(&state_dir).socket_path;
            match request(&socket, ControlRequest::Ping).await {
                Ok(ControlResponse::Pong { version }) => {
                    println!("pong (protocol v{version})");
                    ExitCode::SUCCESS
                }
                Ok(other) => fail(format!("unexpected reply: {other:?}")),
                Err(e) => fail(format!("connect {}: {e}", socket.display())),
            }
        }
    }
}

fn fail(msg: String) -> ExitCode {
    eprintln!("lamquant-runtimed: {msg}");
    ExitCode::FAILURE
}

fn print_pipelines(pipelines: &[RuntimeStatus]) {
    if pipelines.is_empty() {
        println!("(no pipelines)");
        return;
    }
    for p in pipelines {
        println!(
            "[{}] src={} in={}{}",
            p.pipeline,
            p.source.info.label,
            p.source.windows_in,
            if p.source.errored {
                " (source errored)"
            } else {
                ""
            }
        );
        for s in &p.sinks {
            println!(
                "    {} {}: consumed={} dropped={}",
                s.info.kind, s.info.label, s.windows_consumed, s.windows_dropped
            );
        }
    }
}

/// Read `status.jsonl` and keep the LAST record per pipeline — the hybrid
/// fallback used when the daemon's control socket is unavailable.
fn read_status_file(path: &Path) -> Option<Vec<RuntimeStatus>> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut latest: BTreeMap<String, RuntimeStatus> = BTreeMap::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(s) = serde_json::from_str::<RuntimeStatus>(line) {
            latest.insert(s.pipeline.clone(), s);
        }
    }
    if latest.is_empty() {
        None
    } else {
        Some(latest.into_values().collect())
    }
}

fn tracing_subscriber_init() {
    // Best-effort: the crate does not depend on tracing-subscriber, so this is a
    // no-op placeholder. Daemon logs go through `tracing`; a host can install a
    // subscriber. Kept as a hook so `main` reads cleanly.
}
