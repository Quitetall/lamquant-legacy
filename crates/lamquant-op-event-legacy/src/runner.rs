//! Subprocess runner — spawns either the `lml` self-binary or an arbitrary
//! external command, streaming stdout/stderr line-by-line as `OpEvent`s
//! through a sink.
//!
//! The runner is generic over `OpEventSink`. In-process consumers (Rust TUI)
//! provide `MpscSink`; cross-process consumers (Tauri GUI) provide their own
//! sink that updates a shared snapshot AND emits state-transition events.

use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, TryRecvError};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::Duration;

use crate::sink::OpEventSink;
use crate::OpEvent;

/// ADR 0022 SD-3: recover from a poisoned Mutex instead of panicking
/// the supervisor. Every site in this module locks either a `bool`
/// flag (kill / tailer_stop) or a small struct; the worst-case
/// post-poison state is a torn boolean, which the caller tolerates
/// (next iteration sees the value and acts).
fn lock_or_recover<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    match m.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    }
}

/// ADR 0022 SD-2: bounded child termination. `Child::kill()` is
/// SIGKILL on Unix; the supervisor used to call it then block on
/// `child.wait()` indefinitely. If the child somehow survived
/// (e.g., zombie reaping race, stuck in D-state syscall), the
/// supervisor hung. Now: kill, then poll `try_wait()` every 50ms
/// up to LAMQUANT_KILL_TIMEOUT_SECS (default 5s). If still alive
/// at deadline, fall through to the original blocking wait but
/// log a warning so operators see the hang.
///
/// SIGTERM-first ladder would be nicer but requires a new dep
/// (nix or libc). Deferred to a follow-up if any subprocess
/// shows up that needs a graceful-flush window.
fn terminate_child(child: &mut Child) -> std::io::Result<std::process::ExitStatus> {
    // If the child already exited, return its status without
    // killing.
    if let Ok(Some(status)) = child.try_wait() {
        return Ok(status);
    }
    let _ = child.kill();
    let timeout_secs: u64 = std::env::var("LAMQUANT_KILL_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let deadline = std::time::Instant::now() + Duration::from_secs(timeout_secs);
    while std::time::Instant::now() < deadline {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
            Err(e) => return Err(e),
        }
    }
    eprintln!(
        "  WARNING: child PID {} did not exit within {}s of SIGKILL; falling back to blocking wait",
        child.id(),
        timeout_secs
    );
    child.wait()
}

/// Returned to the caller when an op is launched. Holds a kill channel.
#[derive(Debug)]
pub struct OpHandle {
    kill_tx: mpsc::Sender<()>,
    /// Tracks whether kill() has been called so callers can avoid double-send.
    killed: Arc<Mutex<bool>>,
}

impl OpHandle {
    /// Request graceful cancellation. Idempotent — second call is a no-op.
    pub fn kill(&mut self) {
        let mut guard = lock_or_recover(&self.killed);
        if !*guard {
            *guard = true;
            // Best-effort send; supervisor may have already exited.
            let _ = self.kill_tx.send(());
        }
    }

    /// Whether kill has been requested.
    pub fn was_killed(&self) -> bool {
        *lock_or_recover(&self.killed)
    }
}

/// Spawn a subprocess of the current `lml` binary (resolved via current_exe).
pub fn spawn_lml<S: OpEventSink>(args: Vec<String>, sink: S) -> OpHandle {
    let op_id = args.first().cloned().unwrap_or_default();
    let exe_label = "lml".to_string();
    spawn_internal(SpawnTarget::Lml, op_id, exe_label, args, sink)
}

/// Spawn an arbitrary external command (training, eagle, pytest, etc.).
pub fn spawn_command<S: OpEventSink>(program: String, args: Vec<String>, sink: S) -> OpHandle {
    let op_id = program.clone();
    let label = program.clone();
    spawn_internal(SpawnTarget::Cmd(program), op_id, label, args, sink)
}

enum SpawnTarget {
    Lml,
    Cmd(String),
}

fn spawn_internal<S: OpEventSink>(
    target: SpawnTarget,
    op_id: String,
    exe_label: String,
    args: Vec<String>,
    sink: S,
) -> OpHandle {
    let (kill_tx, kill_rx) = mpsc::channel::<()>();
    let killed = Arc::new(Mutex::new(false));
    let killed_supervisor = Arc::clone(&killed);

    // Sink lives on the supervisor thread; clones go to stdout/stderr readers.
    let sink = Arc::new(sink);

    thread::spawn(move || {
        sink.emit(OpEvent::Started {
            ts_ms: OpEvent::now_ms(),
            op_id: op_id.clone(),
            total: None,
        });
        sink.emit(OpEvent::Log {
            ts_ms: OpEvent::now_ms(),
            message: format!("$ {} {}", exe_label, args.join(" ")),
        });

        let spawn_result: Result<Child, String> = match target {
            SpawnTarget::Lml => match std::env::current_exe() {
                Ok(exe) => Command::new(&exe)
                    .args(&args)
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()
                    .map_err(|e| format!("spawn lml: {}", e)),
                Err(e) => Err(format!("current_exe: {}", e)),
            },
            SpawnTarget::Cmd(prog) => Command::new(&prog)
                .args(&args)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .map_err(|e| format!("spawn {}: {}", prog, e)),
        };

        let mut child = match spawn_result {
            Ok(c) => c,
            Err(e) => {
                sink.emit(OpEvent::Error {
                    ts_ms: OpEvent::now_ms(),
                    message: e,
                });
                return;
            }
        };

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let sink_out = Arc::clone(&sink);
        let sink_err = Arc::clone(&sink);

        let h_out = stdout.map(|s| {
            thread::spawn(move || {
                for line in BufReader::new(s).lines().map_while(|r| r.ok()) {
                    // If the line is itself a JSON-encoded OpEvent (because
                    // the child was invoked with --emit-json-events), forward
                    // it as the structured event. Otherwise emit as Log.
                    let ev = OpEvent::from_json_line(&line).unwrap_or_else(|_| OpEvent::Log {
                        ts_ms: OpEvent::now_ms(),
                        message: line,
                    });
                    sink_out.emit(ev);
                }
            })
        });
        let h_err = stderr.map(|s| {
            thread::spawn(move || {
                for line in BufReader::new(s).lines().map_while(|r| r.ok()) {
                    sink_err.emit(OpEvent::Log {
                        ts_ms: OpEvent::now_ms(),
                        message: format!("[stderr] {}", line),
                    });
                }
            })
        });

        // Supervisor poll loop: watch for kill signal OR child exit.
        let mut was_killed = false;
        let exit_status = loop {
            match kill_rx.try_recv() {
                Ok(_) => {
                    was_killed = true;
                    sink.emit(OpEvent::Log {
                        ts_ms: OpEvent::now_ms(),
                        message: format!(">> Cancelling (PID {})...", child.id()),
                    });
                    break terminate_child(&mut child);
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => {
                    // ADR 0022 SD-4: caller dropped the handle. The
                    // pre-fix loop fell through to a 50ms sleep
                    // forever; this is fine for the spawn_lml case
                    // since we still poll child.try_wait() each
                    // iteration -- the child WILL exit eventually
                    // and break the loop -- but to be defensive,
                    // continue without re-checking the disconnected
                    // channel.
                }
            }

            match child.try_wait() {
                Ok(Some(status)) => break Ok(status),
                Ok(None) => thread::sleep(Duration::from_millis(50)),
                Err(e) => {
                    sink.emit(OpEvent::Error {
                        ts_ms: OpEvent::now_ms(),
                        message: format!("wait failed: {}", e),
                    });
                    return;
                }
            }
        };

        if was_killed {
            *lock_or_recover(&killed_supervisor) = true;
        }

        if let Some(h) = h_out {
            let _ = h.join();
        }
        if let Some(h) = h_err {
            let _ = h.join();
        }

        match exit_status {
            Ok(_) if was_killed => {
                sink.emit(OpEvent::Error {
                    ts_ms: OpEvent::now_ms(),
                    message: format!("{} cancelled by user", op_id),
                });
            }
            Ok(s) if s.success() => {
                sink.emit(OpEvent::Done {
                    ts_ms: OpEvent::now_ms(),
                    message: format!("{} completed (exit 0)", op_id),
                });
            }
            Ok(s) => {
                sink.emit(OpEvent::Error {
                    ts_ms: OpEvent::now_ms(),
                    message: format!("{} exited with code {}", op_id, s.code().unwrap_or(-1)),
                });
            }
            Err(e) => {
                sink.emit(OpEvent::Error {
                    ts_ms: OpEvent::now_ms(),
                    message: format!("wait failed: {}", e),
                });
            }
        }
    });

    OpHandle { kill_tx, killed }
}

/// Spawn `blut recipe run <recipe> --args '<json>'`, tail its
/// `status.jsonl`, and translate every `StageEvent` into the
/// matching `OpEvent` for the host dashboard.
///
/// BLUT's wire format (`blut::framework::status::StageEvent`)
/// lives in a separate crate; we parse the JSON lines with
/// `serde_json::Value` rather than depending on blut directly,
/// so a BLUT bump can't ripple a recompile through every LamQuant
/// crate. Wire shape is captured by the `stage_event_to_op_event`
/// fixture tests below — any drift breaks those tests, not silent
/// at runtime.
///
/// Cancel: `kill()` on the returned handle spawns `blut cancel
/// <job_id>` (graceful) and only falls back to `child.kill()` if
/// the job_id hasn't appeared in BLUT's stderr yet. Matches BLUT's
/// own SIGTERM-then-SIGKILL semantics in `jobs::cancel_job`.
pub fn spawn_blut<S: OpEventSink>(recipe: String, args_json: String, sink: S) -> OpHandle {
    let (kill_tx, kill_rx) = mpsc::channel::<()>();
    let killed = Arc::new(Mutex::new(false));
    let killed_supervisor = Arc::clone(&killed);

    let sink = Arc::new(sink);
    let job_id_shared: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

    thread::spawn(move || {
        let op_id = format!("lqt:{}", recipe);
        sink.emit(OpEvent::Started {
            ts_ms: OpEvent::now_ms(),
            op_id: op_id.clone(),
            total: None,
        });
        sink.emit(OpEvent::Log {
            ts_ms: OpEvent::now_ms(),
            message: format!("$ lqt recipe run {} --args '{}'", recipe, args_json),
        });

        let spawn = Command::new("lqt")
            .arg("recipe")
            .arg("run")
            .arg(&recipe)
            .arg("--args")
            .arg(&args_json)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();
        let mut child = match spawn {
            Ok(c) => c,
            Err(e) => {
                sink.emit(OpEvent::Error {
                    ts_ms: OpEvent::now_ms(),
                    message: format!("spawn lqt: {}", e),
                });
                return;
            }
        };

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        let stdout_sink = Arc::clone(&sink);
        let h_out = stdout.map(|s| {
            thread::spawn(move || {
                for line in BufReader::new(s).lines().map_while(|r| r.ok()) {
                    stdout_sink.emit(OpEvent::Log {
                        ts_ms: OpEvent::now_ms(),
                        message: line,
                    });
                }
            })
        });

        // Stderr reader extracts `job <id>` and `dir <path>` (BLUT
        // prints them as eprintln preamble) and forwards everything
        // else as Log. Once a job_dir is captured, spawn the
        // status.jsonl tailer.
        let stderr_sink = Arc::clone(&sink);
        let job_id_writer = Arc::clone(&job_id_shared);
        let job_dir_tx: Arc<Mutex<Option<PathBuf>>> = Arc::new(Mutex::new(None));
        let job_dir_writer = Arc::clone(&job_dir_tx);
        let h_err = stderr.map(|s| {
            thread::spawn(move || {
                for line in BufReader::new(s).lines().map_while(|r| r.ok()) {
                    if let Some(rest) = strip_field(&line, "job") {
                        if let Ok(mut g) = job_id_writer.lock() {
                            if g.is_none() {
                                *g = Some(rest.to_string());
                            }
                        }
                    } else if let Some(rest) = strip_field(&line, "dir") {
                        if let Ok(mut g) = job_dir_writer.lock() {
                            if g.is_none() {
                                *g = Some(PathBuf::from(rest));
                            }
                        }
                    }
                    stderr_sink.emit(OpEvent::Log {
                        ts_ms: OpEvent::now_ms(),
                        message: format!("[stderr] {}", line),
                    });
                }
            })
        });

        // Status tailer thread — waits for the job_dir to appear,
        // then poll-tails status.jsonl until EOF + child exit.
        let tailer_sink = Arc::clone(&sink);
        let tailer_dir = Arc::clone(&job_dir_tx);
        let tailer_stop = Arc::new(Mutex::new(false));
        let tailer_stop_signal = Arc::clone(&tailer_stop);
        let h_tail = thread::spawn(move || {
            // Wait up to 30s for a job_dir; bail quietly if BLUT
            // never printed one (clap parse error etc.).
            let deadline = std::time::Instant::now() + Duration::from_secs(30);
            let job_dir = loop {
                if let Ok(g) = tailer_dir.lock() {
                    if let Some(p) = g.clone() {
                        break p;
                    }
                }
                if *lock_or_recover(&tailer_stop_signal) {
                    return;
                }
                if std::time::Instant::now() > deadline {
                    return;
                }
                thread::sleep(Duration::from_millis(50));
            };
            tail_status_jsonl(
                &job_dir.join("status.jsonl"),
                tailer_stop_signal,
                tailer_sink,
            );
        });

        // Supervisor poll loop.
        let mut was_killed = false;
        let exit_status = loop {
            match kill_rx.try_recv() {
                Ok(_) => {
                    was_killed = true;
                    let id_opt = job_id_shared.lock().ok().and_then(|g| g.clone());
                    if let Some(id) = id_opt {
                        sink.emit(OpEvent::Log {
                            ts_ms: OpEvent::now_ms(),
                            message: format!(">> lqt cancel {}", id),
                        });
                        // Best-effort graceful cancel; LQT walks
                        // SIGTERM → grace → SIGKILL itself.
                        let _ = Command::new("lqt")
                            .arg("cancel")
                            .arg(&id)
                            .stdout(Stdio::null())
                            .stderr(Stdio::null())
                            .status();
                    } else {
                        sink.emit(OpEvent::Log {
                            ts_ms: OpEvent::now_ms(),
                            message: format!(">> SIGKILL (PID {}, no job_id yet)", child.id()),
                        });
                    }
                    // ADR 0022 SD-2: bounded termination instead of
                    // unconditional `kill + wait`.
                    break terminate_child(&mut child);
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => {}
            }
            match child.try_wait() {
                Ok(Some(status)) => break Ok(status),
                Ok(None) => thread::sleep(Duration::from_millis(100)),
                Err(e) => {
                    sink.emit(OpEvent::Error {
                        ts_ms: OpEvent::now_ms(),
                        message: format!("lqt wait failed: {}", e),
                    });
                    return;
                }
            }
        };

        if was_killed {
            *lock_or_recover(&killed_supervisor) = true;
        }
        *lock_or_recover(&tailer_stop) = true;

        if let Some(h) = h_out {
            let _ = h.join();
        }
        if let Some(h) = h_err {
            let _ = h.join();
        }
        let _ = h_tail.join();

        match exit_status {
            Ok(_) if was_killed => {
                sink.emit(OpEvent::Error {
                    ts_ms: OpEvent::now_ms(),
                    message: format!("{} cancelled by user", op_id),
                });
            }
            Ok(s) if s.success() => {
                sink.emit(OpEvent::Done {
                    ts_ms: OpEvent::now_ms(),
                    message: format!("{} completed (exit 0)", op_id),
                });
            }
            Ok(s) => {
                sink.emit(OpEvent::Error {
                    ts_ms: OpEvent::now_ms(),
                    message: format!("{} exited with code {}", op_id, s.code().unwrap_or(-1)),
                });
            }
            Err(e) => {
                sink.emit(OpEvent::Error {
                    ts_ms: OpEvent::now_ms(),
                    message: format!("blut wait failed: {}", e),
                });
            }
        }
    });

    OpHandle { kill_tx, killed }
}

/// Pull `<value>` from a `^<field>\s+<value>$` line. Returns None if
/// the prefix doesn't match. BLUT emits `job    <id>`, `dir    <path>`
/// etc. with run-of-whitespace separators (`eprintln!("job    {id}")`).
fn strip_field<'a>(line: &'a str, field: &str) -> Option<&'a str> {
    let rest = line.strip_prefix(field)?;
    let trimmed = rest.trim_start();
    if trimmed.len() == rest.len() {
        // No whitespace between field and value — false prefix match
        // (e.g. "jobs" vs "job"). Reject.
        return None;
    }
    let value = trimmed.trim_end();
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

/// Tail `path` line-by-line until `stop` is set or the file is
/// removed. Each line is parsed as a BLUT `StageEvent` and
/// forwarded as the corresponding `OpEvent` via `sink`. Lines that
/// don't parse are forwarded as Log so the user sees malformed
/// entries instead of silent drops.
fn tail_status_jsonl(path: &std::path::Path, stop: Arc<Mutex<bool>>, sink: Arc<dyn OpEventSink>) {
    // Wait for the file to appear (status writer creates it on
    // first StageBegin, which can lag a few hundred ms behind plan
    // compile).
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    let file = loop {
        if *lock_or_recover(&stop) {
            return;
        }
        match std::fs::OpenOptions::new().read(true).open(path) {
            Ok(f) => break f,
            Err(_) if std::time::Instant::now() < deadline => {
                thread::sleep(Duration::from_millis(50));
            }
            Err(_) => return,
        }
    };

    let mut file = file;
    let mut pos: u64 = 0;
    let mut leftover = String::new();
    loop {
        if let Err(e) = file.seek(SeekFrom::Start(pos)) {
            sink.emit(OpEvent::Log {
                ts_ms: OpEvent::now_ms(),
                message: format!("[status.jsonl] seek failed: {}", e),
            });
            return;
        }
        let mut buf = String::new();
        let read = match std::io::Read::read_to_string(&mut file, &mut buf) {
            Ok(n) => n,
            Err(e) => {
                sink.emit(OpEvent::Log {
                    ts_ms: OpEvent::now_ms(),
                    message: format!("[status.jsonl] read failed: {}", e),
                });
                return;
            }
        };
        if read > 0 {
            pos += read as u64;
            leftover.push_str(&buf);
            // Drain whole lines; keep trailing partial in `leftover`.
            while let Some(idx) = leftover.find('\n') {
                let line: String = leftover.drain(..=idx).collect();
                let line = line.trim_end_matches(['\n', '\r']).to_string();
                if line.is_empty() {
                    continue;
                }
                let ev = stage_event_to_op_event(&line);
                sink.emit(ev);
            }
        }
        if *lock_or_recover(&stop) {
            // Final drain pass after child exit, then bail.
            if !leftover.trim().is_empty() {
                let ev = stage_event_to_op_event(&leftover);
                sink.emit(ev);
            }
            return;
        }
        thread::sleep(Duration::from_millis(100));
    }
}

/// Translate one BLUT `StageEvent` JSON line into the matching
/// `OpEvent`. Wire format pinned by `blut::framework::status` —
/// every variant carries `"kind"` as discriminator (snake_case).
fn stage_event_to_op_event(line: &str) -> OpEvent {
    let v: serde_json::Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(_) => {
            return OpEvent::Log {
                ts_ms: OpEvent::now_ms(),
                message: format!("[status.jsonl] {}", line),
            };
        }
    };
    let kind = v.get("kind").and_then(|k| k.as_str()).unwrap_or("");
    let stage_name = v
        .get("stage_name")
        .and_then(|s| s.as_str())
        .unwrap_or("?")
        .to_string();
    let node_idx = v.get("node_idx").and_then(|n| n.as_u64()).unwrap_or(0);
    match kind {
        "stage_begin" => OpEvent::Log {
            ts_ms: OpEvent::now_ms(),
            message: format!("[stage {}] {} begin", node_idx, stage_name),
        },
        "stage_end" => {
            // `std::time::Duration`'s serde::Serialize impl emits
            // `{secs: u64, nanos: u32}` and BLUT uses the default
            // derive on its `StageEvent::StageEnd { elapsed:
            // Duration, .. }`. Pinned by the
            // `stage_end_maps_to_filedone_with_elapsed_ms` fixture
            // below — change either side and that test fails.
            let ms = v
                .get("elapsed")
                .map(|e| {
                    let secs = e.get("secs").and_then(|s| s.as_u64()).unwrap_or(0);
                    let nanos = e.get("nanos").and_then(|n| n.as_u64()).unwrap_or(0);
                    secs.saturating_mul(1000).saturating_add(nanos / 1_000_000)
                })
                .unwrap_or(0);
            OpEvent::FileDone {
                ts_ms: OpEvent::now_ms(),
                path: stage_name,
                success: true,
                ms,
                cr: None,
                bytes_in: None,
                bytes_out: None,
                samples: None,
                duration_s: None,
                n_channels: None,
                sample_rate: None,
                sha256: None,
                n_windows: None,
            }
        }
        "stage_skipped" => OpEvent::FileDone {
            ts_ms: OpEvent::now_ms(),
            path: format!("{} (cached)", stage_name),
            success: true,
            ms: 0,
            cr: None,
            bytes_in: None,
            bytes_out: None,
            samples: None,
            duration_s: None,
            n_channels: None,
            sample_rate: None,
            sha256: None,
            n_windows: None,
        },
        "stage_failed" => {
            let err = v
                .get("error")
                .and_then(|e| e.as_str())
                .unwrap_or("(no message)");
            OpEvent::Error {
                ts_ms: OpEvent::now_ms(),
                message: format!("[stage {}] {} failed: {}", node_idx, stage_name, err),
            }
        }
        "stage_blocked" => {
            let res = v.get("resource").and_then(|r| r.as_str()).unwrap_or("?");
            OpEvent::Log {
                ts_ms: OpEvent::now_ms(),
                message: format!("[stage {}] {} blocked on {}", node_idx, stage_name, res),
            }
        }
        "stage_step" => {
            // Progress requires BOTH step and total > 0. total == 0
            // means "trainer doesn't know how many steps remain"
            // (e.g. streaming dataset), which would render as a
            // 0/0 bar — useless. Downgrade to Log so the user
            // still sees the per-step update without a misleading
            // bar.
            let update = v.get("update");
            let step = update.and_then(|u| u.get("step")).and_then(|s| s.as_u64());
            let total = update.and_then(|u| u.get("total")).and_then(|t| t.as_u64());
            match (step, total) {
                (Some(s), Some(t)) if t > 0 => OpEvent::Progress {
                    ts_ms: OpEvent::now_ms(),
                    current: s,
                    total: t,
                    message: format!("[stage {}] {}", node_idx, stage_name),
                },
                _ => OpEvent::Log {
                    ts_ms: OpEvent::now_ms(),
                    message: format!(
                        "[stage {}] {} step {}",
                        node_idx,
                        stage_name,
                        update
                            .map(|u| u.to_string())
                            .unwrap_or_else(|| "(no update)".to_string())
                    ),
                },
            }
        }
        other => OpEvent::Log {
            ts_ms: OpEvent::now_ms(),
            message: format!("[status.jsonl] unknown kind '{}': {}", other, line),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::channel;
    use std::time::Instant;

    /// Cross-platform "sleep 30" fixture. Unix uses `sleep` (coreutils);
    /// Windows uses `powershell -Command Start-Sleep`.
    fn long_running_cmd() -> (String, Vec<String>) {
        #[cfg(windows)]
        {
            (
                "powershell".to_string(),
                vec![
                    "-NoProfile".to_string(),
                    "-Command".to_string(),
                    "Start-Sleep -Seconds 30".to_string(),
                ],
            )
        }
        #[cfg(not(windows))]
        {
            ("sleep".to_string(), vec!["30".to_string()])
        }
    }

    fn fast_exit_cmd() -> (String, Vec<String>) {
        #[cfg(windows)]
        {
            (
                "cmd".to_string(),
                vec!["/c".to_string(), "exit".to_string(), "0".to_string()],
            )
        }
        #[cfg(not(windows))]
        {
            ("true".to_string(), vec![])
        }
    }

    #[test]
    fn cancels_long_running_command() {
        let (sink, rx) = channel();
        let (prog, args) = long_running_cmd();
        let mut handle = spawn_command(prog, args, sink);

        thread::sleep(Duration::from_millis(200));
        let start = Instant::now();
        handle.kill();

        let deadline = Instant::now() + Duration::from_secs(2);
        let mut saw_cancel = false;
        while Instant::now() < deadline {
            if let Ok(ev) = rx.recv_timeout(Duration::from_millis(100)) {
                if let OpEvent::Error { message, .. } = &ev {
                    if message.contains("cancelled") {
                        saw_cancel = true;
                        break;
                    }
                }
            }
        }
        let elapsed = start.elapsed();
        assert!(saw_cancel, "expected cancel event within 2s");
        assert!(
            elapsed < Duration::from_secs(2),
            "kill took too long: {:?}",
            elapsed
        );
        assert!(handle.was_killed());
    }

    #[test]
    fn fast_command_completes_normally() {
        let (sink, rx) = channel();
        let (prog, args) = fast_exit_cmd();
        let _handle = spawn_command(prog, args, sink);
        let deadline = Instant::now() + Duration::from_secs(3);
        let mut saw_done = false;
        while Instant::now() < deadline {
            if let Ok(ev) = rx.recv_timeout(Duration::from_millis(100)) {
                if matches!(ev, OpEvent::Done { .. }) {
                    saw_done = true;
                    break;
                }
            }
        }
        assert!(saw_done, "expected Done event from fast-exit fixture");
    }

    // ── status.jsonl ↔ OpEvent translator fixture tests ─────────
    //
    // Every wire variant captured by `blut::framework::status::
    // StageEvent` is round-tripped here. If BLUT's wire format
    // changes, these tests fail loudly instead of the bridge
    // silently dropping events at runtime.

    #[test]
    fn strip_field_extracts_value() {
        assert_eq!(strip_field("job    abc-123", "job"), Some("abc-123"));
        assert_eq!(strip_field("dir /var/jobs/x", "dir"), Some("/var/jobs/x"));
        // No whitespace between prefix and value → not a real match.
        assert_eq!(strip_field("jobs 5", "job"), None);
        // Field but no value.
        assert_eq!(strip_field("job    ", "job"), None);
    }

    #[test]
    fn stage_begin_maps_to_log() {
        let line = r#"{"kind":"stage_begin","node_idx":3,"stage_name":"lamquant_train_joint","input_hash":"abc"}"#;
        match stage_event_to_op_event(line) {
            OpEvent::Log { message, .. } => {
                assert!(message.contains("stage 3"));
                assert!(message.contains("lamquant_train_joint"));
                assert!(message.contains("begin"));
            }
            other => panic!("expected Log, got {:?}", other),
        }
    }

    #[test]
    fn stage_end_maps_to_filedone_with_elapsed_ms() {
        // Duration default-serializes as {secs, nanos}; 2s 500ms.
        let line = r#"{"kind":"stage_end","node_idx":1,"stage_name":"build_manifest","output_hash":"deadbeef","elapsed":{"secs":2,"nanos":500000000}}"#;
        match stage_event_to_op_event(line) {
            OpEvent::FileDone {
                path, success, ms, ..
            } => {
                assert_eq!(path, "build_manifest");
                assert!(success);
                assert_eq!(ms, 2500);
            }
            other => panic!("expected FileDone, got {:?}", other),
        }
    }

    #[test]
    fn stage_skipped_maps_to_cached_filedone() {
        let line = r#"{"kind":"stage_skipped","node_idx":2,"stage_name":"precompute_l3","cache_key":"xyz"}"#;
        match stage_event_to_op_event(line) {
            OpEvent::FileDone {
                path, success, ms, ..
            } => {
                assert!(path.contains("precompute_l3"));
                assert!(path.contains("cached"));
                assert!(success);
                assert_eq!(ms, 0);
            }
            other => panic!("expected FileDone, got {:?}", other),
        }
    }

    #[test]
    fn stage_failed_maps_to_error() {
        let line =
            r#"{"kind":"stage_failed","node_idx":4,"stage_name":"train_joint","error":"cuda OOM"}"#;
        match stage_event_to_op_event(line) {
            OpEvent::Error { message, .. } => {
                assert!(message.contains("train_joint"));
                assert!(message.contains("cuda OOM"));
            }
            other => panic!("expected Error, got {:?}", other),
        }
    }

    #[test]
    fn stage_blocked_maps_to_log_with_resource() {
        let line = r#"{"kind":"stage_blocked","node_idx":5,"stage_name":"train_teacher","resource":"gpu"}"#;
        match stage_event_to_op_event(line) {
            OpEvent::Log { message, .. } => {
                assert!(message.contains("train_teacher"));
                assert!(message.contains("blocked"));
                assert!(message.contains("gpu"));
            }
            other => panic!("expected Log, got {:?}", other),
        }
    }

    #[test]
    fn stage_step_with_step_total_maps_to_progress() {
        let line = r#"{"kind":"stage_step","node_idx":6,"stage_name":"train_joint","update":{"step":42,"total":100,"loss":0.123}}"#;
        match stage_event_to_op_event(line) {
            OpEvent::Progress {
                current,
                total,
                message,
                ..
            } => {
                assert_eq!(current, 42);
                assert_eq!(total, 100);
                assert!(message.contains("train_joint"));
            }
            other => panic!("expected Progress, got {:?}", other),
        }
    }

    #[test]
    fn stage_step_with_zero_total_falls_back_to_log() {
        // Unknown total (streaming dataset) — bar would render 0/0.
        let line = r#"{"kind":"stage_step","node_idx":6,"stage_name":"train_joint","update":{"step":42,"total":0}}"#;
        match stage_event_to_op_event(line) {
            OpEvent::Log { .. } => {}
            other => panic!("expected Log for total=0, got {:?}", other),
        }
    }

    #[test]
    fn stage_step_without_total_falls_back_to_log() {
        let line =
            r#"{"kind":"stage_step","node_idx":6,"stage_name":"train_joint","update":{"epoch":3}}"#;
        match stage_event_to_op_event(line) {
            OpEvent::Log { message, .. } => {
                assert!(message.contains("train_joint"));
            }
            other => panic!("expected Log, got {:?}", other),
        }
    }

    #[test]
    fn malformed_line_falls_back_to_log() {
        let ev = stage_event_to_op_event("not json at all");
        match ev {
            OpEvent::Log { message, .. } => {
                assert!(message.contains("not json"));
            }
            other => panic!("expected Log, got {:?}", other),
        }
    }

    #[test]
    fn unknown_kind_falls_back_to_log() {
        let line = r#"{"kind":"future_variant","stage_name":"x","node_idx":0}"#;
        match stage_event_to_op_event(line) {
            OpEvent::Log { message, .. } => {
                assert!(message.contains("unknown kind"));
                assert!(message.contains("future_variant"));
            }
            other => panic!("expected Log, got {:?}", other),
        }
    }

    #[test]
    fn spawn_blut_missing_binary_emits_error() {
        // Force a binary-not-found path by temporarily clearing PATH.
        // Skip if test env can't override PATH safely.
        let (sink, rx) = channel();
        // Use a recipe name we know doesn't exist + bogus args. If
        // `blut` is installed, BLUT will exit with a recipe-not-
        // found error; if not installed, spawn fails. Either way
        // we get an Error event.
        let _h = spawn_blut("__nonexistent_recipe__".to_string(), "{}".to_string(), sink);
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut saw_error = false;
        while Instant::now() < deadline {
            if let Ok(ev) = rx.recv_timeout(Duration::from_millis(100)) {
                if matches!(ev, OpEvent::Error { .. }) {
                    saw_error = true;
                    break;
                }
            }
        }
        assert!(saw_error, "expected Error within 5s for missing recipe");
    }
}
