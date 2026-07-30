//! Event sinks — different lifetime models for different consumers.
//!
//! - `MpscSink` is for in-process consumers (Rust TUI). Forwards each
//!   `OpEvent` into a `std::sync::mpsc::Sender` and lets the consumer's
//!   render loop drain on each tick.
//!
//! - Tauri GUI provides its own sink (in `gui/src-tauri/src/op.rs`) that
//!   updates a shared `OpProgressSnapshot` on every event AND emits Tauri
//!   events only on state transitions. Frontend polls `op_snapshot` for
//!   incremental progress at 200ms — Tauri's event delivery is less
//!   reliable than its sync command path under load.
//!
//! The `OpEventSink` trait abstracts both lifetimes without leaking into
//! the runner's signature.

use std::sync::mpsc;

use crate::OpEvent;

/// Coarse-grained state for poll-based readers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum OpState {
    /// Spawned but no Started event seen yet.
    Pending,
    /// Started, no terminal event yet.
    Running,
    /// User pressed cancel; waiting for the child to exit.
    Cancelling,
    /// Done variant received.
    Done,
    /// Error variant received (real failure OR cancellation).
    Failed,
}

/// Snapshot of op progress for poll-based readers.
///
/// Tauri GUI's frontend polls this every 200ms via the `op_snapshot` Tauri
/// command instead of subscribing to per-event Tauri events. Tauri's event
/// delivery under load (window minimize, tab switch, HMR) drops events;
/// the snapshot is the source of truth for incremental progress.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OpProgressSnapshot {
    pub op_id: String,
    pub state: OpState,
    pub current: u64,
    pub total: u64,
    pub message: String,
    /// Most recent FileDone path, if any.
    pub last_file: Option<String>,
    /// Most recent terminal message (Done.message or Error.message).
    pub terminal_message: Option<String>,
    /// Wall-clock ms of the latest update — frontends use this to detect
    /// staleness if the runner hangs.
    pub updated_ms: i64,
}

impl OpProgressSnapshot {
    pub fn new(op_id: impl Into<String>) -> Self {
        Self {
            op_id: op_id.into(),
            state: OpState::Pending,
            current: 0,
            total: 0,
            message: String::new(),
            last_file: None,
            terminal_message: None,
            updated_ms: OpEvent::now_ms(),
        }
    }

    /// Apply an event to the snapshot. Idempotent except for `updated_ms`.
    pub fn apply(&mut self, event: &OpEvent) {
        self.updated_ms = OpEvent::now_ms();
        match event {
            OpEvent::Started { total, .. } => {
                self.state = OpState::Running;
                if let Some(t) = total {
                    self.total = *t;
                }
            }
            OpEvent::Progress {
                current,
                total,
                message,
                ..
            } => {
                self.current = *current;
                self.total = *total;
                self.message = message.clone();
            }
            OpEvent::FileDone { path, .. } => {
                self.last_file = Some(path.clone());
            }
            OpEvent::Done { message, .. } => {
                self.state = OpState::Done;
                self.terminal_message = Some(message.clone());
            }
            OpEvent::Error { message, .. } => {
                self.state = OpState::Failed;
                self.terminal_message = Some(message.clone());
            }
            OpEvent::Log { .. } => { /* logs don't change state */ }
        }
    }
}

/// Trait that runner consumers implement to receive op events.
///
/// Sink is shared via `Arc<S>` across the supervisor + stdout/stderr reader
/// threads, so it must be `Send + Sync + 'static`. Implementors typically
/// store any internal channels behind locks (`MpscSink` wraps a clonable
/// `Sender` which is itself `Sync`; the Tauri `TauriSink` puts its snapshot
/// behind a `Mutex`).
pub trait OpEventSink: Send + Sync + 'static {
    /// Forward a single event. MUST be cheap — runners may call this from
    /// the supervisor thread or from an stdout reader thread, and dropping
    /// events is preferable to blocking the runner.
    fn emit(&self, event: OpEvent);
}

/// In-process sink that forwards events to an mpsc channel — bounded or
/// unbounded. Used by the Rust TUI's output panel: the panel's `tick()`
/// drains the receiver each frame and updates its line buffer.
///
/// `mpsc::Sender` / `SyncSender` are not `Sync` on stable Rust, so we
/// wrap them in a `Mutex` to satisfy the `OpEventSink` trait bound. Lock
/// contention is negligible — events arrive far slower than a mutex can
/// release.
///
/// Backpressure policy for the bounded variant:
///   - emit() uses blocking `send`; on a full buffer the runner thread
///     PAUSES until the consumer drains room. NO events are dropped
///     under any condition — the CR distribution histogram, recent-
///     files list, and every Progress tick reach the dashboard.
///   - Buffer is sized to absorb a multi-second burst before any
///     blocking is observable (DEFAULT_CHANNEL_BOUND = 16384 ≈ a few
///     MB at typical OpEvent size). Realistic encode runs emit at
///     100-1000 events/sec while the TUI drains 64 events/tick at
///     20fps = 1280 events/sec, so steady-state drain > emit and
///     blocking is rare.
///   - When the runner DOES block, it's a real signal that the TUI
///     thread is stalled (panic-in-transit, debugger pause). Treat
///     that as a system-wide problem, not a sink-level concern.
///   - The OutputPanel's `lines` buffer is also capped (LINES_CAP=5000),
///     so total memory is bounded at both ends.
pub struct MpscSink {
    tx: std::sync::Mutex<MpscTx>,
}

enum MpscTx {
    Unbounded(mpsc::Sender<OpEvent>),
    Bounded(mpsc::SyncSender<OpEvent>),
}

impl MpscSink {
    /// Wrap an existing unbounded `Sender`. Use `channel()` /
    /// `bounded_channel()` in normal code; this constructor is for
    /// callers that already own a sender (e.g. tests injecting a
    /// shared bus).
    pub fn new(tx: mpsc::Sender<OpEvent>) -> Self {
        Self {
            tx: std::sync::Mutex::new(MpscTx::Unbounded(tx)),
        }
    }

    /// Wrap an existing bounded `SyncSender`. emit() will `try_send`
    /// and drop on overflow.
    pub fn new_bounded(tx: mpsc::SyncSender<OpEvent>) -> Self {
        Self {
            tx: std::sync::Mutex::new(MpscTx::Bounded(tx)),
        }
    }
}

/// Timeout for the bounded send path. ADR 0022 SD-5: a stuck TUI
/// (consumer never drains) used to block the runner thread
/// indefinitely on send. After this window, the event is dropped
/// with a stderr warning so the subprocess can keep producing
/// progress. Configurable via env for tests.
const BOUNDED_SEND_TIMEOUT_SECS: u64 = 2;

impl OpEventSink for MpscSink {
    fn emit(&self, event: OpEvent) {
        // ADR 0022 SD-3: poisoning recovery -- the lock() Err path
        // was already gracefully handled, but `let Ok` masked
        // diagnostic info. Recover via .into_inner() so the
        // poisoned-but-safe state still drains.
        let tx = match self.tx.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        match &*tx {
            MpscTx::Unbounded(s) => {
                let _ = s.send(event);
            }
            MpscTx::Bounded(s) => {
                // ADR 0022 SD-5: bounded send with manual timeout
                // loop. `SyncSender::send_timeout` is nightly-only;
                // implement via try_send + sleep until deadline.
                // Backpressure is intentional (Bible R33), but a
                // stuck consumer used to deadlock the runner
                // thread. After BOUNDED_SEND_TIMEOUT_SECS, drop
                // the event with a stderr warning so the
                // subprocess can keep producing progress.
                let deadline = std::time::Instant::now()
                    + std::time::Duration::from_secs(BOUNDED_SEND_TIMEOUT_SECS);
                let mut to_send = event;
                loop {
                    match s.try_send(to_send) {
                        Ok(()) => break,
                        Err(mpsc::TrySendError::Full(returned)) => {
                            if std::time::Instant::now() >= deadline {
                                eprintln!(
                                    "  WARNING: MpscSink bounded send timed out \
                                     after {}s (consumer stalled); dropping event",
                                    BOUNDED_SEND_TIMEOUT_SECS
                                );
                                break;
                            }
                            to_send = returned;
                            std::thread::sleep(std::time::Duration::from_millis(10));
                        }
                        Err(mpsc::TrySendError::Disconnected(_)) => {
                            // Receiver dropped (UI navigated away);
                            // terminal Done was the last useful event
                            // anyway, silent drop is acceptable.
                            break;
                        }
                    }
                }
            }
        }
    }
}

/// Receiving half of the op-event channel. Type alias so callers don't
/// have to spell out `mpsc::Receiver<OpEvent>` every time. Used by the
/// [`crate::transport::Transport`] trait return type.
pub type OpReceiver = mpsc::Receiver<OpEvent>;

/// Default bound for in-process op-event channels. Sized to absorb a
/// multi-second burst of FileDone events from a fast batch encode
/// (typically 100-1000/sec) before the bounded sender starts blocking
/// for backpressure, while still capping memory at `BUFFER ×
/// sizeof(OpEvent)` ≈ a few MB worst-case. At 16k slots the buffer
/// holds ~16s of burst at 1000 events/sec, far longer than any TUI
/// drain pause should plausibly take.
pub const DEFAULT_CHANNEL_BOUND: usize = 16384;

/// Convenience: create a paired (sender-sink, receiver) for in-process
/// use. UNBOUNDED — preserved for external callers that depended on
/// `send()` never blocking and never dropping. New TUI / GUI code
/// should use `bounded_channel` instead.
pub fn channel() -> (MpscSink, OpReceiver) {
    let (tx, rx) = mpsc::channel();
    (MpscSink::new(tx), rx)
}

/// Bounded channel. emit() will drop events when the buffer is full —
/// see `MpscSink` doc for the rationale. `bound` must be > 0; smaller
/// values give faster drop-detection at the cost of more frequent
/// drops under burst. Default via [`DEFAULT_CHANNEL_BOUND`].
pub fn bounded_channel(bound: usize) -> (MpscSink, OpReceiver) {
    assert!(bound > 0, "channel bound must be positive");
    let (tx, rx) = mpsc::sync_channel(bound);
    (MpscSink::new_bounded(tx), rx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_progresses_with_events() {
        let mut snap = OpProgressSnapshot::new("encode");
        assert_eq!(snap.state, OpState::Pending);

        snap.apply(&OpEvent::Started {
            ts_ms: 0,
            op_id: "encode".into(),
            total: Some(10),
        });
        assert_eq!(snap.state, OpState::Running);
        assert_eq!(snap.total, 10);

        snap.apply(&OpEvent::Progress {
            ts_ms: 0,
            current: 3,
            total: 10,
            message: "f3".into(),
        });
        assert_eq!(snap.current, 3);
        assert_eq!(snap.message, "f3");

        snap.apply(&OpEvent::FileDone {
            ts_ms: 0,
            path: "f3.lml".into(),
            success: true,
            ms: 1,
            cr: Some(2.0),
            bytes_in: None,
            bytes_out: None,
            samples: None,
            duration_s: None,
            n_channels: None,
            sample_rate: None,
            sha256: None,
            n_windows: None,
        });
        assert_eq!(snap.last_file.as_deref(), Some("f3.lml"));

        snap.apply(&OpEvent::Done {
            ts_ms: 0,
            message: "ok".into(),
        });
        assert_eq!(snap.state, OpState::Done);
    }

    #[test]
    fn mpsc_sink_round_trip() {
        let (sink, rx) = channel();
        sink.emit(OpEvent::Log {
            ts_ms: 0,
            message: "hi".into(),
        });
        match rx.try_recv().unwrap() {
            OpEvent::Log { message, .. } => assert_eq!(message, "hi"),
            _ => panic!("wrong variant"),
        }
    }
}
