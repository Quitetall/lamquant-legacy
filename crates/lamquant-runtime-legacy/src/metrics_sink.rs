//! Metrics sink (feature `metrics`) — a [`Sink`] that counts the windows and
//! samples flowing through its pipeline and exposes them on `GET /metrics` in
//! Prometheus text-exposition format (ADR 0136).
//!
//! Shape: unlike the codec/LSL sinks (which write signal) this one writes
//! *observability* — a second genuinely-different output that exercises the
//! frozen `Sink` trait. It mirrors the codec crate's `serve_metrics`
//! (`async_io.rs`) hand-rolled HTTP/1.1 responder (no `hyper`/`prometheus`
//! dep), but sources its counters from **per-pipeline** atomics owned by the
//! sink instead of the codec crate's process-global statics — so each pipeline's
//! throughput is labeled distinctly.
//!
//! The tiny HTTP server is started lazily on the first `consume` (so it binds
//! inside the engine's tokio runtime) and torn down on `close`. Per Bible R30,
//! only `GET /metrics` returns 200; anything else is 404.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::error::Result;
use crate::sink::{Sink, SinkInfo};
use crate::window::WindowBatch;

/// Per-pipeline throughput counters, shared between the sink (writer) and its
/// HTTP server task (reader).
struct Counters {
    pipeline: String,
    windows: AtomicU64,
    samples: AtomicU64,
}

/// Escape a Prometheus label value (`"`, `\`, newline) so a pipeline name with
/// odd characters cannot break the exposition format.
fn escape_label(v: &str) -> String {
    let mut out = String::with_capacity(v.len());
    for c in v.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            _ => out.push(c),
        }
    }
    out
}

/// Render the counters as Prometheus text-exposition format. Pure (no I/O), so
/// it is unit-testable without binding a socket — the same factoring the codec
/// crate uses (`render_metrics_text` split from `serve_metrics`).
fn render(c: &Counters) -> String {
    let label = format!("{{pipeline=\"{}\"}}", escape_label(&c.pipeline));
    let w = c.windows.load(Ordering::Relaxed);
    let s = c.samples.load(Ordering::Relaxed);
    format!(
        "# HELP lamquant_runtime_windows_total Windows consumed by this pipeline sink.\n\
         # TYPE lamquant_runtime_windows_total counter\n\
         lamquant_runtime_windows_total{label} {w}\n\
         # HELP lamquant_runtime_samples_total Samples (channel×time) consumed by this pipeline sink.\n\
         # TYPE lamquant_runtime_samples_total counter\n\
         lamquant_runtime_samples_total{label} {s}\n"
    )
}

/// Serve `GET /metrics` for `counters` on an already-bound `listener` until
/// `shutdown` fires. Split from the bind so a test can drive it on an ephemeral
/// port. Bible R30: only `/metrics` → 200; everything else → 404.
async fn serve(listener: TcpListener, counters: Arc<Counters>, shutdown: CancellationToken) {
    loop {
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => return,
            accept = listener.accept() => {
                let (mut stream, _peer) = match accept {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!("metrics: accept failed: {e}");
                        continue;
                    }
                };
                let counters = counters.clone();
                tokio::spawn(async move {
                    let mut buf = [0u8; 1024];
                    let n = match stream.read(&mut buf).await {
                        Ok(0) | Err(_) => return,
                        Ok(n) => n,
                    };
                    let req = String::from_utf8_lossy(&buf[..n]);
                    let response = if req.starts_with("GET /metrics ") {
                        let body = render(&counters);
                        format!(
                            "HTTP/1.1 200 OK\r\n\
                             Content-Type: text/plain; version=0.0.4\r\n\
                             Content-Length: {}\r\n\
                             Connection: close\r\n\r\n{body}",
                            body.len()
                        )
                    } else {
                        "HTTP/1.1 404 Not Found\r\n\
                         Content-Length: 0\r\nConnection: close\r\n\r\n"
                            .to_string()
                    };
                    let _ = stream.write_all(response.as_bytes()).await;
                });
            }
        }
    }
}

/// Counts windows/samples for one pipeline and serves them on `bind_addr`.
pub struct MetricsSink {
    counters: Arc<Counters>,
    bind_addr: String,
    shutdown: CancellationToken,
    server: Option<JoinHandle<()>>,
}

impl MetricsSink {
    /// A metrics sink that will serve `pipeline`'s counters on `bind_addr`
    /// (e.g. `"127.0.0.1:9109"`). The server starts on the first consumed window.
    pub fn new(bind_addr: impl Into<String>, pipeline: impl Into<String>) -> Self {
        Self {
            counters: Arc::new(Counters {
                pipeline: pipeline.into(),
                windows: AtomicU64::new(0),
                samples: AtomicU64::new(0),
            }),
            bind_addr: bind_addr.into(),
            shutdown: CancellationToken::new(),
            server: None,
        }
    }

    /// Lazily bind + spawn the HTTP server (idempotent). Called on first consume
    /// so it runs inside the engine's tokio runtime.
    fn ensure_server(&mut self) {
        if self.server.is_some() {
            return;
        }
        let addr = self.bind_addr.clone();
        let counters = self.counters.clone();
        let shutdown = self.shutdown.clone();
        self.server = Some(tokio::spawn(async move {
            match TcpListener::bind(&addr).await {
                Ok(listener) => {
                    tracing::info!("metrics: serving http://{addr}/metrics");
                    serve(listener, counters, shutdown).await;
                }
                Err(e) => tracing::warn!("metrics: bind {addr} failed: {e} (endpoint disabled)"),
            }
        }));
    }
}

#[async_trait]
impl Sink for MetricsSink {
    async fn consume(&mut self, batch: &WindowBatch) -> Result<()> {
        self.ensure_server();
        self.counters.windows.fetch_add(1, Ordering::Relaxed);
        let samples = (batch.n_channels() as u64) * (batch.n_samples() as u64);
        self.counters.samples.fetch_add(samples, Ordering::Relaxed);
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        self.shutdown.cancel();
        if let Some(h) = self.server.take() {
            h.abort();
        }
        Ok(())
    }

    fn info(&self) -> SinkInfo {
        SinkInfo {
            kind: "metrics".into(),
            label: self.bind_addr.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counters(p: &str, w: u64, s: u64) -> Counters {
        Counters {
            pipeline: p.into(),
            windows: AtomicU64::new(w),
            samples: AtomicU64::new(s),
        }
    }

    #[test]
    fn render_is_labeled_prometheus_text() {
        let text = render(&counters("eeg-a", 5, 630));
        assert!(text.contains("lamquant_runtime_windows_total{pipeline=\"eeg-a\"} 5"));
        assert!(text.contains("lamquant_runtime_samples_total{pipeline=\"eeg-a\"} 630"));
        assert!(text.contains("# TYPE lamquant_runtime_windows_total counter"));
    }

    #[test]
    fn label_values_are_escaped() {
        let text = render(&counters("odd\"name", 1, 1));
        assert!(text.contains("pipeline=\"odd\\\"name\""));
    }

    #[tokio::test]
    async fn consume_increments_per_pipeline_counters() {
        let mut sink = MetricsSink::new("127.0.0.1:0", "p");
        let batch = WindowBatch::new(0, vec![vec![1, 2, 3], vec![4, 5, 6]], 250.0);
        sink.consume(&batch).await.unwrap();
        sink.consume(&batch).await.unwrap();
        assert_eq!(sink.counters.windows.load(Ordering::Relaxed), 2);
        assert_eq!(sink.counters.samples.load(Ordering::Relaxed), 12); // 2 windows × 2ch × 3
        sink.close().await.unwrap();
    }

    #[tokio::test]
    async fn serve_answers_metrics_and_404s_others() {
        // Ephemeral port so the test never collides with a real service.
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let c = Arc::new(counters("live", 7, 700));
        let shutdown = CancellationToken::new();
        let srv = tokio::spawn(serve(listener, c, shutdown.clone()));

        // GET /metrics → 200 with our counter.
        let body = http_get(&addr.to_string(), "/metrics").await;
        assert!(body.contains("200 OK"), "metrics resp: {body}");
        assert!(body.contains("lamquant_runtime_windows_total{pipeline=\"live\"} 7"));

        // GET /other → 404.
        let other = http_get(&addr.to_string(), "/nope").await;
        assert!(other.contains("404 Not Found"), "other resp: {other}");

        shutdown.cancel();
        let _ = srv.await;
    }

    async fn http_get(addr: &str, path: &str) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
        s.write_all(format!("GET {path} HTTP/1.1\r\nHost: x\r\n\r\n").as_bytes())
            .await
            .unwrap();
        let mut out = Vec::new();
        s.read_to_end(&mut out).await.unwrap();
        String::from_utf8_lossy(&out).into_owned()
    }
}
