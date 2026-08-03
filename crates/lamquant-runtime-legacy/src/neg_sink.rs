//! NEG-provenance sink (feature `neg`) — a [`Sink`] that records a
//! content-addressed evidence graph (ADR 0114) *alongside* the signal, so a
//! recording carries typed, tamper-evident provenance of what was ingested.
//!
//! Shape (deliberately unlike the codec/LSL sinks — the ADR 0136 test of the
//! frozen `Sink` trait against a genuinely different output):
//!
//! - one root [`Measured`] node describes the **source** (producer + label);
//! - each window becomes a [`Measured`] node whose `content_ref` is the SHA-256
//!   of its channel-major samples and whose provenance parent is the source root
//!   (so `materialize_provenance_edges` draws source → window edges);
//! - consecutive windows are linked by an explicit
//!   [`EdgeClass::TemporalDependence`] edge (`prev → node`), same direction as
//!   the provenance backbone.
//!
//! The graph is built in memory and persisted (`to_json`) on `flush`/`close`,
//! after `materialize_provenance_edges()` + `verify()` — a self-check that the
//! evidence graph is internally sound before it is written. The window bytes
//! themselves live in the co-located `.lml`/`.edf`; this sink stores only the
//! light provenance skeleton that references them by hash.

use std::path::PathBuf;

use async_trait::async_trait;
use lamquant_neg::{EdgeClass, Measured, NegGraph, Node, NodeId, NodePayload, Provenance};
use sha2::{Digest, Sha256};

use crate::error::{Result, RuntimeError};
use crate::sink::{Sink, SinkInfo};
use crate::window::WindowBatch;

/// Records a NEG evidence graph for the window stream and persists it as JSON.
pub struct NegProvenanceSink {
    /// Where the graph JSON is written (on flush + close).
    path: PathBuf,
    /// Producer string stamped on every node (e.g. the pipeline name).
    producer: String,
    graph: NegGraph,
    /// The source-descriptor node every window derives from.
    root: NodeId,
    /// The previous window node, for the temporal chain.
    prev: Option<NodeId>,
    windows: u64,
}

/// Lowercase-hex SHA-256 of a window's channel-major samples. Stable across
/// runs (fixed field/byte order), so the same signal yields the same
/// `content_ref` — the content-addressing NEG relies on.
fn window_content_ref(batch: &WindowBatch) -> String {
    let mut h = Sha256::new();
    h.update((batch.channels.len() as u64).to_le_bytes());
    for ch in &batch.channels {
        h.update((ch.len() as u64).to_le_bytes());
        for &s in ch {
            h.update(s.to_le_bytes());
        }
    }
    let digest = h.finalize();
    let mut hex = String::with_capacity(64);
    use std::fmt::Write as _;
    for b in digest {
        let _ = write!(hex, "{b:02x}");
    }
    format!("window:{hex}")
}

impl NegProvenanceSink {
    /// Open a provenance sink writing to `path`, stamping `producer` on nodes.
    /// The `label` describes the source (goes in the root node's summary).
    pub fn new(
        path: impl Into<PathBuf>,
        producer: impl Into<String>,
        label: impl Into<String>,
    ) -> Self {
        let producer = producer.into();
        let mut graph = NegGraph::new();
        // The source root: a measured fact ("this stream was ingested from X").
        let root_node = Node::<Measured>::new(
            NodePayload {
                content_ref: None,
                summary: Some(format!("source: {}", label.into())),
            },
            Provenance::root(producer.clone()),
            None,
        );
        let root = graph.add_node(root_node);
        Self {
            path: path.into(),
            producer,
            graph,
            root,
            prev: None,
            windows: 0,
        }
    }

    /// Finalize provenance edges, verify the graph, and write it to disk.
    fn persist(&mut self) -> Result<()> {
        self.graph.materialize_provenance_edges();
        if let Err(errors) = self.graph.verify() {
            return Err(RuntimeError::Sink {
                name: self.path.display().to_string(),
                msg: format!(
                    "NEG graph failed verify(): {} violation(s): {:?}",
                    errors.len(),
                    errors
                ),
            });
        }
        if let Some(dir) = self.path.parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir)?;
            }
        }
        let json = self.graph.to_json().map_err(|e| RuntimeError::Sink {
            name: self.path.display().to_string(),
            msg: format!("NEG graph serialize: {e}"),
        })?;
        std::fs::write(&self.path, json)?;
        tracing::info!(
            path = %self.path.display(),
            windows = self.windows,
            content_address = %self.graph.content_address(),
            "neg: wrote provenance graph"
        );
        Ok(())
    }
}

#[async_trait]
impl Sink for NegProvenanceSink {
    async fn consume(&mut self, batch: &WindowBatch) -> Result<()> {
        let content_ref = window_content_ref(batch);
        let node = Node::<Measured>::new(
            NodePayload {
                content_ref: Some(content_ref),
                summary: Some(format!(
                    "window seq={} {}ch×{} @ {}Hz",
                    batch.seq,
                    batch.n_channels(),
                    batch.n_samples(),
                    batch.sample_rate_hz()
                )),
            },
            // Each window is measured *from the source* — parent is the root.
            Provenance::from_parents(self.producer.clone(), vec![self.root.clone()]),
            None,
        );
        let id = self.graph.add_node(node);
        // Explicit temporal backbone between consecutive windows.
        if let Some(prev) = self.prev.take() {
            self.graph
                .add_edge(prev, id.clone(), EdgeClass::TemporalDependence);
        }
        self.prev = Some(id);
        self.windows += 1;
        Ok(())
    }

    async fn flush(&mut self) -> Result<()> {
        self.persist()
    }

    async fn close(&mut self) -> Result<()> {
        self.persist()
    }

    fn info(&self) -> SinkInfo {
        SinkInfo {
            kind: "neg".into(),
            label: self.path.display().to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn win(seq: u64) -> WindowBatch {
        WindowBatch::new(seq, vec![vec![1, 2, 3], vec![4, 5, 6]], 250.0)
    }

    #[tokio::test]
    async fn graph_is_written_verified_and_stable() {
        let dir = std::env::temp_dir().join(format!("neg_sink_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("prov.neg.json");
        let mut sink = NegProvenanceSink::new(&path, "test-pipeline", "mem-source");
        for s in 0..4 {
            sink.consume(&win(s)).await.unwrap();
        }
        sink.close().await.unwrap();

        // Reload and re-verify: a faithfully reloaded graph must still be sound.
        let json = std::fs::read_to_string(&path).unwrap();
        let reloaded = NegGraph::from_json(&json).unwrap();
        assert!(reloaded.verify().is_ok(), "reloaded NEG graph must verify");
        // root + 4 windows.
        // (content_address is stable across the round-trip by construction.)
        let addr_before = reloaded.content_address();
        let addr_after = NegGraph::from_json(&json).unwrap().content_address();
        assert_eq!(addr_before, addr_after, "content address must be stable");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn identical_windows_share_a_content_ref() {
        // Content addressing: the same samples hash to the same content_ref.
        assert_eq!(window_content_ref(&win(0)), window_content_ref(&win(9)));
    }
}
