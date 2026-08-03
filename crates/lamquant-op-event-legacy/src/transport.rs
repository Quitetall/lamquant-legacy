//! Multi-device transport abstraction.
//!
//! Lets a LamQuant op run on a remote peer instead of (or in addition to)
//! the local machine. The TUI picks a [`Peer`] and dispatches via a
//! [`Transport`]; the trait is concrete enough that an SSH-shelling
//! implementation can land today, abstract enough that gRPC / QUIC /
//! direct-TCP variants slot in later without API churn.
//!
//! ## Lifecycle
//!
//! 1. [`Transport::verify`] — cheap reachability + version check. Cached
//!    by callers; not called per op.
//! 2. [`Transport::health`] — periodic poll for the Peers panel.
//! 3. [`Transport::stage_input`] — make the input file visible to peer.
//!    Hybrid: hash-detect first (zero copy on shared FS), rsync fallback.
//! 4. [`Transport::dispatch`] — spawn the op on the peer, return an
//!    [`OpReceiver`] streaming [`OpEvent`]s back. Blocking caller drains.
//! 5. [`Transport::cancel`] — best-effort kill of a running op.
//!
//! ## Concrete impls
//!
//! - [`crate::transport::ssh::SshTransport`] — only impl shipped today
//!
//! Future variants: gRPC (`Transport for GrpcTransport`), QUIC, direct
//! TCP for low-latency LAN. All compose against the same trait.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::sink::OpReceiver;

/// One configured remote machine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Peer {
    /// Stable identifier — used in TrackedProcess.peer + state.selected_peer.
    pub id: String,
    /// Human-friendly label shown in the Peers panel.
    pub display: String,
    /// Network host (DNS or IP).
    pub host: String,
    /// Transport-specific config. The variant picks the impl.
    pub transport: TransportKind,
}

/// Tagged transport variant. Switching variant swaps which `Transport` impl
/// the dispatcher uses for this peer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum TransportKind {
    /// SSH transport — the only variant shipped today.
    Ssh(SshConfig),
    // Future variants reserved (do not add until they have an impl):
    // Grpc(GrpcConfig),
    // Quic(QuicConfig),
    // Direct(DirectConfig),
}

/// SSH transport config. Security stance is "explicit only — no agent
/// trust": the caller must specify a key path AND a peer-specific
/// known_hosts file. This rejects the common mistakes (using
/// `~/.ssh/id_rsa` for everything; no host-key check).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SshConfig {
    /// Username on the remote.
    pub user: String,
    /// Private key path. Used with `-i` and `-o IdentitiesOnly=yes`.
    /// Agent fallback is explicitly disabled at SSH level
    /// (`-o IdentitiesOnly=yes`).
    pub key_path: PathBuf,
    /// Per-peer known_hosts file containing ONLY this peer's host public
    /// key. Generate once with:
    ///   ssh-keyscan -H <host> > /path/to/peer-known-hosts
    /// Used with `-o UserKnownHostsFile=<path> -o StrictHostKeyChecking=yes`
    /// so SSH refuses connection if the host key doesn't match.
    pub known_hosts: PathBuf,
    /// Display-only fingerprint tag (e.g. "SHA256:abc123…"). Shown in the
    /// Peers panel so users can visually verify the configured host. SSH
    /// itself enforces the host key via `known_hosts` — this field is
    /// not used in security checks.
    #[serde(default)]
    pub host_fingerprint: String,
    /// SSH port. Defaults to 22 if omitted.
    #[serde(default = "default_ssh_port")]
    pub port: u16,
}

fn default_ssh_port() -> u16 {
    22
}

/// Result of a successful [`Transport::verify`] call.
#[derive(Debug, Clone)]
pub struct PeerInfo {
    /// LamQuant version reported by the peer (`lamquant --version`).
    pub version: String,
    /// GPU description if available (one-liner, e.g. "NVIDIA RTX 4090").
    pub gpu: Option<String>,
    /// Worker count peer is configured for.
    pub workers: u32,
}

/// Cached health result for the Peers panel.
#[derive(Debug, Clone)]
pub enum PeerHealth {
    /// Reachable, no op in flight.
    Idle,
    /// Reachable, currently running an op (description optional).
    Busy(String),
    /// Cannot reach (network down or process down).
    Unreachable,
    /// Reached the host, but SSH auth was rejected (wrong/missing key,
    /// fingerprint mismatch). Distinct from `Unreachable` — the host is
    /// up, the credentials are wrong.
    AuthFailed,
    /// Reachable but lamquant version is incompatible — refuse to dispatch.
    IncompatibleVersion { local: String, peer: String },
}

/// Path to a file on the remote peer. Returned by [`Transport::stage_input`]
/// and passed to [`Transport::dispatch`] in op args.
#[derive(Debug, Clone)]
pub struct RemotePath(pub String);

/// Opaque handle to a running remote op. Passed back to
/// [`Transport::cancel`] when the user presses Ctrl+C.
#[derive(Debug, Clone)]
pub struct RemoteHandle {
    /// Transport-specific identifier (e.g. SSH PID, gRPC stream ID).
    pub token: String,
}

/// Failure modes shared by every transport.
#[derive(Debug)]
pub enum TransportError {
    /// Peer didn't respond / network down / SSH refused.
    Unreachable(String),
    /// Auth failure (key rejected, fingerprint mismatch, etc.).
    AuthFailed(String),
    /// Peer version is incompatible with local. Hard refuse — never silently
    /// dispatch to a peer running a different major.minor than us.
    VersionMismatch { local: String, peer: String },
    /// Hash detect + rsync both failed.
    StagingFailed(String),
    /// Op spawn or stream parse failed mid-dispatch.
    DispatchFailed(String),
    /// User-initiated cancel succeeded.
    Cancelled,
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unreachable(s) => write!(f, "unreachable: {}", s),
            Self::AuthFailed(s) => write!(f, "auth failed: {}", s),
            Self::VersionMismatch { local, peer } => {
                write!(f, "version mismatch: local {}, peer {}", local, peer)
            }
            Self::StagingFailed(s) => write!(f, "staging failed: {}", s),
            Self::DispatchFailed(s) => write!(f, "dispatch failed: {}", s),
            Self::Cancelled => write!(f, "cancelled by user"),
        }
    }
}

impl std::error::Error for TransportError {}

/// A network transport that can run a LamQuant op on a remote peer.
///
/// Implementations are expected to be cheap to construct (per-peer state
/// is in [`Peer`]) and may keep internal connection pools / caches.
///
/// Send + Sync because the TUI's Peers panel may health-check from a
/// helper thread while the main loop dispatches an op.
pub trait Transport: Send + Sync {
    /// Verify the peer is reachable + version-compatible. Idempotent.
    /// Cheap (one SSH round-trip equivalent). Callers should cache.
    fn verify(&self, peer: &Peer) -> Result<PeerInfo, TransportError>;

    /// Stage a local input file on the peer. Implementations should
    /// optimize: if the peer already has the same content (shared FS,
    /// hash match), return its native path with no transfer. Otherwise
    /// transfer (rsync over SSH, gRPC chunked upload, etc.).
    fn stage_input(&self, peer: &Peer, local: &Path) -> Result<RemotePath, TransportError>;

    /// Dispatch an op to the peer. The op_id is the same one local
    /// callers pass to `op_spec` / `spawn_lml`; args reference the
    /// staged input via [`RemotePath`].
    ///
    /// Returns:
    /// - [`OpReceiver`] streaming [`OpEvent`]s as the op runs
    /// - [`RemoteHandle`] for later cancel
    ///
    /// Implementations MUST emit a terminal `OpEvent::Done` or
    /// `OpEvent::Error` before closing the channel — otherwise the TUI
    /// can't tell completion from disconnect.
    fn dispatch(
        &self,
        peer: &Peer,
        op_id: &str,
        args: &[String],
    ) -> Result<(OpReceiver, RemoteHandle), TransportError>;

    /// Best-effort cancel of a running op.
    fn cancel(&self, peer: &Peer, handle: &RemoteHandle) -> Result<(), TransportError>;

    /// Lightweight health probe — no version check, no input staging.
    /// The Peers panel calls this periodically to refresh status.
    fn health(&self, peer: &Peer) -> Result<PeerHealth, TransportError>;
}

pub mod ssh;
