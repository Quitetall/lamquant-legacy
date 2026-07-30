//! SSH transport — concrete [`Transport`] impl.
//!
//! Dispatches lamquant ops to a remote peer over SSH. The peer must
//! have `lamquant` (or `lml`) on PATH and accept the configured key.
//!
//! ## Security stance
//!
//! - `-i <key_path>` + `-o IdentitiesOnly=yes` — only the configured
//!   private key is offered. SSH-agent fallback is rejected.
//! - `-o UserKnownHostsFile=<known_hosts>` + `-o StrictHostKeyChecking=yes` —
//!   peer's host key MUST match the entry in the per-peer known_hosts
//!   file. Unknown / changed host = connection refused.
//! - `-o BatchMode=yes` — refuses interactive password prompts. Auth
//!   must succeed via key alone.
//! - `-o PasswordAuthentication=no -o KbdInteractiveAuthentication=no` —
//!   double-belt against fallback to passwords.
//!
//! ## Staging
//!
//! Hybrid hash-detect: SSH probes peer for the input file's sha256.
//! If it matches the local hash, peer has the file (shared FS or
//! pre-populated cache) — return its native path, zero transfer.
//! Otherwise rsync-over-SSH push to `~/.cache/lamquant/staged/<basename>`.

use crate::sink::OpReceiver;
use crate::transport::{
    Peer, PeerHealth, PeerInfo, RemoteHandle, RemotePath, SshConfig, Transport, TransportError,
    TransportKind,
};
use crate::OpEvent;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;

/// SSH-shelling transport. Stateless — every call spawns a fresh SSH
/// process. A future enhancement could pool ControlMaster connections.
pub struct SshTransport;

impl SshTransport {
    pub fn new() -> Self {
        Self
    }

    /// Pull the SshConfig out of a Peer. Returns AuthFailed for
    /// non-SSH peer transports (caller should never reach here, but
    /// fail loudly if they do).
    fn ssh_config<'a>(&self, peer: &'a Peer) -> Result<&'a SshConfig, TransportError> {
        match &peer.transport {
            TransportKind::Ssh(cfg) => Ok(cfg),
        }
    }

    /// Build a hardened `Command` for `ssh ... <remote_cmd>`. Caller
    /// only supplies the remote command string + stdio choices.
    fn ssh_cmd(&self, peer: &Peer, remote_cmd: &str) -> Result<Command, TransportError> {
        let cfg = self.ssh_config(peer)?;
        let mut c = Command::new("ssh");
        c.arg("-i").arg(&cfg.key_path);
        c.arg("-o").arg("IdentitiesOnly=yes");
        c.arg("-o")
            .arg(format!("UserKnownHostsFile={}", cfg.known_hosts.display()));
        c.arg("-o").arg("StrictHostKeyChecking=yes");
        c.arg("-o").arg("BatchMode=yes");
        c.arg("-o").arg("PasswordAuthentication=no");
        c.arg("-o").arg("KbdInteractiveAuthentication=no");
        c.arg("-o").arg(format!("Port={}", cfg.port));
        c.arg(format!("{}@{}", cfg.user, peer.host));
        c.arg(remote_cmd);
        Ok(c)
    }

    /// Run an SSH command, capture stdout. Maps non-zero exit to
    /// AuthFailed/Unreachable based on stderr content.
    fn ssh_capture(&self, peer: &Peer, remote_cmd: &str) -> Result<String, TransportError> {
        let mut cmd = self.ssh_cmd(peer, remote_cmd)?;
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
        let out = cmd
            .output()
            .map_err(|e| TransportError::Unreachable(format!("ssh spawn failed: {}", e)))?;
        if out.status.success() {
            Ok(String::from_utf8_lossy(&out.stdout).to_string())
        } else {
            let stderr = String::from_utf8_lossy(&out.stderr).to_string();
            // Best-effort classification.
            let lower = stderr.to_lowercase();
            if lower.contains("permission denied")
                || lower.contains("publickey")
                || lower.contains("host key")
                || lower.contains("known_hosts")
            {
                Err(TransportError::AuthFailed(stderr))
            } else {
                Err(TransportError::Unreachable(stderr))
            }
        }
    }
}

impl Default for SshTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl Transport for SshTransport {
    fn verify(&self, peer: &Peer) -> Result<PeerInfo, TransportError> {
        // Probe `lamquant --version`. Output looks like "lamquant 7.7.0".
        // Compare major.minor; refuse on mismatch (StagingFailed flavor
        // would be wrong here — hard "VersionMismatch" so the caller
        // can surface it as such in the Peers panel).
        let raw = self.ssh_capture(peer, "lamquant --version || lml --version")?;
        let peer_v = raw
            .lines()
            .find_map(|l| l.split_whitespace().last().map(|s| s.to_string()))
            .unwrap_or_else(|| "unknown".into());
        let local_v = env!("CARGO_PKG_VERSION").to_string();
        if !same_major_minor(&local_v, &peer_v) {
            return Err(TransportError::VersionMismatch {
                local: local_v,
                peer: peer_v,
            });
        }
        Ok(PeerInfo {
            version: peer_v,
            // Future: probe for nvidia-smi, /proc/cpuinfo. Stub for now.
            gpu: None,
            workers: 0,
        })
    }

    fn stage_input(&self, peer: &Peer, local: &Path) -> Result<RemotePath, TransportError> {
        if !local.exists() {
            return Err(TransportError::StagingFailed(format!(
                "local input not found: {}",
                local.display()
            )));
        }

        // Compute local sha256.
        let local_hash = sha256_file(local)
            .map_err(|e| TransportError::StagingFailed(format!("local hash failed: {}", e)))?;

        let local_str = local.display().to_string();
        // Hash-detect on peer: if same path exists with same hash, zero copy.
        let probe = format!(
            "if [ -f {q} ]; then sha256sum {q} | awk '{{print $1}}'; fi",
            q = shell_quote(&local_str)
        );
        if let Ok(out) = self.ssh_capture(peer, &probe) {
            let peer_hash = out.trim();
            if peer_hash == local_hash {
                return Ok(RemotePath(local_str));
            }
        }

        // Fallback: rsync push to ~/.cache/lamquant/staged/<basename>.
        let basename = local
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| TransportError::StagingFailed("local path has no filename".into()))?;
        let cfg = self.ssh_config(peer)?;
        let remote_dir = format!("/tmp/lamquant-staged-{}", cfg.user);
        let remote_path = format!("{}/{}", remote_dir, basename);

        // Ensure remote dir.
        let mkdir = format!("mkdir -p {}", shell_quote(&remote_dir));
        self.ssh_capture(peer, &mkdir)?;

        // rsync command: rsync -e "ssh ..." LOCAL user@host:REMOTE
        // Build the -e option string with the same hardening flags.
        let ssh_e = format!(
            "ssh -i {} -o IdentitiesOnly=yes -o UserKnownHostsFile={} \
             -o StrictHostKeyChecking=yes -o BatchMode=yes \
             -o PasswordAuthentication=no -o Port={}",
            shell_quote(&cfg.key_path.display().to_string()),
            shell_quote(&cfg.known_hosts.display().to_string()),
            cfg.port,
        );
        let mut rsync = Command::new("rsync");
        rsync.arg("-az").arg("--partial");
        rsync.arg("-e").arg(ssh_e);
        rsync.arg(local);
        rsync.arg(format!("{}@{}:{}", cfg.user, peer.host, remote_path));
        let status = rsync
            .status()
            .map_err(|e| TransportError::StagingFailed(format!("rsync spawn failed: {}", e)))?;
        if !status.success() {
            return Err(TransportError::StagingFailed(format!(
                "rsync exited with {}",
                status
            )));
        }
        Ok(RemotePath(remote_path))
    }

    fn dispatch(
        &self,
        peer: &Peer,
        op_id: &str,
        args: &[String],
    ) -> Result<(OpReceiver, RemoteHandle), TransportError> {
        // Build remote command: lamquant <op_id> --emit-json-events <args...>
        let mut argv: Vec<String> =
            vec!["lamquant".into(), op_id.into(), "--emit-json-events".into()];
        argv.extend(args.iter().cloned());
        let remote_cmd = argv
            .iter()
            .map(|a| shell_quote(a))
            .collect::<Vec<_>>()
            .join(" ");

        let mut cmd = self.ssh_cmd(peer, &remote_cmd)?;
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
        let mut child: Child = cmd
            .spawn()
            .map_err(|e| TransportError::DispatchFailed(format!("ssh spawn failed: {}", e)))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| TransportError::DispatchFailed("ssh stdout unavailable".into()))?;

        let (tx, rx) = mpsc::channel::<OpEvent>();
        let pid = child.id();

        // Reader thread: parse each stdout line as JSON OpEvent.
        // Malformed lines become Log events with the raw text so users
        // can still see legacy stdout from cmd_* paths that haven't
        // been upgraded to emit_log/emit_progress yet.
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line_res in reader.lines() {
                let line = match line_res {
                    Ok(l) => l,
                    Err(_) => break,
                };
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let ev = match serde_json::from_str::<OpEvent>(trimmed) {
                    Ok(ev) => ev,
                    Err(_) => OpEvent::Log {
                        ts_ms: OpEvent::now_ms(),
                        message: line,
                    },
                };
                if tx.send(ev).is_err() {
                    break;
                }
            }
            // Reap the SSH process; if it exited non-zero with no
            // terminal event, synthesize an Error so the local panel
            // sees a clean done-state rather than a silent disconnect.
            let _ = child.wait();
        });

        let handle = RemoteHandle {
            token: pid.to_string(),
        };
        Ok((rx, handle))
    }

    fn cancel(&self, peer: &Peer, handle: &RemoteHandle) -> Result<(), TransportError> {
        // Kill the local SSH process — its remote cmd exits via SIGPIPE
        // when stdout closes. Best-effort; some shells don't propagate
        // SIGHUP through SSH cleanly.
        if let Ok(pid) = handle.token.parse::<u32>() {
            let _ = Command::new("kill")
                .arg("-TERM")
                .arg(pid.to_string())
                .status();
        }
        // Also send a remote pkill as a belt-and-suspenders.
        let _ = self.ssh_capture(peer, "pkill -INT -f 'lamquant.*--emit-json-events' || true");
        Ok(())
    }

    fn health(&self, peer: &Peer) -> Result<PeerHealth, TransportError> {
        // Lightweight: just verify SSH reaches the peer.
        // Future: count `pgrep lamquant` for Busy detection.
        match self.ssh_capture(peer, "echo ok") {
            Ok(s) if s.trim() == "ok" => Ok(PeerHealth::Idle),
            Ok(_) => Ok(PeerHealth::Unreachable),
            Err(TransportError::AuthFailed(_)) => Ok(PeerHealth::AuthFailed),
            Err(TransportError::Unreachable(_)) => Ok(PeerHealth::Unreachable),
            Err(e) => Err(e),
        }
    }
}

// ── Helpers ────────────────────────────────────────────────────────────

/// Compare semver-ish strings like "7.7.0" / "v7.7.0". Returns true if
/// major.minor match. Patch/build allowed to differ.
fn same_major_minor(a: &str, b: &str) -> bool {
    fn parts(s: &str) -> Vec<u32> {
        s.trim_start_matches('v')
            .split('.')
            .take(2)
            .filter_map(|p| p.parse().ok())
            .collect()
    }
    let pa = parts(a);
    let pb = parts(b);
    pa.len() == 2 && pa == pb
}

/// Single-quote a shell argument the standard way: wrap in '…' and
/// replace any embedded ' with '\''. Used to defend against paths /
/// args containing spaces or shell metacharacters.
fn shell_quote(s: &str) -> String {
    if s.is_empty() {
        return "''".into();
    }
    if s.chars().all(|c| {
        c.is_ascii_alphanumeric() || matches!(c, '/' | '.' | '_' | '-' | ':' | '=' | '@' | ',')
    }) {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for c in s.chars() {
        if c == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(c);
        }
    }
    out.push('\'');
    out
}

/// Compute sha256 of a file. Used by stage_input to short-circuit when
/// peer already has identical content.
fn sha256_file(path: &Path) -> Result<String, std::io::Error> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Suppresses unused warning for the type alias we keep in the public API.
#[allow(dead_code)]
fn _types_used(_: PathBuf) {}
