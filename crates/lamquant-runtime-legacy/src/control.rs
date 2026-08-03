//! The daemon control protocol (feature `daemon`) — length-prefixed JSON frames
//! over a Unix-domain socket. A console prefers this live channel and falls back
//! to tailing `status.jsonl` when the socket is absent (ADR 0135 hybrid control).

use std::io::{Error, ErrorKind};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::status::RuntimeStatus;

pub const PROTOCOL_VERSION: u32 = 1;
const MAX_FRAME: usize = 16 * 1024 * 1024;

/// A request from a controller to the daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ControlRequest {
    /// Liveness + version handshake.
    Ping,
    /// The latest per-pipeline status snapshot.
    Status,
    /// Request a graceful shutdown (drain + finalize).
    Stop,
}

/// The daemon's reply.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ControlResponse {
    Pong { version: u32 },
    Status { pipelines: Vec<RuntimeStatus> },
    Stopping,
    Error { message: String },
}

fn json_err(e: serde_json::Error) -> Error {
    Error::new(ErrorKind::InvalidData, e)
}

/// Write one length-prefixed (`u32` LE) JSON frame.
pub async fn write_frame<W, T>(w: &mut W, msg: &T) -> std::io::Result<()>
where
    W: AsyncWriteExt + Unpin,
    T: Serialize,
{
    let bytes = serde_json::to_vec(msg).map_err(json_err)?;
    let len = u32::try_from(bytes.len())
        .map_err(|_| Error::new(ErrorKind::InvalidData, "control frame too large"))?;
    w.write_all(&len.to_le_bytes()).await?;
    w.write_all(&bytes).await?;
    w.flush().await?;
    Ok(())
}

/// Read one length-prefixed JSON frame.
pub async fn read_frame<R, T>(r: &mut R) -> std::io::Result<T>
where
    R: AsyncReadExt + Unpin,
    T: DeserializeOwned,
{
    let mut len_bytes = [0u8; 4];
    r.read_exact(&mut len_bytes).await?;
    let len = u32::from_le_bytes(len_bytes) as usize;
    if len > MAX_FRAME {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "control frame exceeds max",
        ));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).await?;
    serde_json::from_slice(&buf).map_err(json_err)
}
