//! Socket auth (feature `auth`, default-off) — the hardening ADR 0136 adds to
//! the daemon's Unix control socket.
//!
//! Two layers, smallest-sufficient-first:
//!
//! 1. **`SO_PEERCRED` uid gate (the active control).** A Unix socket exposes the
//!    connecting process's credentials to the kernel; the daemon refuses any
//!    peer whose uid is not its own. This is the right primary gate for a local
//!    single-host clinical daemon — kernel-enforced, no key management, and it
//!    cannot be spoofed by a process running as another user. Wired into the
//!    control accept loop (`daemon.rs`) behind `#[cfg(feature = "auth")]`.
//!
//! 2. **Ed25519 capability token (the seam for non-local attach).** A standalone
//!    sign/verify primitive (mirroring only the Ed25519 identity half of
//!    `training/engine/src/p2p/crypto.rs`, NOT its X25519/AES payload sealing,
//!    which encrypts data a local socket does not need). Kept dependency-clean:
//!    a small token here, not a cross-workspace dep on the p2p crate. It is the
//!    capability a console would present when the socket is ever forwarded off
//!    the host; the local path relies on the uid gate.
//!
//! When the `auth` feature is OFF, none of this compiles in and the control
//! protocol is byte-identical to ADR 0135.

use std::io;

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};

/// An Ed25519 keypair used to mint/verify control-capability tokens. The secret
/// never leaves the process except via [`AuthKey::secret_bytes`] (for durable
/// storage by the operator).
pub struct AuthKey {
    signing: SigningKey,
}

impl AuthKey {
    /// Generate a fresh keypair from OS entropy.
    pub fn generate() -> io::Result<Self> {
        let mut secret = [0u8; 32];
        getrandom::getrandom(&mut secret)
            .map_err(|e| io::Error::other(format!("getrandom: {e}")))?;
        Ok(Self {
            signing: SigningKey::from_bytes(&secret),
        })
    }

    /// Reconstruct from a stored 32-byte secret seed.
    pub fn from_secret_bytes(seed: &[u8; 32]) -> Self {
        Self {
            signing: SigningKey::from_bytes(seed),
        }
    }

    /// The 32-byte secret seed (for the operator to persist securely).
    pub fn secret_bytes(&self) -> [u8; 32] {
        self.signing.to_bytes()
    }

    /// The 32-byte public verifying key (handed to a verifier / the daemon).
    pub fn verifying_key_bytes(&self) -> [u8; 32] {
        self.signing.verifying_key().to_bytes()
    }

    /// Sign a control challenge, returning the 64-byte detached signature.
    pub fn sign(&self, msg: &[u8]) -> [u8; 64] {
        self.signing.sign(msg).to_bytes()
    }
}

/// Verify a detached Ed25519 signature over `msg` against a public key. Returns
/// `false` on a malformed key/signature or a bad signature — fail-closed.
pub fn verify(verifying_key: &[u8; 32], msg: &[u8], signature: &[u8; 64]) -> bool {
    let Ok(vk) = VerifyingKey::from_bytes(verifying_key) else {
        return false;
    };
    let sig = Signature::from_bytes(signature);
    vk.verify(msg, &sig).is_ok()
}

/// The uid this process runs as. Control connections from any other uid are
/// refused by [`authorize_peer_uid`].
pub fn own_uid() -> u32 {
    // getuid() is always successful and has no failure mode (POSIX).
    unsafe { libc::getuid() }
}

/// The `SO_PEERCRED` gate: authorize a control peer by its kernel-reported uid.
/// Only the daemon's own uid may drive it. `Ok(uid)` allowed, `Err(uid)` refused.
pub fn authorize_peer_uid(peer_uid: u32) -> Result<u32, u32> {
    if peer_uid == own_uid() {
        Ok(peer_uid)
    } else {
        Err(peer_uid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_verify_roundtrips() {
        let key = AuthKey::generate().unwrap();
        let vk = key.verifying_key_bytes();
        let msg = b"lamquant-runtime control challenge v1";
        let sig = key.sign(msg);
        assert!(verify(&vk, msg, &sig), "valid signature must verify");
    }

    #[test]
    fn verify_rejects_tamper() {
        let key = AuthKey::generate().unwrap();
        let vk = key.verifying_key_bytes();
        let sig = key.sign(b"original message");
        assert!(
            !verify(&vk, b"tampered message", &sig),
            "wrong message must fail"
        );
        // A different key must not verify the signature.
        let other = AuthKey::generate().unwrap();
        assert!(!verify(
            &other.verifying_key_bytes(),
            b"original message",
            &sig
        ));
    }

    #[test]
    fn secret_roundtrips_to_same_public_key() {
        let key = AuthKey::generate().unwrap();
        let restored = AuthKey::from_secret_bytes(&key.secret_bytes());
        assert_eq!(key.verifying_key_bytes(), restored.verifying_key_bytes());
    }

    #[test]
    fn peer_uid_gate_allows_self_rejects_others() {
        let me = own_uid();
        assert_eq!(authorize_peer_uid(me), Ok(me));
        assert_eq!(
            authorize_peer_uid(me.wrapping_add(1)),
            Err(me.wrapping_add(1))
        );
    }
}
