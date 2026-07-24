// SPDX-License-Identifier: AGPL-3.0-or-later
//! ADR 0139 P3 compute slice: cross-backend byte equality.
//!
//! Encodes one deterministic conformance vector with `ComputeBackend::Firmware`
//! and again with `ComputeBackend::Desktop`, then prints the SHA-256 of each
//! encoded `.lml` payload as a single deterministic JSON line.
//!
//! The wire format is backend-independent by contract, so the two digests MUST
//! match; any optimisation that changes them is a wire-format change, not a
//! speedup. Reporting both digests (rather than a boolean) keeps the divergence
//! visible to the caller instead of hidden behind an assertion here.

use lamquant_lml_desktop::backend::{compress_with_backend, ComputeBackend};
use lamquant_lml_desktop::lpc::LpcMode;
use sha2::{Digest, Sha256};

/// xorshift64 — deterministic across machines and architectures.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Mirrors the codec's conformance signal generator so this evidence covers the
/// same shapes as the in-repo byte-equality gate.
fn synth_signal(n_ch: usize, t: usize, seed: u64) -> Vec<Vec<i64>> {
    (0..n_ch)
        .map(|channel| {
            let mut state = seed.wrapping_add((channel as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut samples = Vec::with_capacity(t);
            let mut previous: i64 = 0;
            for _ in 0..t {
                let step = (xorshift64(&mut state) as i32 >> 24) as i64;
                previous = (previous + step).clamp(-2000, 2000);
                samples.push(previous);
            }
            samples
        })
        .collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

struct Vector {
    channels: usize,
    samples: usize,
    seed: u64,
    noise_bits: u8,
}

fn vector(name: &str) -> Option<Vector> {
    match name {
        "1ch_100" => Some(Vector {
            channels: 1,
            samples: 100,
            seed: 0x1111_2222_3333_4444,
            noise_bits: 0,
        }),
        "4ch_2500" => Some(Vector {
            channels: 4,
            samples: 2500,
            seed: 0xDEAD_BEEF_CAFE_BABE,
            noise_bits: 0,
        }),
        "32ch_2500" => Some(Vector {
            channels: 32,
            samples: 2500,
            seed: 0xCAFE_BABE_F00D_BEEF,
            noise_bits: 0,
        }),
        _ => None,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let name = arguments.next().ok_or("missing vector name")?;
    if arguments.next().is_some() {
        return Err("unexpected extra argument".into());
    }
    let spec = vector(&name).ok_or_else(|| format!("unknown vector: {name}"))?;
    let signal = synth_signal(spec.channels, spec.samples, spec.seed);
    let mode = LpcMode::Anytime {
        max_order: 16,
        deadline: None,
    };

    let firmware = compress_with_backend(&signal, spec.noise_bits, mode, ComputeBackend::Firmware)?;
    let desktop = compress_with_backend(&signal, spec.noise_bits, mode, ComputeBackend::Desktop)?;
    println!(
        concat!(
            "{{\"vector\":\"{}\",\"channels\":{},\"samples\":{},",
            "\"firmware_sha256\":\"{}\",\"desktop_sha256\":\"{}\",",
            "\"firmware_bytes\":{},\"desktop_bytes\":{}}}"
        ),
        name,
        spec.channels,
        spec.samples,
        sha256_hex(&firmware),
        sha256_hex(&desktop),
        firmware.len(),
        desktop.len(),
    );
    Ok(())
}
