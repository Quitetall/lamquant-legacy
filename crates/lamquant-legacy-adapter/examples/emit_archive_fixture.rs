// SPDX-License-Identifier: AGPL-3.0-or-later
//! Write one real retired LMA archive to disk.
//!
//! The ADR 0144 input producer needs an actual retired archive to derive from,
//! and nothing else in this repository writes the v2 layout with a non-signal
//! sibling. Framed here from the specification, exactly as the evidence probe
//! frames its own fixtures.

use lamquant_legacy_ir::{Bcs1Header, BCS1_VERSION_MAJOR, BCS1_VERSION_MINOR, CODEC_LML_53};
use sha2::{Digest, Sha256};

fn lml1(signal: &[Vec<i64>], rate_mhz: u32) -> Vec<u8> {
    let payload = lamquant_lml_mcu::lml::compress(signal, 0).expect("fixture compresses");
    let metadata = br#"{"channels":["Fp1","Fp2"]}"#;
    let samples = signal.first().map_or(0, Vec::len) as u32;
    let header = Bcs1Header {
        version_major: BCS1_VERSION_MAJOR,
        version_minor: BCS1_VERSION_MINOR,
        modality_tag: 0,
        modality_source: 0,
        codec_descriptor: CODEC_LML_53,
        mode: 0,
        tier: 0,
        decode_capability: 0,
        n_channels: signal.len() as u16,
        n_windows: 1,
        total_samples: samples,
        window_size: samples as u16,
        sample_rate_mhz: rate_mhz,
        bit_depth: 16,
        flags: 0,
        metadata_length: metadata.len() as u32,
    };
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"LML1");
    bytes.extend_from_slice(&1_u16.to_le_bytes());
    bytes.extend_from_slice(&header.n_channels.to_le_bytes());
    bytes.extend_from_slice(&header.n_windows.to_le_bytes());
    bytes.extend_from_slice(&header.total_samples.to_le_bytes());
    bytes.extend_from_slice(&header.window_size.to_le_bytes());
    bytes.extend_from_slice(&header.sample_rate_mhz.to_le_bytes());
    bytes.push(header.bit_depth);
    bytes.push(header.flags);
    bytes.extend_from_slice(&header.metadata_length.to_le_bytes());
    bytes.extend_from_slice(&[0_u8; 6]);
    bytes.extend_from_slice(metadata);
    bytes.extend_from_slice(&0_u32.to_le_bytes());
    bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&payload);
    bytes
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output = std::env::args()
        .nth(1)
        .ok_or("usage: emit_archive_fixture <path>")?;
    let entries: Vec<(String, Vec<u8>)> = vec![
        (
            "recording-a.lml".to_owned(),
            lml1(
                &[vec![-9, -1, 0, 7, 11], vec![100, 101, 99, 102, 98]],
                250_000,
            ),
        ),
        (
            "recording-b.lml".to_owned(),
            lml1(&[vec![5, 6, 7, 8, 9], vec![-5, -6, -7, -8, -9]], 256_000),
        ),
    ];
    let sidecar: (&str, &[u8]) = ("annotations.tse", b"0.0 1.0 bckg 1.0000\n");

    let mut files = String::from("{\"compressor\":\"zstd\",\"compressor_level\":3,\"files\":[");
    let mut payloads: Vec<u8> = Vec::new();
    let listed: Vec<(&str, &[u8], &str)> = entries
        .iter()
        .map(|(p, b)| (p.as_str(), b.as_slice(), "lml"))
        .chain(std::iter::once((sidecar.0, sidecar.1, "store")))
        .collect();
    for (index, (path, bytes, method)) in listed.iter().enumerate() {
        if index > 0 {
            files.push(',');
        }
        files.push_str(&format!(
            "{{\"path\":\"{path}\",\"original_size\":{size},\"compressed_size\":{size},\
             \"method\":\"{method}\",\"sha256\":\"{digest:x}\",\"offset\":{offset}}}",
            size = bytes.len(),
            digest = Sha256::digest(bytes),
            offset = payloads.len(),
        ));
        payloads.extend_from_slice(bytes);
    }
    files.push_str("]}");
    let manifest = files.into_bytes();

    let mut archive = Vec::new();
    archive.extend_from_slice(b"LMA2");
    archive.extend_from_slice(&2_u32.to_le_bytes());
    archive.extend_from_slice(&[0_u8; 8]);
    archive.extend_from_slice(&payloads);
    archive.extend_from_slice(&manifest);
    archive.extend_from_slice(&((manifest.len() as u32) | 0x8000_0000).to_le_bytes());
    archive.extend_from_slice(&((entries.len() + 1) as u32).to_le_bytes());
    archive.extend_from_slice(b"LFT2");
    let digest = Sha256::digest(&archive);
    archive.extend_from_slice(&digest);
    std::fs::write(&output, &archive)?;
    println!("{{\"archive\":\"{output}\",\"bytes\":{}}}", archive.len());
    Ok(())
}
