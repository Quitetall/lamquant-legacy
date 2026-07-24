// SPDX-License-Identifier: AGPL-3.0-or-later
//! ADR 0143 retired-wire evidence probe.
//!
//! Emits ONE `lamquant.adr0143-evidence/v1` receipt as JSON on stdout for a
//! (profile, role) pair. Every assertion in the receipt is *measured* by
//! driving the real adapter entry points (`import_semantic`) against a
//! fixture built in this file — nothing is asserted from a literal.
//!
//! Usage:
//!     legacy_evidence_probe <profile> <role>
//!
//! The implementation revision is supplied by the caller through
//! `LAMQUANT_LEGACY_REVISION` (40 hex characters) so the receipt binds to the
//! commit under test without the probe shelling out to git — the probe stays a
//! pure function of (profile, role, revision) and re-runs byte-identically.
//!
//! Roles map onto ADR 0143's required legacy evidence:
//!
//! * `positive` — a valid retired container imports.
//! * `malformed` — every corrupted variant is refused before output.
//! * `source-preservation` — the source file is byte-identical after import.
//! * `mapping-report` — the emitted mapping report validates and names the
//!   disposition of every source region.
//! * `fidelity-report` — the fidelity report states the semantic loss
//!   explicitly rather than claiming equivalence.
//! * `lazy-no-overwrite` — a conflicting destination is rejected and the source
//!   is still untouched.

use std::collections::BTreeMap;
use std::path::Path;

use lamquant_legacy_adapter::{import_semantic, LegacyError, SemanticImportRequest};
use lamquant_legacy_ir::{Bcs1Header, BCS1_VERSION_MAJOR, BCS1_VERSION_MINOR, CODEC_LML_53};
use serde::Serialize;
use sha2::{Digest, Sha256};

const SCHEMA: &str = "lamquant.adr0143-evidence/v1";
const PRODUCER_NAME: &str = "lamquant-legacy-evidence-probe";
const PRODUCER_VERSION: &str = "1";
const PRODUCER_AUTHORITY: &str = "implementation-test";
/// This probe hashes its own source so the receipt binds to the exact
/// measurement code, not merely to the repository revision.
const PROBE_SOURCE: &[u8] = include_bytes!("legacy_evidence_probe.rs");
const MAX_BYTES: u64 = 1 << 20;

/// Fields are declared in the receipt's canonical (alphabetical) order;
/// serde emits declaration order, so the rendering is stable regardless of
/// which serde_json map feature the dependency graph selects.
#[derive(Serialize)]
struct Receipt {
    assertions: BTreeMap<String, Assertion>,
    command: Vec<String>,
    implementation_revision: String,
    producer: Producer,
    profile: String,
    role: String,
    schema: String,
    status: String,
}

#[derive(Serialize)]
#[serde(untagged)]
enum Assertion {
    Count(u64),
    Flag(bool),
    Text(String),
}

#[derive(Serialize)]
struct Producer {
    authority: String,
    executable_sha256: String,
    name: String,
    version: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

// ---- fixtures ---------------------------------------------------------

fn lml1_container(signal: &[Vec<i64>], sample_rate_mhz: u32) -> Vec<u8> {
    let payload = lamquant_lml_mcu::lml::compress(signal, 0).expect("fixture signal compresses");
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
        sample_rate_mhz,
        bit_depth: 16,
        flags: 0,
        metadata_length: metadata.len() as u32,
    };
    // Re-frame the header as the LML1 header the retired wire carried: the
    // payload section after the fixed header is byte-identical either way.
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

/// The two archived recordings. Distinct signals, because content addressing
/// deduplicates identical payloads into one ContentId.
fn archive_entries() -> Vec<(String, Vec<u8>)> {
    vec![
        (
            "recording-a.lml".to_owned(),
            lml1_container(
                &[vec![-9, -1, 0, 7, 11], vec![100, 101, 99, 102, 98]],
                250_000,
            ),
        ),
        (
            "recording-b.lml".to_owned(),
            lml1_container(&[vec![5, 6, 7, 8, 9], vec![-5, -6, -7, -8, -9]], 256_000),
        ),
    ]
}

/// A non-signal sibling: it must survive byte-exact in the source capsule and
/// must never be promoted to a recording.
const SIDECAR: (&str, &[u8]) = ("annotations.tse", b"0.0 1.0 bckg 1.0000\n");

/// The manifest + concatenated payload section both retired generations share.
///
/// Signal entries carry `method: "lml"`; the sidecar carries `method: "store"`
/// so the importer sees a genuine non-signal sibling to quarantine. The
/// codec's own `pack_lml_entries` cannot build this fixture — it marks every
/// entry as LML by construction — so the retired layout is framed here.
fn manifest_and_payloads(
    entries: &[(String, Vec<u8>)],
    sidecar: (&str, &[u8]),
) -> (Vec<u8>, Vec<u8>) {
    let mut files = String::from("{\"compressor\":\"zstd\",\"compressor_level\":3,\"files\":[");
    let mut payloads: Vec<u8> = Vec::new();
    let listed: Vec<(&str, &[u8], &str)> = entries
        .iter()
        .map(|(path, bytes)| (path.as_str(), bytes.as_slice(), "lml"))
        .chain(std::iter::once((sidecar.0, sidecar.1, "store")))
        .collect();
    for (index, (path, bytes, method)) in listed.iter().enumerate() {
        if index > 0 {
            files.push(',');
        }
        files.push_str(&format!(
            "{{\"path\":\"{path}\",\"original_size\":{size},\"compressed_size\":{size},\
             \"method\":\"{method}\",\"sha256\":\"{digest}\",\"offset\":{offset}}}",
            size = bytes.len(),
            digest = sha256_hex(bytes),
            offset = payloads.len(),
        ));
        payloads.extend_from_slice(bytes);
    }
    files.push_str("]}");
    (files.into_bytes(), payloads)
}

/// Frame the v1 (`LMA1`, front-manifest) layout by hand.
///
/// Nothing writes v1 any more — the codec emits v2 — so the retired generation
/// only exists as a fixture. The manifest is stored uncompressed via the
/// length field's top bit, which both generations' reader honours, so the
/// fixture needs no compressor.
fn lma_v1_archive(entries: &[(String, Vec<u8>)], sidecar: (&str, &[u8])) -> Vec<u8> {
    let (manifest, payloads) = manifest_and_payloads(entries, sidecar);
    let mut archive = Vec::new();
    archive.extend_from_slice(b"LMA1");
    archive.extend_from_slice(&1_u32.to_le_bytes());
    archive.extend_from_slice(&((entries.len() + 1) as u32).to_le_bytes());
    // Top bit marks the manifest as stored rather than zstd.
    archive.extend_from_slice(&((manifest.len() as u32) | 0x8000_0000).to_le_bytes());
    archive.extend_from_slice(&manifest);
    archive.extend_from_slice(&payloads);
    let digest = Sha256::digest(&archive);
    archive.extend_from_slice(&digest);
    archive
}

/// Frame the v2 (`LMA2`, trailing footer/EOCD) layout by hand — same manifest
/// schema, positioned at the end so payloads stream in one forward pass.
fn lma_v2_archive(entries: &[(String, Vec<u8>)], sidecar: (&str, &[u8])) -> Vec<u8> {
    let (manifest, payloads) = manifest_and_payloads(entries, sidecar);
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
    archive
}

/// The LMQC neural container: a real montage plus an opaque encoded latent.
fn lmqc_container() -> Vec<u8> {
    lamquant_lml_mcu::lmqc::encode_lmqc(
        2,
        32,
        79,
        250,
        2500,
        lamquant_lml_mcu::lmqc::PAYLOAD_FP16_LATENT,
        Some(&[0.1, 0.2, 0.3, -0.1, -0.2, -0.3]),
        Some(&["Fp1".to_owned(), "Fp2".to_owned()]),
        b"opaque encoded latent bytes",
    )
    .expect("fixture montage encodes")
}

/// A fixed AEAD key for the envelope fixture. It protects nothing real: the
/// plaintext is a synthetic five-sample recording built in this file.
const FIXTURE_KEY: [u8; 32] = [0x2b; 32];

fn seal(plaintext: &[u8], nonce: &[u8; 12]) -> Vec<u8> {
    use aes_gcm::aead::{Aead, KeyInit};
    let cipher = aes_gcm::Aes256Gcm::new_from_slice(&FIXTURE_KEY).expect("32-byte key");
    let ciphertext = cipher
        .encrypt(aes_gcm::Nonce::from_slice(nonce), plaintext)
        .expect("fixture plaintext seals");
    let mut blob = Vec::new();
    blob.extend_from_slice(b"LMLCRYPT");
    blob.push(1);
    blob.extend_from_slice(nonce);
    blob.extend_from_slice(&ciphertext);
    blob
}

/// A retired multi-view training snapshot. Two views so the fixture exercises
/// both an f32 view (decoded) and a u8 view (carried stored), and `seed` varies
/// the label bytes so a second snapshot is genuinely a different file.
fn tensor_pack_v2(seed: u8) -> Vec<u8> {
    use lamquant_lml_legacy::tensor_pack_v2::{
        PackV2Dtype, PackV2Encoding, PackV2Writer, ViewSpec,
    };
    let scratch = tempfile::tempdir().expect("scratch directory");
    let path = scratch.path().join("snapshot.lqtp2");
    let specs = vec![
        ViewSpec::new(
            "fullband",
            PackV2Dtype::F32,
            PackV2Encoding::BfpInt16,
            &[2, 3],
            true,
            [0x11; 32],
        )
        .expect("f32 view spec"),
        ViewSpec::new(
            "labels",
            PackV2Dtype::U8,
            PackV2Encoding::Raw,
            &[4],
            true,
            [0x22; 32],
        )
        .expect("u8 view spec"),
    ];
    let mut writer = PackV2Writer::create(
        &path,
        2,
        [0xaa; 32],
        [0xbb; 32],
        br#"{"schema":"lamquant.training-window-metadata/1"}"#.to_vec(),
        specs,
    )
    .expect("snapshot writer");
    for row in 0..2_u8 {
        let offset = f32::from(row) * 10.0;
        writer
            .write_f32_row(
                "fullband",
                &[
                    1.0 + offset,
                    -2.0 - offset,
                    3.0 + offset,
                    4.0 + offset,
                    -5.0 - offset,
                    6.0 + offset,
                ],
            )
            .expect("f32 row");
        writer
            .write_raw_row("labels", &[seed + row, row + 1, row + 2, row + 3])
            .expect("raw row");
    }
    writer.finish().expect("snapshot publishes");
    std::fs::read(&path).expect("snapshot reads back")
}

/// The chunked generation: one zstd-compressed view and one uncompressed one,
/// so both chunk codecs are on the evidence path.
fn tensor_pack_v3(seed: u8) -> Vec<u8> {
    use lamquant_lml_legacy::tensor_pack_v3::{
        PackV3Compression, PackV3Dtype, PackV3Encoding, PackV3Writer, ViewSpecV3,
    };
    let scratch = tempfile::tempdir().expect("scratch directory");
    let path = scratch.path().join("snapshot.lqtp3");
    let specs = vec![
        ViewSpecV3::new(
            "fullband",
            PackV3Dtype::F32,
            PackV3Encoding::BfpInt16,
            &[2, 3],
            true,
            [0x11; 32],
            2,
            PackV3Compression::Zstd,
            1,
        )
        .expect("f32 view spec"),
        ViewSpecV3::new(
            "labels",
            PackV3Dtype::U8,
            PackV3Encoding::Raw,
            &[4],
            true,
            [0x22; 32],
            3,
            PackV3Compression::None,
            0,
        )
        .expect("u8 view spec"),
    ];
    let mut writer = PackV3Writer::create(
        &path,
        4,
        [0xaa; 32],
        [0xbb; 32],
        br#"{"schema":"lamquant.training-window-metadata/1"}"#.to_vec(),
        specs,
    )
    .expect("snapshot writer");
    for row in 0..4_u8 {
        let offset = f32::from(row) * 10.0;
        writer
            .write_f32_row(
                "fullband",
                &[
                    1.0 + offset,
                    -2.0 - offset,
                    3.0 + offset,
                    4.0 + offset,
                    -5.0 - offset,
                    6.0 + offset,
                ],
            )
            .expect("f32 row");
        writer
            .write_raw_row("labels", &[seed + row, row + 1, row + 2, row + 3])
            .expect("raw row");
    }
    writer.finish().expect("snapshot publishes");
    std::fs::read(&path).expect("snapshot reads back")
}

/// Every corruption a retired container can arrive with. Each must be refused
/// before any output is produced.
fn malformed_variants(valid: &[u8]) -> Vec<(String, Vec<u8>)> {
    let mut cases = Vec::new();

    let mut wrong_magic = valid.to_vec();
    wrong_magic[0..4].copy_from_slice(b"XXXX");
    cases.push(("unknown-magic".to_owned(), wrong_magic));

    cases.push((
        "truncated-body".to_owned(),
        valid[..valid.len() / 2].to_vec(),
    ));

    let mut short = valid[..16.min(valid.len())].to_vec();
    short.truncate(16);
    cases.push(("header-only".to_owned(), short));

    let mut flipped = valid.to_vec();
    // Corrupt one interior byte. For an archive the manifest must not silently
    // drop entries; for a CRC- or AEAD-protected container the integrity check
    // must fire rather than the corruption being carried into the dataset.
    let target = flipped.len() / 3;
    flipped[target] ^= 0xff;
    cases.push(("interior-bit-flip".to_owned(), flipped));

    let mut tail = valid.to_vec();
    let last = tail.len() - 1;
    tail[last] ^= 0xff;
    cases.push(("trailing-integrity-flip".to_owned(), tail));

    cases
}

// ---- profiles ---------------------------------------------------------

struct Profile {
    id: &'static str,
    build: fn() -> Vec<u8>,
}

fn build_lma_v1() -> Vec<u8> {
    lma_v1_archive(&archive_entries(), SIDECAR)
}

fn build_lma_v2() -> Vec<u8> {
    lma_v2_archive(&archive_entries(), SIDECAR)
}

fn build_lmqc() -> Vec<u8> {
    lmqc_container()
}

fn build_lmlcrypt() -> Vec<u8> {
    seal(&archive_entries().remove(0).1, &[7; 12])
}

fn build_lqtp2() -> Vec<u8> {
    tensor_pack_v2(0)
}

fn build_lqtp3() -> Vec<u8> {
    tensor_pack_v3(0)
}

const PROFILES: &[Profile] = &[
    Profile {
        id: "legacy.lma.v1",
        build: build_lma_v1,
    },
    Profile {
        id: "legacy.lma.v2",
        build: build_lma_v2,
    },
    Profile {
        id: "legacy.lmqc.v1",
        build: build_lmqc,
    },
    Profile {
        id: "legacy.lmlcrypt.v1",
        build: build_lmlcrypt,
    },
    Profile {
        id: "legacy.lqtp.v2",
        build: build_lqtp2,
    },
    Profile {
        id: "legacy.lqtp.v3",
        build: build_lqtp3,
    },
];

fn profile(id: &str) -> &'static Profile {
    PROFILES
        .iter()
        .find(|profile| profile.id == id)
        .unwrap_or_else(|| fail(&format!("unknown retired profile: {id}")))
}

fn fail(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(2);
}

// ---- measurement ------------------------------------------------------

fn stage(root: &Path, name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path = root.join(name);
    std::fs::write(&path, bytes).expect("fixture stages to disk");
    path
}

fn import(source: &Path, destination: &Path) -> Result<(), LegacyError> {
    import_semantic(&SemanticImportRequest {
        source: source.to_path_buf(),
        destination: destination.to_path_buf(),
        accept_fidelity: true,
        max_source_bytes: MAX_BYTES,
        max_decoded_bytes: MAX_BYTES,
    })
    .map(|_| ())
}

fn count(value: u64) -> Assertion {
    Assertion::Count(value)
}

fn flag(value: bool) -> Assertion {
    Assertion::Flag(value)
}

fn text(value: String) -> Assertion {
    Assertion::Text(value)
}

fn measure(profile: &Profile, role: &str) -> BTreeMap<String, Assertion> {
    let scratch = tempfile::tempdir().expect("scratch directory");
    let root = scratch.path();
    let archive = (profile.build)();
    let source = stage(root, "source.legacy", &archive);
    let before = sha256_hex(&std::fs::read(&source).expect("source reads"));
    let mut assertions = BTreeMap::new();

    match role {
        "positive" => {
            let receipt = import_semantic(&SemanticImportRequest {
                source: source.clone(),
                destination: root.join("semantic"),
                accept_fidelity: true,
                max_source_bytes: MAX_BYTES,
                max_decoded_bytes: MAX_BYTES,
            })
            .unwrap_or_else(|error| fail(&format!("valid fixture was refused: {error}")));
            assertions.insert("case_count".to_owned(), count(1));
            assertions.insert("accepted_cases".to_owned(), count(1));
            assertions.insert("imported_profile".to_owned(), text(receipt.profile.clone()));
            assertions.insert(
                "decoded_channels".to_owned(),
                count(receipt.decoded_channels),
            );
            assertions.insert(
                "exact_source_restoration".to_owned(),
                flag(receipt.exact_source_restoration),
            );
        }
        "malformed" => {
            let cases = malformed_variants(&archive);
            let mut rejected = 0_u64;
            for (index, (name, bytes)) in cases.iter().enumerate() {
                let path = stage(root, &format!("malformed-{index}.legacy"), bytes);
                let destination = root.join(format!("malformed-out-{index}"));
                match import(&path, &destination) {
                    Err(_) => rejected += 1,
                    Ok(()) => fail(&format!("malformed case {name} was accepted")),
                }
                if destination.exists() {
                    fail(&format!("malformed case {name} produced output"));
                }
            }
            assertions.insert("case_count".to_owned(), count(cases.len() as u64));
            assertions.insert("rejected_cases".to_owned(), count(rejected));
            assertions.insert("outputs_written".to_owned(), count(0));
        }
        "source-preservation" => {
            import(&source, &root.join("semantic"))
                .unwrap_or_else(|error| fail(&format!("valid fixture was refused: {error}")));
            let after = sha256_hex(&std::fs::read(&source).expect("source still reads"));
            assertions.insert("case_count".to_owned(), count(1));
            assertions.insert("source_preserved".to_owned(), flag(before == after));
            assertions.insert("source_before_sha256".to_owned(), text(before.clone()));
            assertions.insert("source_after_sha256".to_owned(), text(after));
        }
        "mapping-report" => {
            let destination = root.join("semantic");
            import(&source, &destination)
                .unwrap_or_else(|error| fail(&format!("valid fixture was refused: {error}")));
            let report: serde_json::Value = serde_json::from_slice(
                &std::fs::read(destination.join("mapping-report.json"))
                    .unwrap_or_else(|error| fail(&format!("mapping report is absent: {error}"))),
            )
            .unwrap_or_else(|error| fail(&format!("mapping report is not JSON: {error}")));
            let entries = report
                .get("entries")
                .and_then(serde_json::Value::as_array)
                .unwrap_or_else(|| fail("mapping report carries no entries"));
            // Every mapped region names a target and a disposition, and any
            // disposition other than `exact` states a reason -- an unexplained
            // loss is exactly what this role exists to refuse.
            let mut explained = 0_u64;
            for entry in entries {
                let disposition = entry
                    .get("disposition")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_else(|| fail("mapping entry has no disposition"));
                if entry
                    .get("target")
                    .and_then(serde_json::Value::as_str)
                    .is_none()
                {
                    fail("mapping entry has no target");
                }
                let reason = entry.get("reason").and_then(serde_json::Value::as_str);
                if disposition != "exact" && reason.is_none() {
                    fail("mapping entry lost data without stating a reason");
                }
                explained += 1;
            }
            let coverage = report
                .get("semantic_coverage")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_else(|| fail("mapping report declares no semantic coverage"));
            assertions.insert("case_count".to_owned(), count(1));
            assertions.insert("report_validated".to_owned(), flag(true));
            assertions.insert("mapped_entries".to_owned(), count(explained));
            assertions.insert("semantic_coverage".to_owned(), text(coverage.to_owned()));
        }
        "fidelity-report" => {
            let destination = root.join("semantic");
            import(&source, &destination)
                .unwrap_or_else(|error| fail(&format!("valid fixture was refused: {error}")));
            let report: serde_json::Value = serde_json::from_slice(
                &std::fs::read(destination.join("fidelity-report.json"))
                    .unwrap_or_else(|error| fail(&format!("fidelity report is absent: {error}"))),
            )
            .unwrap_or_else(|error| fail(&format!("fidelity report is not JSON: {error}")));
            let boolean = |name: &str| -> bool {
                report
                    .get(name)
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or_else(|| fail(&format!("fidelity report lacks {name}")))
            };
            let caveats = report
                .get("caveats")
                .and_then(serde_json::Value::as_array)
                .unwrap_or_else(|| fail("fidelity report carries no caveat list"));
            // Semantic loss is explicit when the report refuses to claim
            // equivalence AND says in words what was not carried across.
            let loss_explicit = !boolean("semantic_equivalence") && !caveats.is_empty();
            if !loss_explicit {
                fail("fidelity report claims semantic equivalence without evidence");
            }
            if report
                .get("source_capsule_file")
                .and_then(serde_json::Value::as_str)
                .is_none()
            {
                fail("fidelity report does not name the exact source capsule");
            }
            assertions.insert("case_count".to_owned(), count(1));
            assertions.insert("report_validated".to_owned(), flag(true));
            assertions.insert("semantic_loss_explicit".to_owned(), flag(loss_explicit));
            assertions.insert("caveat_count".to_owned(), count(caveats.len() as u64));
            assertions.insert(
                "exact_source_restoration".to_owned(),
                flag(boolean("exact_source_restoration")),
            );
        }
        "lazy-no-overwrite" => {
            let destination = root.join("semantic");
            import(&source, &destination)
                .unwrap_or_else(|error| fail(&format!("valid fixture was refused: {error}")));
            // A DIFFERENT retired source now targets the SAME destination.
            // The importer must refuse rather than overwrite the evidence
            // already there, and must leave both sources untouched.
            let other_entries = vec![archive_entries().remove(1)];
            let other = match profile.id {
                "legacy.lma.v1" => lma_v1_archive(&other_entries, SIDECAR),
                "legacy.lma.v2" => lma_v2_archive(&other_entries, SIDECAR),
                // A different latent yields a different container.
                "legacy.lmqc.v1" => {
                    let mut altered = lmqc_container();
                    let position = altered.len() - 8;
                    altered[position] ^= 0x01;
                    lamquant_lml_mcu::lmqc::encode_lmqc(
                        2,
                        32,
                        79,
                        250,
                        2500,
                        lamquant_lml_mcu::lmqc::PAYLOAD_FP16_LATENT,
                        Some(&[0.1, 0.2, 0.3, -0.1, -0.2, -0.3]),
                        Some(&["Fp1".to_owned(), "Fp2".to_owned()]),
                        b"a different encoded latent",
                    )
                    .expect("second fixture encodes")
                }
                "legacy.lqtp.v2" => tensor_pack_v2(64),
                "legacy.lqtp.v3" => tensor_pack_v3(64),
                // A different nonce over a different plaintext.
                _ => seal(&other_entries[0].1, &[9; 12]),
            };
            let other_path = stage(root, "other.legacy", &other);
            let error = import(&other_path, &destination)
                .expect_err("conflicting destination must be refused");
            let rejected = matches!(error, LegacyError::DestinationConflict);
            if !rejected {
                fail(&format!("conflict produced the wrong error: {error}"));
            }
            let after = sha256_hex(&std::fs::read(&source).expect("source still reads"));
            assertions.insert("case_count".to_owned(), count(1));
            assertions.insert("source_preserved".to_owned(), flag(before == after));
            assertions.insert(
                "conflicting_destination_rejected".to_owned(),
                flag(rejected),
            );
            assertions.insert("source_before_sha256".to_owned(), text(before.clone()));
            assertions.insert("source_after_sha256".to_owned(), text(after));
        }
        other => fail(&format!("unknown evidence role: {other}")),
    }
    assertions
}

fn main() {
    let mut arguments = std::env::args().skip(1);
    let profile_id = arguments
        .next()
        .unwrap_or_else(|| fail("usage: legacy_evidence_probe <profile> <role>"));
    let role = arguments
        .next()
        .unwrap_or_else(|| fail("usage: legacy_evidence_probe <profile> <role>"));
    if arguments.next().is_some() {
        fail("usage: legacy_evidence_probe <profile> <role>");
    }
    let revision = std::env::var("LAMQUANT_LEGACY_REVISION")
        .unwrap_or_else(|_| fail("LAMQUANT_LEGACY_REVISION must carry the 40-hex revision"));
    if revision.len() != 40 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        fail("LAMQUANT_LEGACY_REVISION must be 40 hexadecimal characters");
    }

    let selected = profile(&profile_id);
    if selected.id == "legacy.lmlcrypt.v1" {
        // The envelope profile reads its key from the documented environment
        // variable. Set the fixture key here so the receipt measures the
        // adapter, not the operator's shell.
        std::env::set_var(
            "LAMQUANT_KEY",
            FIXTURE_KEY
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>(),
        );
    }
    let assertions = measure(selected, &role);
    let receipt = Receipt {
        assertions,
        command: vec![
            "cargo".to_owned(),
            "run".to_owned(),
            "--quiet".to_owned(),
            "--manifest-path".to_owned(),
            "legacy/Cargo.toml".to_owned(),
            "-p".to_owned(),
            "lamquant-legacy-adapter".to_owned(),
            "--example".to_owned(),
            "legacy_evidence_probe".to_owned(),
            "--".to_owned(),
            profile_id.clone(),
            role.clone(),
        ],
        implementation_revision: revision.to_ascii_lowercase(),
        producer: Producer {
            authority: PRODUCER_AUTHORITY.to_owned(),
            executable_sha256: sha256_hex(PROBE_SOURCE),
            name: PRODUCER_NAME.to_owned(),
            version: PRODUCER_VERSION.to_owned(),
        },
        profile: profile_id,
        role,
        schema: SCHEMA.to_owned(),
        status: "PASS".to_owned(),
    };
    println!(
        "{}",
        serde_json::to_string(&receipt).expect("receipt serialises")
    );
}
