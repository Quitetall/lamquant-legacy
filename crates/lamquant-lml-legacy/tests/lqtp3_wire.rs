use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use filetime::{set_file_mtime, FileTime};
use lamquant_lml_legacy::tensor_pack_v3::{
    PackV3Compression, PackV3Dtype, PackV3Encoding, PackV3Error, PackV3Reader, PackV3Writer,
    ViewSpecV3, LQTP3_MAGIC,
};
use sha2::{Digest, Sha256};

fn specs(reverse: bool) -> Vec<ViewSpecV3> {
    let fullband = ViewSpecV3::new(
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
    .unwrap();
    let labels = ViewSpecV3::new(
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
    .unwrap();
    let l3 = ViewSpecV3::new(
        "l3",
        PackV3Dtype::F32,
        PackV3Encoding::Raw,
        &[2, 2],
        false,
        [0x33; 32],
        2,
        PackV3Compression::Zstd,
        3,
    )
    .unwrap();
    if reverse {
        vec![labels, l3, fullband]
    } else {
        vec![fullband, labels, l3]
    }
}

fn write_pack(path: &Path, reverse: bool) {
    let mut writer = PackV3Writer::create(
        path,
        5,
        [0xaa; 32],
        [0xbb; 32],
        br#"{"schema":"lamquant.training-window-metadata/1"}"#.to_vec(),
        specs(reverse),
    )
    .unwrap();
    for row in 0..5 {
        let offset = row as f32 * 10.0;
        writer
            .write_f32_row(
                "fullband",
                &[
                    1.0 + offset,
                    -2.0 - offset,
                    3.0 + offset,
                    0.5 + offset,
                    0.0,
                    -0.5 - offset,
                ],
            )
            .unwrap();
        writer
            .write_raw_row("labels", &[row as u8, 1, 2, 3])
            .unwrap();
        writer
            .write_f32_row(
                "l3",
                &[offset + 1.0, offset + 2.0, offset + 3.0, offset + 4.0],
            )
            .unwrap();
    }
    writer.finish().unwrap();
}

fn write_signal_pack(
    path: &Path,
    encoding: PackV3Encoding,
    compression: PackV3Compression,
    level: i32,
    chunk_rows: usize,
) {
    let mut writer = PackV3Writer::create(
        path,
        6,
        [0xa1; 32],
        [compression.to_u8(); 32],
        b"canonical-row-plan".to_vec(),
        vec![ViewSpecV3::new(
            "signal",
            PackV3Dtype::F32,
            encoding,
            &[2, 2],
            true,
            [0x51; 32],
            chunk_rows,
            compression,
            level,
        )
        .unwrap()],
    )
    .unwrap();
    for row in 0..6 {
        let base = row as f32 * 4.0;
        writer
            .write_f32_row("signal", &[base + 1.0, base + 2.0, base + 3.0, base + 4.0])
            .unwrap();
    }
    writer.finish().unwrap();
}

fn write_identity_variant(
    path: &Path,
    manifest: [u8; 32],
    physical_view_spec: [u8; 32],
    semantic_view_spec: [u8; 32],
    metadata: &[u8],
) {
    let mut writer = PackV3Writer::create(
        path,
        2,
        manifest,
        physical_view_spec,
        metadata.to_vec(),
        vec![ViewSpecV3::new(
            "signal",
            PackV3Dtype::F32,
            PackV3Encoding::Raw,
            &[2],
            true,
            semantic_view_spec,
            1,
            PackV3Compression::None,
            0,
        )
        .unwrap()],
    )
    .unwrap();
    writer.write_f32_row("signal", &[1.0, 2.0]).unwrap();
    writer.write_f32_row("signal", &[3.0, 4.0]).unwrap();
    writer.finish().unwrap();
}

#[test]
fn deterministic_multi_view_chunks_round_trip_and_gather() {
    let dir = tempfile::tempdir().unwrap();
    let canonical = dir.path().join("canonical.lqtp3");
    let reordered = dir.path().join("reordered.lqtp3");
    write_pack(&canonical, false);
    write_pack(&reordered, true);
    assert_eq!(
        std::fs::read(&canonical).unwrap(),
        std::fs::read(&reordered).unwrap()
    );

    let reader = PackV3Reader::open(&canonical, Some([0xaa; 32]), Some([0xbb; 32])).unwrap();
    assert_eq!(reader.row_count(), 5);
    assert_eq!(reader.view_names(), vec!["fullband", "l3", "labels"]);
    assert_eq!(
        reader.metadata(),
        br#"{"schema":"lamquant.training-window-metadata/1"}"#
    );
    assert_ne!(reader.logical_root_sha256(), &[0; 32]);

    let fullband = reader.view("fullband").unwrap();
    assert_eq!(fullband.chunk_count(), 3);
    assert_eq!(fullband.compression(), PackV3Compression::Zstd);
    let chunks = reader.chunks_for_view("fullband").unwrap();
    assert_eq!(chunks[0].row_start(), 0);
    assert_eq!(chunks[0].row_count(), 2);
    assert_eq!(chunks[1].row_start(), 2);
    assert_eq!(chunks[1].row_count(), 2);
    assert_eq!(chunks[2].row_start(), 4);
    assert_eq!(chunks[2].row_count(), 1);
    assert!(chunks.iter().all(|chunk| {
        chunk.stored_length() > 0
            && chunk.encoded_length() > 0
            && chunk.decoded_length() > 0
            && chunk.payload_sha256() != &[0; 32]
    }));

    assert_eq!(reader.read_raw_row("labels", 4).unwrap(), vec![4, 1, 2, 3]);
    assert_eq!(
        reader.gather_f32("l3", &[4, 0, 2]).unwrap(),
        vec![
            vec![41.0, 42.0, 43.0, 44.0],
            vec![1.0, 2.0, 3.0, 4.0],
            vec![21.0, 22.0, 23.0, 24.0],
        ]
    );
    let decoded = reader.dequantize_f32("fullband", 0).unwrap();
    for (actual, expected) in decoded.iter().zip([1.0_f32, -2.0, 3.0, 0.5, 0.0, -0.5]) {
        assert!((actual - expected).abs() <= 3.0 / 32767.0 + 1e-6);
    }
}

#[test]
fn expected_hashes_short_writes_and_invalid_specs_fail_closed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("snapshot.lqtp3");
    write_pack(&path, false);
    assert!(matches!(
        PackV3Reader::open(&path, Some([0; 32]), Some([0xbb; 32])),
        Err(PackV3Error::ManifestMismatch)
    ));
    assert!(matches!(
        PackV3Reader::open(&path, Some([0xaa; 32]), Some([0; 32])),
        Err(PackV3Error::ViewSpecMismatch)
    ));

    assert!(ViewSpecV3::new(
        "bad-bfp",
        PackV3Dtype::I16,
        PackV3Encoding::BfpInt16,
        &[2, 3],
        true,
        [0; 32],
        2,
        PackV3Compression::None,
        0,
    )
    .is_err());
    assert!(ViewSpecV3::new(
        "bad-level",
        PackV3Dtype::F32,
        PackV3Encoding::Raw,
        &[1],
        true,
        [0; 32],
        2,
        PackV3Compression::None,
        1,
    )
    .is_err());

    let short = dir.path().join("short-write.lqtp3");
    let mut writer = PackV3Writer::create(
        &short,
        2,
        [0; 32],
        [0; 32],
        vec![],
        vec![ViewSpecV3::new(
            "x",
            PackV3Dtype::F32,
            PackV3Encoding::Raw,
            &[1],
            true,
            [0; 32],
            1,
            PackV3Compression::None,
            0,
        )
        .unwrap()],
    )
    .unwrap();
    writer.write_f32_row("x", &[1.0]).unwrap();
    assert!(writer.finish().is_err());
    assert!(!short.exists());
}

#[test]
fn owned_file_reader_remains_bound_across_path_replacement() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("snapshot.lqtp3");
    let displaced = dir.path().join("displaced.lqtp3");
    write_pack(&path, false);

    let file = std::fs::File::open(&path).unwrap();
    std::fs::rename(&path, &displaced).unwrap();
    std::fs::write(&path, b"not an LQTP3 snapshot").unwrap();

    let reader = PackV3Reader::from_file(file, Some([0xaa; 32]), Some([0xbb; 32])).unwrap();
    assert_eq!(reader.read_raw_row("labels", 4).unwrap(), vec![4, 1, 2, 3]);
    assert!(PackV3Reader::open(&path, None, None).is_err());
}

#[test]
fn corruption_truncation_trailing_bytes_and_hostile_lengths_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source.lqtp3");
    write_pack(&source, false);
    let bytes = std::fs::read(&source).unwrap();
    assert_eq!(&bytes[..4], LQTP3_MAGIC);

    for length in [0, 3, 511, bytes.len() - 1] {
        let path = dir.path().join(format!("truncated-{length}.lqtp3"));
        std::fs::write(&path, &bytes[..length]).unwrap();
        assert!(PackV3Reader::open(&path, None, None).is_err());
    }

    let corrupt = dir.path().join("corrupt.lqtp3");
    let mut corrupt_bytes = bytes.clone();
    let last = corrupt_bytes.len() - 1;
    corrupt_bytes[last] ^= 0x80;
    std::fs::write(&corrupt, corrupt_bytes).unwrap();
    assert!(matches!(
        PackV3Reader::open(&corrupt, None, None),
        Err(PackV3Error::IntegrityMismatch(_))
    ));

    let trailing = dir.path().join("trailing.lqtp3");
    let mut trailing_bytes = bytes.clone();
    trailing_bytes.push(0);
    std::fs::write(&trailing, trailing_bytes).unwrap();
    assert!(PackV3Reader::open(&trailing, None, None).is_err());

    // Header claims hostile view count. Header hash no longer matches, so parser
    // must reject before allocating from attacker-controlled count.
    let hostile = dir.path().join("hostile-count.lqtp3");
    let mut hostile_bytes = bytes;
    hostile_bytes[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
    std::fs::write(&hostile, hostile_bytes).unwrap();
    assert!(PackV3Reader::open(&hostile, None, None).is_err());
}

#[test]
fn logical_identity_ignores_chunking_and_compression_but_artifact_identity_does_not() {
    let dir = tempfile::tempdir().unwrap();
    let zstd1 = dir.path().join("zstd1.lqtp3");
    let zstd3 = dir.path().join("zstd3.lqtp3");
    let bfp8 = dir.path().join("bfp8.lqtp3");
    write_signal_pack(
        &zstd1,
        PackV3Encoding::BfpInt16,
        PackV3Compression::Zstd,
        1,
        2,
    );
    write_signal_pack(
        &zstd3,
        PackV3Encoding::BfpInt16,
        PackV3Compression::Zstd,
        3,
        5,
    );
    write_signal_pack(
        &bfp8,
        PackV3Encoding::BfpInt8,
        PackV3Compression::Zstd,
        3,
        5,
    );

    let left = PackV3Reader::open(&zstd1, None, None).unwrap();
    let right = PackV3Reader::open(&zstd3, None, None).unwrap();
    let lossy = PackV3Reader::open(&bfp8, None, None).unwrap();
    assert_eq!(
        left.logical_value_root_sha256(),
        right.logical_value_root_sha256()
    );
    assert_ne!(left.artifact_root_sha256(), right.artifact_root_sha256());
    assert_ne!(
        std::fs::read(&zstd1).unwrap(),
        std::fs::read(&zstd3).unwrap()
    );
    assert_ne!(
        left.logical_value_root_sha256(),
        lossy.logical_value_root_sha256()
    );
    let decoded = lossy.dequantize_f32("signal", 5).unwrap();
    for (actual, expected) in decoded.iter().zip([21.0_f32, 22.0, 23.0, 24.0]) {
        assert!((actual - expected).abs() <= 24.0 / 127.0 + 1e-6);
    }
}

#[test]
fn materialized_identity_is_independent_of_source_row_plan_and_view_specs() {
    let dir = tempfile::tempdir().unwrap();
    let left_path = dir.path().join("left.lqtp3");
    let right_path = dir.path().join("right.lqtp3");
    write_identity_variant(&left_path, [1; 32], [2; 32], [3; 32], b"row-plan-a");
    write_identity_variant(&right_path, [4; 32], [5; 32], [6; 32], b"row-plan-b");
    let left = PackV3Reader::open(&left_path, None, None).unwrap();
    let right = PackV3Reader::open(&right_path, None, None).unwrap();
    assert_eq!(
        left.materialized_value_root_sha256(),
        right.materialized_value_root_sha256()
    );
    assert_ne!(left.artifact_root_sha256(), right.artifact_root_sha256());
    assert_ne!(
        Sha256::digest(std::fs::read(left_path).unwrap()),
        Sha256::digest(std::fs::read(right_path).unwrap())
    );
}

#[test]
fn writer_publication_never_replaces_concurrent_destination() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("race.lqtp3");
    let mut writer = PackV3Writer::create(
        &path,
        1,
        [0; 32],
        [0; 32],
        vec![],
        vec![ViewSpecV3::new(
            "signal",
            PackV3Dtype::F32,
            PackV3Encoding::Raw,
            &[1],
            true,
            [0; 32],
            1,
            PackV3Compression::None,
            0,
        )
        .unwrap()],
    )
    .unwrap();
    writer.write_f32_row("signal", &[1.0]).unwrap();
    std::fs::write(&path, b"concurrent-winner").unwrap();
    assert!(writer.finish().is_err());
    assert_eq!(std::fs::read(&path).unwrap(), b"concurrent-winner");
}

#[test]
fn stored_chunk_seam_gather_order_and_bounded_lru_are_observable() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cache.lqtp3");
    write_pack(&path, false);
    let reader = PackV3Reader::open_with_cache_slots(&path, None, None, 2).unwrap();
    let stored = reader.chunk_stored_bytes("fullband", 0).unwrap();
    assert_eq!(
        &Sha256::digest(stored)[..],
        reader.chunks_for_view("fullband").unwrap()[0].payload_sha256()
    );
    assert!(matches!(
        reader.chunk_stored_bytes("fullband", 3),
        Err(PackV3Error::ChunkOutOfBounds { .. })
    ));

    let gathered = reader.gather_f32("l3", &[4, 0, 4, 2]).unwrap();
    assert_eq!(gathered[0], vec![41.0, 42.0, 43.0, 44.0]);
    assert_eq!(gathered[0], gathered[2]);
    assert_eq!(gathered[1], vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(gathered[3], vec![21.0, 22.0, 23.0, 24.0]);
    assert_eq!(
        reader.gather_f32_flat("l3", &[4, 0, 4, 2]).unwrap(),
        gathered.into_iter().flatten().collect::<Vec<_>>()
    );
    let stats = reader.cache_stats().unwrap();
    assert_eq!(stats.slots, 2);
    assert!(stats.resident_chunks <= 2);
    assert!(stats.misses >= 3);
    assert!(stats.evictions >= 1);
    reader.dequantize_f32("l3", 4).unwrap();
    assert!(reader.cache_stats().unwrap().hits >= 1);
}

#[test]
fn verified_open_defers_payload_scan_but_first_access_fails_closed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("receipt.lqtp3");
    write_pack(&path, false);
    let reader = PackV3Reader::open(&path, None, None).unwrap();
    let bundle_sha: [u8; 32] = Sha256::digest(std::fs::read(&path).unwrap()).into();
    assert_ne!(bundle_sha.as_slice(), reader.artifact_root_sha256());
    let receipt = reader.verification_receipt(bundle_sha).encode();
    let stored_offset = reader.chunks_for_view("fullband").unwrap()[0].stored_offset();
    drop(reader);

    let metadata = std::fs::metadata(&path).unwrap();
    let original_mtime = FileTime::from_last_modification_time(&metadata);
    let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.seek(SeekFrom::Start(stored_offset)).unwrap();
    file.write_all(&[0xff]).unwrap();
    file.sync_all().unwrap();
    drop(file);
    set_file_mtime(&path, original_mtime).unwrap();

    let deferred =
        PackV3Reader::open_verified(&path, Some([0xaa; 32]), Some([0xbb; 32]), &receipt).unwrap();
    assert!(matches!(
        deferred.dequantize_f32("fullband", 0),
        Err(PackV3Error::IntegrityMismatch("chunk payload"))
    ));
    assert!(PackV3Reader::open(&path, None, None).is_err());
}

#[test]
fn verified_owned_file_reader_remains_bound_across_path_replacement() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("receipt.lqtp3");
    let displaced = dir.path().join("displaced.lqtp3");
    write_pack(&path, false);
    let verified = PackV3Reader::open(&path, None, None).unwrap();
    let bundle_sha: [u8; 32] = Sha256::digest(std::fs::read(&path).unwrap()).into();
    let receipt = verified.verification_receipt(bundle_sha).encode();
    drop(verified);

    let file = std::fs::File::open(&path).unwrap();
    std::fs::rename(&path, &displaced).unwrap();
    std::fs::write(&path, b"not an LQTP3 snapshot").unwrap();

    let reader =
        PackV3Reader::from_verified_file(file, Some([0xaa; 32]), Some([0xbb; 32]), &receipt)
            .unwrap();
    assert_eq!(reader.read_raw_row("labels", 4).unwrap(), vec![4, 1, 2, 3]);
}

#[test]
fn stale_receipt_and_symlink_paths_fail_closed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("source.lqtp3");
    write_pack(&path, false);
    let reader = PackV3Reader::open(&path, None, None).unwrap();
    let bundle_sha: [u8; 32] = Sha256::digest(std::fs::read(&path).unwrap()).into();
    let receipt = reader.verification_receipt(bundle_sha).encode();
    drop(reader);
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap();
    file.write_all(&[0]).unwrap();
    drop(file);
    assert!(matches!(
        PackV3Reader::open_verified(&path, None, None, &receipt),
        Err(PackV3Error::ReceiptMismatch("file size"))
    ));

    #[cfg(unix)]
    {
        let link = dir.path().join("link.lqtp3");
        std::os::unix::fs::symlink(&path, &link).unwrap();
        assert!(PackV3Reader::open(&link, None, None).is_err());
    }
}
