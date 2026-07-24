use std::path::Path;

use lamquant_lml_legacy::tensor_pack_v2::{
    PackV2Dtype, PackV2Encoding, PackV2Error, PackV2Reader, PackV2Writer, ViewSpec, LQTP2_MAGIC,
};
use sha2::{Digest, Sha256};

fn view_specs(reverse: bool) -> Vec<ViewSpec> {
    let fullband = ViewSpec::new(
        "fullband",
        PackV2Dtype::F32,
        PackV2Encoding::BfpInt16,
        &[2, 3],
        true,
        [0x11; 32],
    )
    .unwrap();
    let labels = ViewSpec::new(
        "labels",
        PackV2Dtype::U8,
        PackV2Encoding::Raw,
        &[4],
        true,
        [0x22; 32],
    )
    .unwrap();
    let l3 = ViewSpec::new(
        "l3",
        PackV2Dtype::F32,
        PackV2Encoding::Raw,
        &[2, 2],
        false,
        [0x33; 32],
    )
    .unwrap();
    if reverse {
        vec![labels, l3, fullband]
    } else {
        vec![fullband, labels, l3]
    }
}

fn write_pack(path: &Path, reverse: bool) {
    let mut writer = PackV2Writer::create(
        path,
        2,
        [0xaa; 32],
        [0xbb; 32],
        br#"{"schema":"lamquant.training-window-metadata/1"}"#.to_vec(),
        view_specs(reverse),
    )
    .unwrap();
    for row in 0..2 {
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

#[test]
fn multi_view_pack_is_deterministic_indexed_and_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let canonical = dir.path().join("canonical.lqtp2");
    let reordered = dir.path().join("reordered.lqtp2");
    write_pack(&canonical, false);
    write_pack(&reordered, true);
    assert_eq!(
        std::fs::read(&canonical).unwrap(),
        std::fs::read(&reordered).unwrap()
    );

    let reader = PackV2Reader::open(&canonical, Some([0xaa; 32]), Some([0xbb; 32])).unwrap();
    assert_eq!(reader.row_count(), 2);
    assert_eq!(reader.view_names(), vec!["fullband", "l3", "labels"]);
    assert_eq!(
        reader.metadata(),
        br#"{"schema":"lamquant.training-window-metadata/1"}"#
    );
    assert!(reader.view("fullband").unwrap().required());
    assert!(!reader.view("l3").unwrap().required());
    assert_eq!(reader.row_raw("labels", 1).unwrap(), &[1, 1, 2, 3]);

    let raw_l3 = reader.dequantize_f32("l3", 1).unwrap();
    assert_eq!(raw_l3, vec![11.0, 12.0, 13.0, 14.0]);
    let fullband = reader.dequantize_f32("fullband", 0).unwrap();
    let expected = [1.0_f32, -2.0, 3.0, 0.5, 0.0, -0.5];
    for (actual, expected) in fullband.iter().zip(expected) {
        assert!((actual - expected).abs() <= 3.0 / 32767.0 + 1e-6);
    }
}

#[test]
fn expected_snapshot_hashes_are_fail_closed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("snapshot.lqtp2");
    write_pack(&path, false);
    assert!(matches!(
        PackV2Reader::open(&path, Some([0x00; 32]), Some([0xbb; 32])),
        Err(PackV2Error::ManifestMismatch)
    ));
    assert!(matches!(
        PackV2Reader::open(&path, Some([0xaa; 32]), Some([0x00; 32])),
        Err(PackV2Error::ViewSpecMismatch)
    ));
}

#[test]
fn owned_file_reader_remains_bound_across_path_replacement() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("snapshot.lqtp2");
    let displaced = dir.path().join("displaced.lqtp2");
    write_pack(&path, false);

    let file = std::fs::File::open(&path).unwrap();
    std::fs::rename(&path, &displaced).unwrap();
    std::fs::write(&path, b"not an LQTP2 snapshot").unwrap();

    let reader = PackV2Reader::from_file(file, Some([0xaa; 32]), Some([0xbb; 32])).unwrap();
    assert_eq!(reader.row_raw("labels", 1).unwrap(), &[1, 1, 2, 3]);
    assert!(PackV2Reader::open(&path, None, None).is_err());
}

#[test]
fn corruption_truncation_and_trailing_bytes_are_rejected_at_open() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("source.lqtp2");
    write_pack(&path, false);
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..4], LQTP2_MAGIC);

    for length in [0, 3, 255, bytes.len() - 1] {
        let candidate = dir.path().join(format!("short-{length}.lqtp2"));
        std::fs::write(&candidate, &bytes[..length]).unwrap();
        assert!(PackV2Reader::open(&candidate, None, None).is_err());
    }

    let corrupt = dir.path().join("corrupt.lqtp2");
    let mut corrupt_bytes = bytes.clone();
    let last = corrupt_bytes.len() - 1;
    corrupt_bytes[last] ^= 0x80;
    std::fs::write(&corrupt, corrupt_bytes).unwrap();
    assert!(matches!(
        PackV2Reader::open(&corrupt, None, None),
        Err(PackV2Error::IntegrityMismatch(_))
    ));

    let header_corrupt = dir.path().join("header-corrupt.lqtp2");
    let mut header_bytes = bytes.clone();
    header_bytes[80] ^= 1;
    std::fs::write(&header_corrupt, header_bytes).unwrap();
    assert!(matches!(
        PackV2Reader::open(&header_corrupt, None, None),
        Err(PackV2Error::IntegrityMismatch("header"))
    ));

    let directory_corrupt = dir.path().join("directory-corrupt.lqtp2");
    let mut directory_bytes = bytes.clone();
    directory_bytes[256 + 128] ^= 1;
    std::fs::write(&directory_corrupt, directory_bytes).unwrap();
    assert!(matches!(
        PackV2Reader::open(&directory_corrupt, None, None),
        Err(PackV2Error::IntegrityMismatch("directory"))
    ));

    let metadata_corrupt = dir.path().join("metadata-corrupt.lqtp2");
    let mut metadata_bytes = bytes.clone();
    let metadata_offset = u64::from_le_bytes(metadata_bytes[48..56].try_into().unwrap()) as usize;
    metadata_bytes[metadata_offset] ^= 1;
    std::fs::write(&metadata_corrupt, metadata_bytes).unwrap();
    assert!(matches!(
        PackV2Reader::open(&metadata_corrupt, None, None),
        Err(PackV2Error::IntegrityMismatch("metadata"))
    ));

    let trailing = dir.path().join("trailing.lqtp2");
    let mut trailing_bytes = bytes;
    trailing_bytes.push(0);
    std::fs::write(&trailing, trailing_bytes).unwrap();
    assert!(PackV2Reader::open(&trailing, None, None).is_err());

    let source_reader = PackV2Reader::open(&path, None, None).unwrap();
    let fullband = source_reader.view("fullband").unwrap();
    let padding_start = (fullband.data_offset() + fullband.data_length()) as usize;
    let padding_end = source_reader.view("l3").unwrap().data_offset() as usize;
    assert!(padding_end > padding_start);
    drop(source_reader);
    let padding = dir.path().join("padding.lqtp2");
    let mut padding_bytes = std::fs::read(&path).unwrap();
    padding_bytes[padding_start] = 1;
    std::fs::write(&padding, padding_bytes).unwrap();
    assert!(matches!(
        PackV2Reader::open(&padding, None, None),
        Err(PackV2Error::InvalidLayout("view padding"))
    ));
}

#[test]
fn writer_refuses_short_views_and_invalid_specs() {
    assert!(ViewSpec::new(
        "",
        PackV2Dtype::F32,
        PackV2Encoding::Raw,
        &[2, 3],
        true,
        [0; 32],
    )
    .is_err());
    assert!(ViewSpec::new(
        "too-deep",
        PackV2Dtype::F32,
        PackV2Encoding::Raw,
        &[1, 1, 1, 1, 1],
        true,
        [0; 32],
    )
    .is_err());
    assert!(ViewSpec::new(
        "bad-bfp",
        PackV2Dtype::I16,
        PackV2Encoding::BfpInt16,
        &[2, 3],
        true,
        [0; 32],
    )
    .is_err());

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("short.lqtp2");
    let mut writer = PackV2Writer::create(
        &path,
        2,
        [0; 32],
        [0; 32],
        b"{}".to_vec(),
        vec![ViewSpec::new(
            "x",
            PackV2Dtype::F32,
            PackV2Encoding::Raw,
            &[1],
            true,
            [0; 32],
        )
        .unwrap()],
    )
    .unwrap();
    writer.write_f32_row("x", &[1.0]).unwrap();
    assert!(writer.finish().is_err());
    assert!(!path.exists());

    let bool_path = dir.path().join("bool.lqtp2");
    let mut bool_writer = PackV2Writer::create(
        &bool_path,
        1,
        [0; 32],
        [0; 32],
        b"{}".to_vec(),
        vec![ViewSpec::new(
            "valid",
            PackV2Dtype::Bool,
            PackV2Encoding::Raw,
            &[1],
            true,
            [0; 32],
        )
        .unwrap()],
    )
    .unwrap();
    assert!(bool_writer.write_raw_row("valid", &[2]).is_err());

    let nonfinite_path = dir.path().join("nonfinite.lqtp2");
    let mut nonfinite_writer = PackV2Writer::create(
        &nonfinite_path,
        1,
        [0; 32],
        [0; 32],
        b"{}".to_vec(),
        vec![ViewSpec::new(
            "signal",
            PackV2Dtype::F32,
            PackV2Encoding::BfpInt16,
            &[1],
            true,
            [0; 32],
        )
        .unwrap()],
    )
    .unwrap();
    assert!(nonfinite_writer
        .write_f32_row("signal", &[f32::NAN])
        .is_err());
}

#[test]
fn lqtp2_wire_layout_is_pinned() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("golden.lqtp2");
    write_pack(&path, false);
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(bytes.len(), 1_032);
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        "7d8b5961521b1eaa807cf763b5de92f657ed2ec5244a523860a30cdbc9650f88"
    );
}

#[cfg(unix)]
#[test]
fn lqtp2_path_reader_rejects_terminal_symlinks() {
    use std::os::unix::fs::symlink;

    let dir = tempfile::tempdir().unwrap();
    let real = dir.path().join("real.lqtp2");
    let link = dir.path().join("link.lqtp2");
    write_pack(&real, false);
    symlink(&real, &link).unwrap();

    assert!(PackV2Reader::open(&link, Some([0xaa; 32]), Some([0xbb; 32])).is_err());
}
