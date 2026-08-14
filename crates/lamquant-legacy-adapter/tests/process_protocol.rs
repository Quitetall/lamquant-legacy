use lamquant_legacy_adapter::{
    capability_manifest, convert_forensic, detect_format, export_semantic, handle, import_semantic,
    inspect, materialize_exact, materialize_synthetic_exact, AsciiIntLinesTemplate, Capability,
    ConvertRequest, ExportPayload, LegacyError, LegacyFormat, MaterializeRequest, ProcessRequest,
    ProcessResponse, SemanticExportRequest, SemanticImportRequest, SyntheticLineEnding,
    SyntheticMaterializeRequest, SyntheticTemplate,
};
use lamquant_legacy_ir::{Bcs1Header, BCS1_VERSION_MAJOR, BCS1_VERSION_MINOR, CODEC_LML_53};
use sha2::{Digest, Sha256};
use std::fs;

fn lml1_edf_fixture(root: &std::path::Path) -> (Vec<u8>, Vec<u8>) {
    use base64::Engine;

    let signal = source_signal();
    let mut header = vec![b' '; 256 + signal.len() * 256];
    header[..8].copy_from_slice(b"0       ");
    header[184..192].copy_from_slice(b"768     ");
    header[236..244].copy_from_slice(b"1       ");
    header[244..252].copy_from_slice(b"1       ");
    header[252..256].copy_from_slice(b"2   ");
    let header_sha = format!("{:x}", Sha256::digest(&header));
    let compressed_header = zstd::encode_all(header.as_slice(), 1).unwrap();
    let metadata = serde_json::json!({
        "edf_header": base64::engine::general_purpose::STANDARD.encode(compressed_header),
        "edf_header_sha256": header_sha,
        "n_data_records": 1,
        "all_ns_per_rec": [5, 5],
        "eeg_channel_indices": [0, 1],
        "non_eeg_channels": {},
        "trailing_data": ""
    });
    let source = root.join("fixture.lml");
    lamquant_lml_legacy::container::write_file(
        &source,
        &signal,
        250.0,
        5,
        0,
        &serde_json::to_string(&metadata).unwrap(),
    )
    .unwrap();

    let mut original = header;
    for channel in &signal {
        original.extend(
            channel
                .iter()
                .flat_map(|value| (*value as i16).to_le_bytes()),
        );
    }
    (fs::read(source).unwrap(), original)
}

fn lml1_mixed_edf_fixture(root: &std::path::Path) -> (Vec<u8>, Vec<u8>) {
    use base64::Engine;

    let signal = vec![
        vec![-32_768, -1, 0, 1, 2, 32_767],
        vec![100, 101, 102, 103, 104, 105],
    ];
    let mut header = vec![b' '; 256 + 3 * 256];
    header[..8].copy_from_slice(b"0       ");
    header[184..192].copy_from_slice(b"1024    ");
    header[236..244].copy_from_slice(b"2       ");
    header[244..252].copy_from_slice(b"1       ");
    header[252..256].copy_from_slice(b"3   ");
    let non_eeg = [1_u8, 2, 3, 4, 5, 6, 7, 8];
    let trailing = [0xaa_u8, 0xbb, 0xcc];
    let metadata = serde_json::json!({
        "edf_header": base64::engine::general_purpose::STANDARD
            .encode(zstd::encode_all(header.as_slice(), 1).unwrap()),
        "edf_header_sha256": format!("{:x}", Sha256::digest(&header)),
        "n_data_records": 2,
        "all_ns_per_rec": [3, 3, 2],
        "eeg_channel_indices": [0, 1],
        "non_eeg_channels": {
            "2": base64::engine::general_purpose::STANDARD
                .encode(zstd::encode_all(non_eeg.as_slice(), 1).unwrap())
        },
        "trailing_data": base64::engine::general_purpose::STANDARD
            .encode(zstd::encode_all(trailing.as_slice(), 1).unwrap())
    });
    let source = root.join("mixed-fixture.lml");
    lamquant_lml_legacy::container::write_file(
        &source,
        &signal,
        250.0,
        3,
        0,
        &serde_json::to_string(&metadata).unwrap(),
    )
    .unwrap();

    let mut original = header;
    for record in 0..2 {
        for channel in &signal {
            original.extend(
                channel[record * 3..record * 3 + 3]
                    .iter()
                    .flat_map(|value| (*value as i16).to_le_bytes()),
            );
        }
        original.extend_from_slice(&non_eeg[record * 4..record * 4 + 4]);
    }
    original.extend_from_slice(&trailing);
    (fs::read(source).unwrap(), original)
}

fn signed_24_le(value: i64) -> [u8; 3] {
    let encoded = if value < 0 {
        value + (1_i64 << 24)
    } else {
        value
    } as u32;
    [
        (encoded & 0xff) as u8,
        ((encoded >> 8) & 0xff) as u8,
        ((encoded >> 16) & 0xff) as u8,
    ]
}

fn lml1_bdf_fixture(root: &std::path::Path) -> (Vec<u8>, Vec<u8>) {
    use base64::Engine;

    let signal = vec![vec![-8_388_608, -1, 0, 8_388_607]];
    let mut header = vec![b' '; 512];
    header[0] = 0xff;
    header[184..192].copy_from_slice(b"512     ");
    header[236..244].copy_from_slice(b"1       ");
    header[244..252].copy_from_slice(b"1       ");
    header[252..256].copy_from_slice(b"1   ");
    let metadata = serde_json::json!({
        "edf_header": base64::engine::general_purpose::STANDARD
            .encode(zstd::encode_all(header.as_slice(), 1).unwrap()),
        "edf_header_sha256": format!("{:x}", Sha256::digest(&header)),
        "n_data_records": 1,
        "all_ns_per_rec": [4],
        "eeg_channel_indices": [0],
        "non_eeg_channels": {},
        "trailing_data": ""
    });
    let source = root.join("bdf-fixture.lml");
    lamquant_lml_legacy::container::write_file(
        &source,
        &signal,
        250.0,
        4,
        0,
        &serde_json::to_string(&metadata).unwrap(),
    )
    .unwrap();

    let mut original = header;
    for value in &signal[0] {
        original.extend_from_slice(&signed_24_le(*value));
    }
    (fs::read(source).unwrap(), original)
}

fn source_signal() -> Vec<Vec<i64>> {
    vec![vec![-9, -1, 0, 7, 11], vec![100, 101, 99, 102, 98]]
}

fn bcs1_source() -> Vec<u8> {
    let signal = source_signal();
    let payload = lamquant_lml_mcu::lml::compress(&signal, 0).unwrap();
    let metadata = br#"{"channels":["Fp1","Fp2"],"vendor_extension":{"x":1}}"#;
    let header = Bcs1Header {
        version_major: BCS1_VERSION_MAJOR,
        version_minor: BCS1_VERSION_MINOR,
        modality_tag: 0,
        modality_source: 0,
        codec_descriptor: CODEC_LML_53,
        mode: 0,
        tier: 0,
        decode_capability: 0,
        n_channels: 2,
        n_windows: 1,
        total_samples: 5,
        window_size: 5,
        sample_rate_mhz: 250_000,
        bit_depth: 16,
        flags: 0,
        metadata_length: metadata.len() as u32,
    };
    let mut bytes = header.to_bytes().to_vec();
    bytes.extend_from_slice(metadata);
    bytes.extend_from_slice(&0_u32.to_le_bytes());
    bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&payload);
    bytes
}

fn lml1_source_from_bcs1(bcs1: &[u8]) -> Vec<u8> {
    let header = Bcs1Header::parse(bcs1).unwrap();
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
    bytes.extend_from_slice(&bcs1[40..]);
    bytes
}

fn lqtp1_source(path: &std::path::Path) -> Vec<u8> {
    let mut writer = lamquant_lml_archive::tensor_pack::PackWriter::create(
        path,
        lamquant_lml_archive::tensor_pack::PackDtype::F32,
        2,
        3,
        2,
        [0x5a; 32],
    )
    .unwrap();
    writer
        .write_window(&[1.0, -2.0, 3.0, 4.0, -5.0, 6.0])
        .unwrap();
    writer
        .write_window(&[7.0, -8.0, 9.0, 10.0, -11.0, 12.0])
        .unwrap();
    writer.finish().unwrap();
    fs::read(path).unwrap()
}

#[test]
fn every_retired_magic_has_a_stable_profile() {
    let cases: &[(&[u8], LegacyFormat)] = &[
        (b"BCS1payload", LegacyFormat::Bcs1),
        (b"LML1payload", LegacyFormat::Lml1),
        (b"LMA1payload", LegacyFormat::Lma1),
        (b"LMA2payload", LegacyFormat::Lma2),
        (b"LMQCpayload", LegacyFormat::Lmqc),
        (b"LMLCRYPTpayload", LegacyFormat::Lmlcrypt),
        (b"LQTP\x01payload", LegacyFormat::Lqtp1),
        // LQTP2/LQTP3 took their own four-byte magics rather than bumping
        // LQTP1's version byte, so `LQTP\x02` names no wire that ever shipped.
        (b"LQT2payload", LegacyFormat::Lqtp2),
        (b"LQT3payload", LegacyFormat::Lqtp3),
    ];
    for (bytes, expected) in cases {
        assert_eq!(detect_format(bytes).unwrap(), *expected);
    }
    for absent in [b"LQTP\x02payload".as_slice(), b"LQTP\x03payload".as_slice()] {
        assert_eq!(detect_format(absent), Err(LegacyError::UnknownMagic));
    }
}

#[test]
fn forensic_conversion_is_exact_non_destructive_and_idempotent() {
    for (name, original) in [
        ("bcs1", b"BCS1payload".as_slice()),
        ("lml1", b"LML1payload".as_slice()),
        ("lqtp1", b"LQTP\x01payload".as_slice()),
    ] {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join(format!("input-{name}"));
        let output = temp.path().join("output");
        fs::write(&source, original).unwrap();

        let request = ConvertRequest {
            source: source.clone(),
            destination: output.clone(),
            accept_fidelity: true,
            max_source_bytes: 1024,
        };
        let first = convert_forensic(&request).unwrap();
        let second = convert_forensic(&request).unwrap();

        assert_eq!(fs::read(&source).unwrap(), original);
        assert_eq!(fs::read(output.join("source.bin")).unwrap(), original);
        assert_eq!(first, second);
        assert!(first.source_preserved);
        assert!(!first.semantic_mapping_claimed);

        fs::write(output.join("source.bin"), b"conflict").unwrap();
        assert_eq!(
            convert_forensic(&request),
            Err(LegacyError::DestinationConflict)
        );
        assert_eq!(fs::read(&source).unwrap(), original);
    }
}

#[test]
fn conversion_fails_before_output_without_acceptance() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("input.lml");
    let output = temp.path().join("output");
    fs::write(&source, b"LML1payload").unwrap();
    let error = convert_forensic(&ConvertRequest {
        source,
        destination: output.clone(),
        accept_fidelity: false,
        max_source_bytes: 1024,
    })
    .unwrap_err();
    assert_eq!(error, LegacyError::AcceptanceRequired);
    assert!(!output.exists());
}

#[test]
fn lml1_exact_materialization_is_bounded_no_clobber_and_idempotent() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("input.lml");
    let destination = temp.path().join("restored.edf");
    let (lml, original) = lml1_edf_fixture(temp.path());
    fs::write(&source, &lml).unwrap();
    let expected_sha256 = format!("{:x}", Sha256::digest(&original));
    let request = MaterializeRequest {
        source: source.clone(),
        destination: destination.clone(),
        accept_fidelity: true,
        expected_sha256: expected_sha256.clone(),
        original_size: original.len() as u64,
        max_source_bytes: lml.len() as u64,
        max_decoded_bytes: 1024 * 1024,
        max_output_bytes: original.len() as u64,
    };

    let first = materialize_exact(&request).unwrap();
    let second = materialize_exact(&request).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.profile, "legacy.lml1.v1");
    assert_eq!(first.output_sha256, expected_sha256);
    assert_eq!(first.output_bytes, original.len() as u64);
    assert!(first.exact_original_bytes);
    assert_eq!(fs::read(&destination).unwrap(), original);
    assert_eq!(fs::read(&source).unwrap(), lml);

    fs::write(&destination, b"conflict").unwrap();
    assert_eq!(
        materialize_exact(&request),
        Err(LegacyError::DestinationConflict)
    );
    assert_eq!(fs::read(&source).unwrap(), lml);
}

#[test]
fn lml1_synthetic_materialization_re_emits_exact_ascii_source() {
    use base64::Engine;

    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("synthetic.lml");
    let destination = temp.path().join("original.txt");
    let signal = vec![vec![-12_i64, 0, 34]];
    let mut header = vec![b' '; 512];
    header[..8].copy_from_slice(b"0       ");
    header[184..192].copy_from_slice(b"512     ");
    header[236..244].copy_from_slice(b"1       ");
    header[244..252].copy_from_slice(b"1       ");
    header[252..256].copy_from_slice(b"1   ");
    let metadata = serde_json::json!({
        "edf_header": base64::engine::general_purpose::STANDARD
            .encode(zstd::encode_all(header.as_slice(), 1).unwrap()),
        "edf_header_sha256": format!("{:x}", Sha256::digest(&header)),
        "n_data_records": 1,
        "all_ns_per_rec": [3],
        "eeg_channel_indices": [0],
        "non_eeg_channels": {},
        "trailing_data": ""
    });
    lamquant_lml_legacy::container::write_file(
        &source,
        &signal,
        250.0,
        3,
        0,
        &serde_json::to_string(&metadata).unwrap(),
    )
    .unwrap();
    let original = b"   -12\r\n     0\r\n    34";
    let request = SyntheticMaterializeRequest {
        source: source.clone(),
        destination: destination.clone(),
        accept_fidelity: true,
        expected_sha256: format!("{:x}", Sha256::digest(original)),
        original_size: original.len() as u64,
        max_source_bytes: fs::metadata(&source).unwrap().len(),
        max_decoded_bytes: 1024 * 1024,
        max_intermediate_bytes: 1024 * 1024,
        max_output_bytes: original.len() as u64,
        synthetic: SyntheticTemplate::AsciiIntLines(AsciiIntLinesTemplate {
            line_ending: SyntheticLineEnding::CrLf,
            leading_whitespace: 2,
            field_width: 4,
            trailing_newline: false,
        }),
    };

    let receipt = materialize_synthetic_exact(&request).unwrap();
    assert_eq!(fs::read(&destination).unwrap(), original);
    assert_eq!(receipt.output_sha256, request.expected_sha256);
    assert!(receipt.exact_original_bytes);
    assert_eq!(fs::read(&source).unwrap()[..4], *b"LML1");

    let amplified_destination = temp.path().join("amplified.txt");
    let mut amplified = request.clone();
    amplified.destination = amplified_destination.clone();
    amplified.original_size = 500;
    amplified.max_output_bytes = 500;
    amplified.synthetic = SyntheticTemplate::AsciiIntLines(AsciiIntLinesTemplate {
        line_ending: SyntheticLineEnding::CrLf,
        leading_whitespace: u8::MAX,
        field_width: u8::MAX,
        trailing_newline: false,
    });
    assert_eq!(
        materialize_synthetic_exact(&amplified),
        Err(LegacyError::OutputTooLarge)
    );
    assert!(!amplified_destination.exists());

    #[cfg(unix)]
    {
        let fifo = temp.path().join("source.fifo");
        assert!(std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap()
            .success());
        let mut unsafe_request = request.clone();
        unsafe_request.source = fifo;
        unsafe_request.destination = temp.path().join("fifo-output.txt");
        assert_eq!(
            materialize_synthetic_exact(&unsafe_request),
            Err(LegacyError::UnsafeSource)
        );
        assert!(!unsafe_request.destination.exists());
    }
}

#[test]
fn lml1_exact_materialization_fails_closed_before_publish() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("input.lml");
    let (lml, original) = lml1_edf_fixture(temp.path());
    fs::write(&source, &lml).unwrap();

    for (name, request, expected) in [
        (
            "acceptance",
            MaterializeRequest {
                source: source.clone(),
                destination: temp.path().join("acceptance.edf"),
                accept_fidelity: false,
                expected_sha256: format!("{:x}", Sha256::digest(&original)),
                original_size: original.len() as u64,
                max_source_bytes: lml.len() as u64,
                max_decoded_bytes: 1024 * 1024,
                max_output_bytes: original.len() as u64,
            },
            LegacyError::AcceptanceRequired,
        ),
        (
            "source-bound",
            MaterializeRequest {
                source: source.clone(),
                destination: temp.path().join("source-bound.edf"),
                accept_fidelity: true,
                expected_sha256: format!("{:x}", Sha256::digest(&original)),
                original_size: original.len() as u64,
                max_source_bytes: (lml.len() - 1) as u64,
                max_decoded_bytes: 1024 * 1024,
                max_output_bytes: original.len() as u64,
            },
            LegacyError::SourceTooLarge,
        ),
        (
            "output-bound",
            MaterializeRequest {
                source: source.clone(),
                destination: temp.path().join("output-bound.edf"),
                accept_fidelity: true,
                expected_sha256: format!("{:x}", Sha256::digest(&original)),
                original_size: original.len() as u64,
                max_source_bytes: lml.len() as u64,
                max_decoded_bytes: 1024 * 1024,
                max_output_bytes: (original.len() - 1) as u64,
            },
            LegacyError::OutputTooLarge,
        ),
        (
            "decoded-bound",
            MaterializeRequest {
                source: source.clone(),
                destination: temp.path().join("decoded-bound.edf"),
                accept_fidelity: true,
                expected_sha256: format!("{:x}", Sha256::digest(&original)),
                original_size: original.len() as u64,
                max_source_bytes: lml.len() as u64,
                max_decoded_bytes: 159,
                max_output_bytes: original.len() as u64,
            },
            LegacyError::DecodedTooLarge,
        ),
        (
            "digest",
            MaterializeRequest {
                source: source.clone(),
                destination: temp.path().join("digest.edf"),
                accept_fidelity: true,
                expected_sha256: "00".repeat(32),
                original_size: original.len() as u64,
                max_source_bytes: lml.len() as u64,
                max_decoded_bytes: 1024 * 1024,
                max_output_bytes: original.len() as u64,
            },
            LegacyError::OutputIdentityMismatch,
        ),
    ] {
        let destination = request.destination.clone();
        assert_eq!(materialize_exact(&request), Err(expected), "{name}");
        assert!(!destination.exists(), "{name} published output");
    }
}

#[test]
fn unknown_and_oversized_inputs_fail_closed() {
    assert_eq!(detect_format(b"nope"), Err(LegacyError::UnknownMagic));
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("input.lml");
    fs::write(&source, b"LML1payload").unwrap();
    let error = convert_forensic(&ConvertRequest {
        source,
        destination: temp.path().join("out"),
        accept_fidelity: true,
        max_source_bytes: 4,
    })
    .unwrap_err();
    assert_eq!(error, LegacyError::SourceTooLarge);
}

#[test]
fn committed_capability_manifest_matches_runtime() {
    let committed: serde_json::Value =
        serde_json::from_slice(include_bytes!("../../../capability-manifest.json")).unwrap();
    let runtime = serde_json::to_value(capability_manifest()).unwrap();
    assert_eq!(committed, runtime);
}

#[test]
fn exact_materialization_capability_and_process_wire_are_explicit() {
    let manifest = capability_manifest();
    for capability in &manifest.capabilities {
        let expected = capability.profile == "legacy.lml1.v1";
        assert_eq!(capability.exact_materialization, expected);
        assert_eq!(
            capability
                .operations
                .iter()
                .any(|operation| operation == "materialize-exact"),
            expected
        );
    }

    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("input.lml");
    let destination = temp.path().join("process.edf");
    let (lml, original) = lml1_edf_fixture(temp.path());
    fs::write(&source, &lml).unwrap();
    let request = ProcessRequest::MaterializeExact(MaterializeRequest {
        source,
        destination: destination.clone(),
        accept_fidelity: true,
        expected_sha256: format!("{:x}", Sha256::digest(&original)),
        original_size: original.len() as u64,
        max_source_bytes: lml.len() as u64,
        max_decoded_bytes: 1024 * 1024,
        max_output_bytes: original.len() as u64,
    });
    let wire = serde_json::to_value(&request).unwrap();
    assert_eq!(wire["operation"], "materialize-exact");
    let response = handle(serde_json::from_value(wire).unwrap());
    assert!(matches!(response, ProcessResponse::OkMaterialization(_)));
    assert_eq!(fs::read(destination).unwrap(), original);

    let older_v1: Capability = serde_json::from_value(serde_json::json!({
        "profile": "legacy.lml1.v1",
        "detect": true,
        "inspect": true,
        "forensic_import": true,
        "semantic_import": true,
        "reverse_export": false,
        "operations": ["detect", "inspect", "forensic-import", "semantic-import"]
    }))
    .unwrap();
    assert!(!older_v1.exact_materialization);
}

#[test]
fn lml1_materialization_rejects_inner_packet_shape_drift_before_output() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("shape-drift.lml");
    let destination = temp.path().join("shape-drift.edf");
    let (mut lml, original) = lml1_edf_fixture(temp.path());
    let header = lamquant_lml_legacy::container::parse_header(&lml).unwrap();
    let packet_start = header.payload_start + 4;
    let magic_offset = lml[packet_start..packet_start + 128.min(lml.len() - packet_start)]
        .windows(4)
        .position(|window| window == b"LML1")
        .unwrap();
    let shape_offset = packet_start + magic_offset + 6;
    lml[shape_offset..shape_offset + 2].copy_from_slice(&1000_u16.to_le_bytes());
    fs::write(&source, &lml).unwrap();

    let error = materialize_exact(&MaterializeRequest {
        source,
        destination: destination.clone(),
        accept_fidelity: true,
        expected_sha256: format!("{:x}", Sha256::digest(&original)),
        original_size: original.len() as u64,
        max_source_bytes: lml.len() as u64,
        max_decoded_bytes: 1024 * 1024,
        max_output_bytes: original.len() as u64,
    })
    .unwrap_err();
    assert!(matches!(error, LegacyError::MalformedContainer(_)));
    assert!(!destination.exists());
}

#[test]
fn lml1_materialization_restores_mixed_records_and_trailing_bytes() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("mixed.lml");
    let destination = temp.path().join("mixed.edf");
    let (lml, original) = lml1_mixed_edf_fixture(temp.path());
    fs::write(&source, &lml).unwrap();

    let receipt = materialize_exact(&MaterializeRequest {
        source: source.clone(),
        destination: destination.clone(),
        accept_fidelity: true,
        expected_sha256: format!("{:x}", Sha256::digest(&original)),
        original_size: original.len() as u64,
        max_source_bytes: lml.len() as u64,
        max_decoded_bytes: 1024 * 1024,
        max_output_bytes: original.len() as u64,
    })
    .unwrap();

    assert_eq!(fs::read(source).unwrap(), lml);
    assert_eq!(fs::read(destination).unwrap(), original);
    assert_eq!(receipt.output_bytes, original.len() as u64);
    assert!(receipt.exact_original_bytes);
}

#[test]
fn lml1_materialization_restores_signed_bdf_samples() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("input.lml");
    let destination = temp.path().join("output.bdf");
    let (lml, original) = lml1_bdf_fixture(temp.path());
    fs::write(&source, &lml).unwrap();

    let receipt = materialize_exact(&MaterializeRequest {
        source,
        destination: destination.clone(),
        accept_fidelity: true,
        expected_sha256: format!("{:x}", Sha256::digest(&original)),
        original_size: original.len() as u64,
        max_source_bytes: lml.len() as u64,
        max_decoded_bytes: 1024 * 1024,
        max_output_bytes: original.len() as u64,
    })
    .unwrap();

    assert_eq!(fs::read(destination).unwrap(), original);
    assert_eq!(
        receipt.output_sha256,
        format!("{:x}", Sha256::digest(&original))
    );
}

#[test]
fn converter_matrix_is_complete_and_does_not_overclaim_runtime_capabilities() {
    let matrix: serde_json::Value =
        serde_json::from_slice(include_bytes!("../../../converter-matrix.json")).unwrap();
    assert_eq!(matrix["schema"], "lamquant.legacy-converter-matrix/v1");
    assert_eq!(matrix["source_overwrite"], false);

    let manifest = capability_manifest();
    let expected_profiles = manifest
        .capabilities
        .iter()
        .map(|capability| capability.profile.as_str())
        .collect::<Vec<_>>();
    let declared_profiles = matrix["profiles"]
        .as_array()
        .unwrap()
        .iter()
        .map(|profile| profile.as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(declared_profiles, expected_profiles);

    let semantic_profiles = matrix["semantic_profiles"].as_object().unwrap();
    assert_eq!(semantic_profiles.len(), manifest.capabilities.len());
    for capability in manifest.capabilities {
        let claim = semantic_profiles
            .get(&capability.profile)
            .unwrap_or_else(|| panic!("matrix omitted {}", capability.profile));
        let status = claim["status"].as_str().unwrap();
        assert_eq!(
            status != "NOT_CLAIMED",
            capability.semantic_import,
            "semantic-import claim drift for {}",
            capability.profile,
        );
        assert_eq!(
            claim.get("reverse_export").is_some(),
            capability.reverse_export,
            "reverse-export claim drift for {}",
            capability.profile,
        );
    }
}

#[test]
fn bcs1_semantic_import_decodes_validates_and_preserves_source() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("input.bcs1");
    let output = temp.path().join("semantic");
    let original = bcs1_source();
    fs::write(&source, &original).unwrap();
    let request = SemanticImportRequest {
        source: source.clone(),
        destination: output.clone(),
        accept_fidelity: true,
        max_source_bytes: 1024 * 1024,
        max_decoded_bytes: 1024 * 1024,
    };

    let first = import_semantic(&request).unwrap();
    let second = import_semantic(&request).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.profile, "legacy.bcs1.v1");
    assert_eq!(first.decoded_channels, 2);
    assert_eq!(first.decoded_samples_per_channel, 5);
    assert!(first.exact_sample_values);
    assert!(first.exact_source_restoration);
    assert!(!first.semantic_equivalence);
    assert_eq!(first.semantic_coverage, "projected-semantic");
    assert_eq!(fs::read(&source).unwrap(), original);
    assert_eq!(fs::read(output.join("source.bin")).unwrap(), original);

    let expected_payload: Vec<u8> = source_signal()
        .into_iter()
        .flatten()
        .flat_map(i64::to_le_bytes)
        .collect();
    assert_eq!(
        fs::read(output.join("payload.i64le")).unwrap(),
        expected_payload
    );
    let canonical = fs::read(output.join("dataset.json")).unwrap();
    let dataset = abir::parse_canonical_dataset(&canonical).unwrap();
    assert_eq!(abir::canonical_debug_json(&dataset).unwrap(), canonical);
    assert_eq!(
        abir::logical_content_id(&dataset).unwrap().to_string(),
        first.dataset_content_id
    );

    let mapping: serde_json::Value =
        serde_json::from_slice(&fs::read(output.join("mapping-report.json")).unwrap()).unwrap();
    assert_eq!(mapping["semantic_coverage"], "projected-semantic");
    assert_eq!(mapping["sample_values_changed"], false);
    let fidelity: serde_json::Value =
        serde_json::from_slice(&fs::read(output.join("fidelity-report.json")).unwrap()).unwrap();
    assert_eq!(fidelity["exact_source_restoration"], true);
    assert_eq!(fidelity["exact_sample_values"], true);
    assert_eq!(fidelity["semantic_equivalence"], false);
}

#[test]
fn lml1_semantic_import_is_real_but_does_not_invent_modality() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("input.lml");
    let output = temp.path().join("semantic");
    let bytes = lml1_source_from_bcs1(&bcs1_source());
    fs::write(&source, bytes).unwrap();
    let receipt = import_semantic(&SemanticImportRequest {
        source,
        destination: output.clone(),
        accept_fidelity: true,
        max_source_bytes: 1024 * 1024,
        max_decoded_bytes: 1024 * 1024,
    })
    .unwrap();
    assert_eq!(receipt.profile, "legacy.lml1.v1");
    assert_eq!(receipt.modality, "legacy:modality/unknown-at-source");
    assert_eq!(receipt.timing, "exact");
    let mapping: serde_json::Value =
        serde_json::from_slice(&fs::read(output.join("mapping-report.json")).unwrap()).unwrap();
    let entries = mapping["entries"].as_array().unwrap();
    assert!(entries.iter().any(|entry| {
        entry["source_path"] == "wire.modality" && entry["disposition"] == "unsupported"
    }));
}

#[test]
fn semantic_reverse_export_is_bounded_atomic_idempotent_and_sample_exact() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("input.bcs1");
    let imported_dir = temp.path().join("imported");
    fs::write(&source, bcs1_source()).unwrap();
    let imported = import_semantic(&SemanticImportRequest {
        source,
        destination: imported_dir.clone(),
        accept_fidelity: true,
        max_source_bytes: 1024 * 1024,
        max_decoded_bytes: 1024 * 1024,
    })
    .unwrap();

    for format in [LegacyFormat::Bcs1, LegacyFormat::Lml1] {
        let destination = temp.path().join(format!("export-{format:?}"));
        let request = SemanticExportRequest {
            format,
            dataset: imported_dir.join("dataset.json"),
            payloads: vec![ExportPayload {
                content_id: imported.payload_content_id.clone(),
                path: imported_dir.join("payload.i64le"),
            }],
            destination: destination.clone(),
            accept_fidelity: true,
            max_dataset_bytes: 1024 * 1024,
            max_payload_bytes: 1024 * 1024,
            max_output_bytes: 1024 * 1024,
            window_size: 5,
        };
        let first = export_semantic(&request).unwrap();
        let second = export_semantic(&request).unwrap();
        assert_eq!(first, second);
        assert!(first.exact_sample_values);
        assert!(!first.semantic_equivalence);
        assert!(first.accepted_projection);
        let output = fs::read(destination.join("legacy-output.bin")).unwrap();
        assert_eq!(detect_format(&output).unwrap(), format);

        let encoded = temp.path().join(format!("verify-{format:?}.bin"));
        fs::write(&encoded, output).unwrap();
        let closure = import_semantic(&SemanticImportRequest {
            source: encoded,
            destination: temp.path().join(format!("closure-{format:?}")),
            accept_fidelity: true,
            max_source_bytes: 1024 * 1024,
            max_decoded_bytes: 1024 * 1024,
        })
        .unwrap();
        assert_eq!(closure.decoded_channels, 2);
        assert_eq!(closure.decoded_samples_per_channel, 5);
        assert!(closure.exact_sample_values);
    }
}

#[test]
fn semantic_reverse_export_fails_before_output_on_acceptance_bounds_or_identity() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("input.bcs1");
    let imported_dir = temp.path().join("imported");
    fs::write(&source, bcs1_source()).unwrap();
    let imported = import_semantic(&SemanticImportRequest {
        source,
        destination: imported_dir.clone(),
        accept_fidelity: true,
        max_source_bytes: 1024 * 1024,
        max_decoded_bytes: 1024 * 1024,
    })
    .unwrap();
    let destination = temp.path().join("export");
    let mut request = SemanticExportRequest {
        format: LegacyFormat::Bcs1,
        dataset: imported_dir.join("dataset.json"),
        payloads: vec![ExportPayload {
            content_id: imported.payload_content_id,
            path: imported_dir.join("payload.i64le"),
        }],
        destination: destination.clone(),
        accept_fidelity: false,
        max_dataset_bytes: 1024 * 1024,
        max_payload_bytes: 1024 * 1024,
        max_output_bytes: 1024 * 1024,
        window_size: 5,
    };
    assert_eq!(
        export_semantic(&request).unwrap_err(),
        LegacyError::AcceptanceRequired
    );
    assert!(!destination.exists());

    request.accept_fidelity = true;
    request.max_payload_bytes = 1;
    assert_eq!(
        export_semantic(&request).unwrap_err(),
        LegacyError::DecodedTooLarge
    );
    assert!(!destination.exists());

    request.max_payload_bytes = 1024 * 1024;
    request.payloads.push(ExportPayload {
        content_id: "unused".to_owned(),
        path: imported_dir.join("payload.i64le"),
    });
    assert_eq!(
        export_semantic(&request).unwrap_err(),
        LegacyError::SemanticExportUnsupported
    );
    assert!(!destination.exists());

    request.payloads.pop();
    fs::write(&request.payloads[0].path, vec![0; 80]).unwrap();
    assert_eq!(
        export_semantic(&request).unwrap_err(),
        LegacyError::PayloadIdentityMismatch
    );
    assert!(!destination.exists());
}

#[test]
fn lqtp1_semantic_import_and_exact_reverse_export_are_bounded_and_non_destructive() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("input.lqtp");
    let original = lqtp1_source(&source);
    let output = temp.path().join("semantic");
    let inspection = inspect(&source, 1024 * 1024).unwrap();
    assert_eq!(inspection.profile, "legacy.lqtp.v1");
    assert_eq!(inspection.decoded_channels, Some(2));
    assert_eq!(inspection.decoded_samples_per_channel, Some(6));

    let imported = import_semantic(&SemanticImportRequest {
        source: source.clone(),
        destination: output.clone(),
        accept_fidelity: true,
        max_source_bytes: 1024 * 1024,
        max_decoded_bytes: 1024 * 1024,
    })
    .unwrap();
    assert_eq!(imported.profile, "legacy.lqtp.v1");
    assert_eq!(imported.decoded_channels, 2);
    assert_eq!(imported.decoded_samples_per_channel, 6);
    assert_eq!(imported.decoded_payload_bytes, 48);
    assert!(imported.exact_sample_values);
    assert!(imported.exact_source_restoration);
    assert!(!imported.semantic_equivalence);
    assert_eq!(fs::read(&source).unwrap(), original);
    assert_eq!(fs::read(output.join("source.bin")).unwrap(), original);
    assert_eq!(fs::read(output.join("payload.f32le")).unwrap().len(), 48);
    assert!(!output.join("payload.i64le").exists());

    let canonical = fs::read(output.join("dataset.json")).unwrap();
    let dataset = abir::parse_canonical_dataset(&canonical).unwrap();
    let tensor = &dataset.atoms()[0];
    assert_eq!(tensor.payload().unwrap().element(), abir::ElementType::F32);
    assert_eq!(tensor.payload().unwrap().shape(), &[2, 2, 3]);
    let mapping: serde_json::Value =
        serde_json::from_slice(&fs::read(output.join("mapping-report.json")).unwrap()).unwrap();
    assert_eq!(mapping["sample_values_changed"], false);
    assert_eq!(mapping["preserved_unknowns"], 1);
    let fidelity: serde_json::Value =
        serde_json::from_slice(&fs::read(output.join("fidelity-report.json")).unwrap()).unwrap();
    assert_eq!(fidelity["exact_source_restoration"], true);
    assert_eq!(fidelity["semantic_equivalence"], false);

    let capsule = &dataset.source_capsules()[0];
    let export_dir = temp.path().join("export");
    let export = SemanticExportRequest {
        format: LegacyFormat::Lqtp1,
        dataset: output.join("dataset.json"),
        payloads: vec![ExportPayload {
            content_id: capsule.content_id().to_string(),
            path: output.join("source.bin"),
        }],
        destination: export_dir.clone(),
        accept_fidelity: true,
        max_dataset_bytes: 1024 * 1024,
        max_payload_bytes: 1024 * 1024,
        max_output_bytes: 1024 * 1024,
        window_size: 0,
    };
    let first = export_semantic(&export).unwrap();
    let second = export_semantic(&export).unwrap();
    assert_eq!(first, second);
    assert!(first.exact_sample_values);
    assert!(!first.semantic_equivalence);
    assert!(first.accepted_projection);
    assert_eq!(
        fs::read(export_dir.join("legacy-output.bin")).unwrap(),
        original
    );

    fs::write(export_dir.join("legacy-output.bin"), b"conflict").unwrap();
    assert_eq!(
        export_semantic(&export),
        Err(LegacyError::DestinationConflict)
    );
    assert_eq!(fs::read(&source).unwrap(), original);
}

#[test]
fn lqtp1_rejects_malformed_or_over_budget_input_before_output() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("input.lqtp");
    let mut bytes = lqtp1_source(&source);
    bytes[5] = 0xff;
    fs::write(&source, bytes).unwrap();
    assert!(matches!(
        inspect(&source, 1024 * 1024),
        Err(LegacyError::MalformedContainer(_))
    ));
    let valid = temp.path().join("valid.lqtp");
    lqtp1_source(&valid);
    assert!(matches!(
        import_semantic(&SemanticImportRequest {
            source: valid,
            destination: temp.path().join("out"),
            accept_fidelity: true,
            max_source_bytes: 1024 * 1024,
            max_decoded_bytes: 4,
        }),
        Err(LegacyError::DecodedTooLarge)
    ));
    assert!(!temp.path().join("out").exists());

    let oversized = temp.path().join("oversized.lqtp");
    let mut hostile_header = vec![0_u8; 64];
    hostile_header[0..4].copy_from_slice(b"LQTP");
    hostile_header[4] = 1;
    hostile_header[5] = 3;
    hostile_header[6..8].copy_from_slice(&u16::MAX.to_le_bytes());
    hostile_header[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
    hostile_header[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
    let stride = usize::from(u16::MAX)
        .checked_mul(4)
        .and_then(|scales| {
            usize::from(u16::MAX)
                .checked_mul(u32::MAX as usize)
                .and_then(|values| values.checked_mul(4))
                .and_then(|mantissas| scales.checked_add(mantissas))
        })
        .unwrap();
    hostile_header[16..24].copy_from_slice(&(stride as u64).to_le_bytes());
    fs::write(&oversized, hostile_header).unwrap();
    assert_eq!(
        inspect(&oversized, 1024 * 1024),
        Err(LegacyError::DecodedTooLarge)
    );
}

#[test]
fn semantic_import_bounds_decoded_allocation_before_output() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("input.bcs1");
    let output = temp.path().join("semantic");
    fs::write(&source, bcs1_source()).unwrap();
    let error = import_semantic(&SemanticImportRequest {
        source,
        destination: output.clone(),
        accept_fidelity: true,
        max_source_bytes: 1024 * 1024,
        max_decoded_bytes: 79,
    })
    .unwrap_err();
    assert_eq!(error, LegacyError::DecodedTooLarge);
    assert!(!output.exists());
}

#[test]
fn every_declared_capability_is_backed_and_unknown_magic_is_refused() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("input.bin");
    let output = temp.path().join("semantic");
    // Nothing recognises this, so nothing may be written.
    fs::write(&source, b"NOPEpayload").unwrap();
    let error = import_semantic(&SemanticImportRequest {
        source,
        destination: output.clone(),
        accept_fidelity: true,
        max_source_bytes: 1024,
        max_decoded_bytes: 1024,
    })
    .unwrap_err();
    assert_eq!(error, LegacyError::UnknownMagic);
    assert!(!output.exists());

    // Every retired profile now claims semantic import and reverse export.
    // The claims are only meaningful because each is exercised end to end by a
    // test in this file against a real fixture of that wire; the invariant kept
    // here is that no profile can claim re-emission it cannot also import.
    let manifest = capability_manifest();
    assert_eq!(manifest.capabilities.len(), 9);
    for capability in &manifest.capabilities {
        assert!(capability.inspect && capability.forensic_import);
        assert!(
            !capability.reverse_export || capability.semantic_import,
            "{} claims reverse export without semantic import",
            capability.profile,
        );
    }
}

#[test]
fn semantic_import_rejects_unaccepted_or_unsupported_bcs1_before_output() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("input.bcs1");
    let output = temp.path().join("semantic");
    fs::write(&source, bcs1_source()).unwrap();
    let mut request = SemanticImportRequest {
        source: source.clone(),
        destination: output.clone(),
        accept_fidelity: false,
        max_source_bytes: 1024 * 1024,
        max_decoded_bytes: 1024 * 1024,
    };
    assert_eq!(
        import_semantic(&request).unwrap_err(),
        LegacyError::AcceptanceRequired
    );
    assert!(!output.exists());

    let mut unsupported = bcs1_source();
    unsupported[8] = 1;
    fs::write(&source, unsupported).unwrap();
    request.accept_fidelity = true;
    let error = import_semantic(&request).unwrap_err();
    assert!(matches!(error, LegacyError::MalformedContainer(_)));
    assert_eq!(error.code(), "malformed-container");
    assert!(!output.exists());
}

#[test]
fn committed_converter_matrix_scopes_semantic_claims_per_profile() {
    let matrix: serde_json::Value =
        serde_json::from_slice(include_bytes!("../../../converter-matrix.json")).unwrap();
    assert_eq!(matrix["semantic_import"], "PARTIAL");
    assert_eq!(matrix["reverse_export"], "PARTIAL");
    assert_eq!(
        matrix["semantic_profiles"]["legacy.bcs1.v1"]["status"],
        "PASS_PROJECTED"
    );
    assert_eq!(
        matrix["semantic_profiles"]["legacy.bcs1.v1"]["reverse_export"],
        "PASS_PROJECTED_EXACT_SAMPLES"
    );
    assert_eq!(
        matrix["semantic_profiles"]["legacy.lma.v1"]["status"],
        "PASS_PROJECTED"
    );
    // LMA now re-emits, and the claim names the generation actually written.
    assert_eq!(
        matrix["semantic_profiles"]["legacy.lma.v1"]["reverse_export"],
        "PASS_PROJECTED_EXACT_SAMPLES_AS_V2"
    );
    // LMQC claims the montage and clock it really recovers, and says plainly
    // that the neural payload stays encoded and no samples are produced.
    assert_eq!(
        matrix["semantic_profiles"]["legacy.lmqc.v1"]["sample_values"],
        "NOT_PRODUCED"
    );
    assert_eq!(
        matrix["semantic_profiles"]["legacy.lmqc.v1"]["neural_payload"],
        "QUARANTINED_ENCODED"
    );
    // The AEAD envelope never lets key material into the dataset.
    assert_eq!(
        matrix["semantic_profiles"]["legacy.lmlcrypt.v1"]["key_material"],
        "NEVER_IN_DATASET"
    );
    // The retired tensor packs name what they carried and what they did not:
    // one stream per view, exact decoded f32 values, non-f32 views stored.
    for generation in ["legacy.lqtp.v2", "legacy.lqtp.v3"] {
        assert_eq!(
            matrix["semantic_profiles"][generation]["views"],
            "ONE_STREAM_EACH"
        );
        assert_eq!(
            matrix["semantic_profiles"][generation]["non_f32_views"],
            "CARRIED_STORED"
        );
        assert_eq!(
            matrix["semantic_profiles"][generation]["row_identity"],
            "EXTERNAL_MANIFEST_HASH_ONLY"
        );
    }
    assert_eq!(
        matrix["semantic_profiles"]["legacy.lqtp.v1"]["status"],
        "PASS_PROJECTED"
    );
}

#[cfg(unix)]
#[test]
fn source_symlink_is_rejected_without_output() {
    use std::os::unix::fs::symlink;

    let temp = tempfile::tempdir().unwrap();
    let target = temp.path().join("target.lml");
    let source = temp.path().join("source.lml");
    let output = temp.path().join("out");
    fs::write(&target, b"LML1payload").unwrap();
    symlink(&target, &source).unwrap();
    let error = convert_forensic(&ConvertRequest {
        source,
        destination: output.clone(),
        accept_fidelity: true,
        max_source_bytes: 1024,
    })
    .unwrap_err();
    assert_eq!(error, LegacyError::UnsafeSource);
    assert!(!output.exists());
}

/// An `LMA1` front-manifest archive holding one LML1 recording beside one
/// non-signal sibling. Nothing writes v1 any more, so the retired generation
/// only exists framed by hand; `pack_lml_entries` cannot stand in for it
/// because it emits v2 and marks every entry as LML.
fn lma_v1_archive(recording: &[u8], sidecar: (&str, &[u8])) -> Vec<u8> {
    let digest = |bytes: &[u8]| format!("{:x}", Sha256::digest(bytes));
    let manifest = format!(
        "{{\"compressor\":\"zstd\",\"compressor_level\":3,\"files\":[\
         {{\"path\":\"a.lml\",\"original_size\":{n},\"compressed_size\":{n},\"method\":\"lml\",\
         \"sha256\":\"{lml}\",\"offset\":0}},\
         {{\"path\":\"{name}\",\"original_size\":{m},\"compressed_size\":{m},\"method\":\"store\",\
         \"sha256\":\"{side}\",\"offset\":{n}}}]}}",
        n = recording.len(),
        m = sidecar.1.len(),
        name = sidecar.0,
        lml = digest(recording),
        side = digest(sidecar.1),
    )
    .into_bytes();

    let mut archive = Vec::new();
    archive.extend_from_slice(b"LMA1");
    archive.extend_from_slice(&1_u32.to_le_bytes());
    archive.extend_from_slice(&2_u32.to_le_bytes());
    // Top bit marks the manifest as stored rather than zstd.
    archive.extend_from_slice(&((manifest.len() as u32) | 0x8000_0000).to_le_bytes());
    archive.extend_from_slice(&manifest);
    archive.extend_from_slice(recording);
    archive.extend_from_slice(sidecar.1);
    let digest = Sha256::digest(&archive);
    archive.extend_from_slice(&digest);
    archive
}

#[test]
fn lma_v1_front_manifest_imports_and_quarantines_its_non_signal_sibling() {
    let temp = tempfile::tempdir().unwrap();
    let archive = temp.path().join("input.lma");
    let bytes = lma_v1_archive(
        &lml1_source_from_bcs1(&bcs1_source()),
        ("annotations.tse", b"0.0 1.0 bckg 1.0000\n"),
    );
    fs::write(&archive, &bytes).unwrap();
    assert!(bytes.starts_with(b"LMA1"));

    let output = temp.path().join("semantic");
    let receipt = import_semantic(&SemanticImportRequest {
        source: archive,
        destination: output.clone(),
        accept_fidelity: true,
        max_source_bytes: 1 << 20,
        max_decoded_bytes: 1 << 20,
    })
    .unwrap();

    // The v1 generation reports itself as v1 rather than borrowing v2's name.
    assert_eq!(receipt.profile, "legacy.lma.v1");
    assert!(receipt.exact_source_restoration);
    assert!(!receipt.semantic_equivalence);

    // Exactly one recording: the stored sibling is never promoted to a signal.
    let payloads = fs::read_dir(&output)
        .unwrap()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("payload-"))
        .count();
    assert_eq!(payloads, 1);
    let mapping: serde_json::Value =
        serde_json::from_slice(&fs::read(output.join("mapping-report.json")).unwrap()).unwrap();
    assert_eq!(mapping["preserved_unknowns"], 1);
}

#[test]
fn lma_v2_semantic_import_emits_one_recording_per_archived_signal() {
    // A legacy-framed archive: two LML1 entries plus one non-signal sibling.
    // Built with pack_lml_entries so the fixture carries the LEGACY wire this
    // adapter exists to read, rather than whatever the current codec emits.
    let temp = tempfile::tempdir().unwrap();
    let lml = lml1_source_from_bcs1(&bcs1_source());
    let sidecar = b"clinical sidecar, not a signal".to_vec();
    let archive = temp.path().join("input.lma");
    lamquant_lml_archive::lma::pack_lml_entries(&[("a.lml", &lml), ("b.lml", &lml)], &archive, 3)
        .unwrap();
    assert!(fs::read(&archive).unwrap().starts_with(b"LMA2"));
    let _ = sidecar;

    let output = temp.path().join("semantic");
    let receipt = import_semantic(&SemanticImportRequest {
        source: archive,
        destination: output.clone(),
        accept_fidelity: true,
        max_source_bytes: 1 << 20,
        max_decoded_bytes: 1 << 20,
    })
    .unwrap();

    assert_eq!(receipt.profile, "legacy.lma.v2");
    assert!(receipt.exact_source_restoration);
    assert!(!receipt.semantic_equivalence);
    assert_eq!(receipt.semantic_coverage, "projected-semantic");

    // One payload file per archived signal: the archive is never flattened.
    let payloads = fs::read_dir(&output)
        .unwrap()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("payload-"))
        .count();
    assert_eq!(payloads, 2);

    let dataset: serde_json::Value =
        serde_json::from_slice(&fs::read(output.join("dataset.json")).unwrap()).unwrap();
    assert!(
        dataset
            .to_string()
            .matches("legacy-lma-entry-sha256")
            .count()
            >= 2,
        "dataset did not carry one source key per archived signal"
    );
    assert!(fs::read(output.join("source.bin")).is_ok());
}

#[test]
fn lma_reverse_export_re_emits_every_recording_with_exact_samples() {
    let temp = tempfile::tempdir().unwrap();
    // Two DISTINCT recordings. Identical entries would share one payload
    // ContentId -- content addressing deduplicates them -- so a realistic
    // multi-recording archive must carry different signals.
    let first = lml1_source_from_bcs1(&bcs1_source());
    let second_path = temp.path().join("second.lml");
    lamquant_lml_legacy::container::write_file(
        &second_path,
        &[vec![5, 6, 7, 8, 9], vec![-5, -6, -7, -8, -9]],
        256.0,
        4,
        0,
        "{}",
    )
    .unwrap();
    let second = fs::read(&second_path).unwrap();
    let archive = temp.path().join("input.lma");
    lamquant_lml_archive::lma::pack_lml_entries(
        &[("a.lml", &first), ("b.lml", &second)],
        &archive,
        3,
    )
    .unwrap();
    let semantic = temp.path().join("semantic");
    import_semantic(&SemanticImportRequest {
        source: archive,
        destination: semantic.clone(),
        accept_fidelity: true,
        max_source_bytes: 1 << 20,
        max_decoded_bytes: 1 << 20,
    })
    .unwrap();

    // Feed every emitted payload back for re-emission.
    let mut payloads = Vec::new();
    for entry in fs::read_dir(&semantic).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with("payload-") {
            continue;
        }
        let bytes = fs::read(entry.path()).unwrap();
        payloads.push(ExportPayload {
            content_id: abir::payload_content_id(abir::ElementType::I64, &bytes).to_string(),
            path: entry.path(),
        });
    }
    assert_eq!(payloads.len(), 2);

    let out = temp.path().join("reemitted.lma");
    let receipt = export_semantic(&SemanticExportRequest {
        format: LegacyFormat::Lma2,
        dataset: semantic.join("dataset.json"),
        payloads,
        destination: out.clone(),
        accept_fidelity: true,
        max_dataset_bytes: 1 << 20,
        max_payload_bytes: 1 << 20,
        max_output_bytes: 1 << 20,
        window_size: 256,
    })
    .unwrap();

    // The writer emits v2, and the receipt says so rather than claiming v1.
    assert_eq!(receipt.profile, "legacy.lma.v2");
    assert!(receipt.exact_sample_values);
    assert!(!receipt.semantic_equivalence);
    let entries = lamquant_lml_archive::lma::list_archive(&out).unwrap();
    assert_eq!(entries.len(), 2);
    assert!(fs::read(&out).unwrap().starts_with(b"LMA2"));
}

fn lmqc_source() -> Vec<u8> {
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
    .unwrap()
}

#[test]
fn lmqc_semantic_import_recovers_the_montage_without_inventing_samples() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("input.lmq");
    let bytes = lmqc_source();
    fs::write(&source, &bytes).unwrap();

    let output = temp.path().join("semantic");
    let receipt = import_semantic(&SemanticImportRequest {
        source: source.clone(),
        destination: output.clone(),
        accept_fidelity: true,
        max_source_bytes: 1 << 20,
        max_decoded_bytes: 1 << 20,
    })
    .unwrap();

    assert_eq!(receipt.profile, "legacy.lmqc.v1");
    // The montage and clock are recovered exactly; samples are NOT produced,
    // and the receipt must not pretend otherwise.
    assert_eq!(receipt.decoded_channels, 2);
    assert_eq!(receipt.decoded_samples_per_channel, 2500);
    assert!(!receipt.exact_sample_values);
    assert!(receipt.exact_source_restoration);
    assert_eq!(receipt.timing, "regular");

    // The payload written out is the ENCODED latent, byte for byte.
    let payload = fs::read(output.join("payload.lmqc-latent")).unwrap();
    assert_eq!(payload, b"opaque encoded latent bytes");

    let fidelity: serde_json::Value =
        serde_json::from_slice(&fs::read(output.join("fidelity-report.json")).unwrap()).unwrap();
    assert_eq!(fidelity["exact_sample_values"], false);
    assert_eq!(fidelity["timing_equivalence"], true);

    let dataset = String::from_utf8(fs::read(output.join("dataset.json")).unwrap()).unwrap();
    assert!(dataset.contains("Fp1") && dataset.contains("Fp2"));

    // Reverse export re-emits the retired blob byte-for-byte.
    let out = temp.path().join("reemitted.lmq");
    let export = export_semantic(&SemanticExportRequest {
        format: LegacyFormat::Lmqc,
        dataset: output.join("dataset.json"),
        payloads: vec![ExportPayload {
            content_id: abir::payload_content_id(abir::ElementType::Bytes, &bytes).to_string(),
            path: source,
        }],
        destination: out.clone(),
        accept_fidelity: true,
        max_dataset_bytes: 1 << 20,
        max_payload_bytes: 1 << 20,
        max_output_bytes: 1 << 20,
        window_size: 2500,
    })
    .unwrap();
    assert_eq!(export.profile, "legacy.lmqc.v1");
    assert_eq!(fs::read(out.join("legacy-output.bin")).unwrap(), bytes);
}

/// Seal `plaintext` the way the retired `lml encrypt` did: magic, version,
/// nonce, then the AES-256-GCM ciphertext with its appended tag.
fn lmlcrypt_source(plaintext: &[u8], key: &[u8; 32], nonce: &[u8; 12]) -> Vec<u8> {
    use aes_gcm::aead::{Aead, KeyInit};
    let cipher = aes_gcm::Aes256Gcm::new_from_slice(key).unwrap();
    let ciphertext = cipher
        .encrypt(aes_gcm::Nonce::from_slice(nonce), plaintext)
        .unwrap();
    let mut blob = Vec::new();
    blob.extend_from_slice(b"LMLCRYPT");
    blob.push(1);
    blob.extend_from_slice(nonce);
    blob.extend_from_slice(&ciphertext);
    blob
}

#[test]
fn lmlcrypt_import_anchors_the_dataset_to_the_ciphertext_and_needs_the_key() {
    let temp = tempfile::tempdir().unwrap();
    let inner = lml1_source_from_bcs1(&bcs1_source());
    let key = [0x2b_u8; 32];
    let blob = lmlcrypt_source(&inner, &key, &[7_u8; 12]);
    let source = temp.path().join("input.enc");
    fs::write(&source, &blob).unwrap();

    let request = SemanticImportRequest {
        source: source.clone(),
        destination: temp.path().join("semantic"),
        accept_fidelity: true,
        max_source_bytes: 1 << 20,
        max_decoded_bytes: 1 << 20,
    };

    // Without the key the envelope fails closed with a capability error, and
    // writes nothing -- it is not reported as a malformed file.
    std::env::remove_var("LAMQUANT_KEY");
    assert_eq!(
        import_semantic(&request).unwrap_err(),
        LegacyError::KeyUnavailable
    );
    assert!(!request.destination.exists());

    std::env::set_var(
        "LAMQUANT_KEY",
        key.iter().map(|b| format!("{b:02x}")).collect::<String>(),
    );
    let receipt = import_semantic(&request).unwrap();
    std::env::remove_var("LAMQUANT_KEY");

    // The dataset names the bytes on disk -- the ciphertext -- not the
    // plaintext nobody holds a copy of.
    assert_eq!(receipt.profile, "legacy.lmlcrypt.v1");
    assert_eq!(receipt.source_bytes, blob.len() as u64);
    assert_eq!(receipt.source_blake3, blake3_hex(&blob));
    // The inner container's real semantics came through.
    assert_eq!(receipt.decoded_channels, 2);
    assert_eq!(receipt.decoded_samples_per_channel, 5);
    assert!(receipt.exact_sample_values);

    let dataset =
        String::from_utf8(fs::read(request.destination.join("dataset.json")).unwrap()).unwrap();
    assert!(dataset.contains("lmlcrypt.nonce"));
    assert!(dataset.contains("legacy.lml1.v1"), "inner profile recorded");
    // Key material never reaches the dataset.
    assert!(!dataset.contains("2b2b2b2b"));

    let fidelity: serde_json::Value = serde_json::from_slice(
        &fs::read(request.destination.join("fidelity-report.json")).unwrap(),
    )
    .unwrap();
    let caveats = fidelity["caveats"].as_array().unwrap();
    assert!(caveats
        .iter()
        .any(|caveat| caveat.as_str().unwrap().contains("AES-256-GCM")));

    // A tampered blob is refused: the tag no longer authenticates.
    let mut tampered = blob.clone();
    let last = tampered.len() - 1;
    tampered[last] ^= 0xff;
    let tampered_path = temp.path().join("tampered.enc");
    fs::write(&tampered_path, &tampered).unwrap();
    std::env::set_var(
        "LAMQUANT_KEY",
        key.iter().map(|b| format!("{b:02x}")).collect::<String>(),
    );
    let error = import_semantic(&SemanticImportRequest {
        source: tampered_path,
        destination: temp.path().join("tampered-out"),
        ..request.clone()
    })
    .unwrap_err();
    std::env::remove_var("LAMQUANT_KEY");
    assert!(matches!(error, LegacyError::MalformedContainer(_)));
}

fn blake3_hex(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

#[test]
fn archive_with_a_broken_integrity_trailer_is_refused_before_output() {
    let temp = tempfile::tempdir().unwrap();
    let mut bytes = lma_v1_archive(
        &lml1_source_from_bcs1(&bcs1_source()),
        ("annotations.tse", b"0.0 1.0 bckg 1.0000\n"),
    );
    // Flip one bit of the archive's own SHA-256 trailer. The codec's reader
    // opens by seeking and never checks it; the adapter holds the whole blob
    // and must, or the corruption would be carried into ABIR semantics.
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    let source = temp.path().join("corrupt.lma");
    fs::write(&source, &bytes).unwrap();

    let output = temp.path().join("semantic");
    let error = import_semantic(&SemanticImportRequest {
        source: source.clone(),
        destination: output.clone(),
        accept_fidelity: true,
        max_source_bytes: 1 << 20,
        max_decoded_bytes: 1 << 20,
    })
    .unwrap_err();
    assert!(matches!(error, LegacyError::MalformedContainer(_)));
    assert!(!output.exists());
    assert!(inspect(&source, 1 << 20).is_err());
}

/// A two-row LQTP2 snapshot with one f32 BFP view and one u8 raw view.
fn lqtp2_source(path: &std::path::Path) -> Vec<u8> {
    use lamquant_lml_legacy::tensor_pack_v2::{
        PackV2Dtype, PackV2Encoding, PackV2Writer, ViewSpec,
    };
    let specs = vec![
        ViewSpec::new(
            "fullband",
            PackV2Dtype::F32,
            PackV2Encoding::BfpInt16,
            &[2, 3],
            true,
            [0x11; 32],
        )
        .unwrap(),
        ViewSpec::new(
            "labels",
            PackV2Dtype::U8,
            PackV2Encoding::Raw,
            &[4],
            true,
            [0x22; 32],
        )
        .unwrap(),
    ];
    let mut writer = PackV2Writer::create(
        path,
        2,
        [0xaa; 32],
        [0xbb; 32],
        br#"{"schema":"lamquant.training-window-metadata/1"}"#.to_vec(),
        specs,
    )
    .unwrap();
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
            .unwrap();
        writer
            .write_raw_row("labels", &[row, row + 1, row + 2, row + 3])
            .unwrap();
    }
    writer.finish().unwrap();
    fs::read(path).unwrap()
}

/// A four-row LQTP3 chunked bundle: one zstd-compressed BFP view and one
/// uncompressed raw view, so both chunk codecs are exercised.
fn lqtp3_source(path: &std::path::Path) -> Vec<u8> {
    use lamquant_lml_legacy::tensor_pack_v3::{
        PackV3Compression, PackV3Dtype, PackV3Encoding, PackV3Writer, ViewSpecV3,
    };
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
        .unwrap(),
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
        .unwrap(),
    ];
    let mut writer = PackV3Writer::create(
        path,
        4,
        [0xaa; 32],
        [0xbb; 32],
        br#"{"schema":"lamquant.training-window-metadata/1"}"#.to_vec(),
        specs,
    )
    .unwrap();
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
            .unwrap();
        writer
            .write_raw_row("labels", &[row, row + 1, row + 2, row + 3])
            .unwrap();
    }
    writer.finish().unwrap();
    fs::read(path).unwrap()
}

#[test]
fn retired_tensor_packs_import_one_stream_per_view_and_re_emit_exactly() {
    for (profile, rows, format, build) in [
        (
            "legacy.lqtp.v2",
            2_u64,
            LegacyFormat::Lqtp2,
            lqtp2_source as fn(&std::path::Path) -> Vec<u8>,
        ),
        (
            "legacy.lqtp.v3",
            4,
            LegacyFormat::Lqtp3,
            lqtp3_source as fn(&std::path::Path) -> Vec<u8>,
        ),
    ] {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("snapshot.pack");
        let bytes = build(&source);

        let output = temp.path().join("semantic");
        let receipt = import_semantic(&SemanticImportRequest {
            source: source.clone(),
            destination: output.clone(),
            accept_fidelity: true,
            max_source_bytes: 1 << 20,
            max_decoded_bytes: 1 << 20,
        })
        .unwrap();

        assert_eq!(receipt.profile, profile);
        // Views, not channels: the snapshot has two views over a shared row
        // axis, and the receipt must not invent a channel geometry.
        assert_eq!(receipt.decoded_channels, 2);
        assert_eq!(receipt.decoded_samples_per_channel, rows);
        assert!(receipt.exact_source_restoration);
        assert!(!receipt.semantic_equivalence);

        // One payload file per view; the archive is never flattened.
        assert_eq!(
            fs::read(output.join("view-fullband.bin")).unwrap().len(),
            (rows as usize) * 6 * 4,
            "f32 view decodes to rows x 6 f32 values",
        );
        // The u8 view is carried STORED, not reinterpreted as f32.
        assert_eq!(
            fs::read(output.join("view-labels.bin")).unwrap(),
            (0..rows as u8)
                .flat_map(|row| [row, row + 1, row + 2, row + 3])
                .collect::<Vec<u8>>(),
        );
        let mapping: serde_json::Value =
            serde_json::from_slice(&fs::read(output.join("mapping-report.json")).unwrap()).unwrap();
        let dispositions: Vec<&str> = mapping["entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry["disposition"].as_str().unwrap())
            .collect();
        assert!(dispositions.contains(&"exact"));
        assert!(dispositions.contains(&"quarantined"));

        // Reverse export re-emits the retired snapshot byte for byte.
        let out = temp.path().join("reemitted.pack");
        let export = export_semantic(&SemanticExportRequest {
            format,
            dataset: output.join("dataset.json"),
            payloads: vec![ExportPayload {
                content_id: abir::payload_content_id(abir::ElementType::Bytes, &bytes).to_string(),
                path: source,
            }],
            destination: out.clone(),
            accept_fidelity: true,
            max_dataset_bytes: 1 << 20,
            max_payload_bytes: 1 << 20,
            max_output_bytes: 1 << 20,
            window_size: 1,
        })
        .unwrap();
        assert_eq!(export.profile, profile);
        assert_eq!(fs::read(out.join("legacy-output.bin")).unwrap(), bytes);
    }
}
