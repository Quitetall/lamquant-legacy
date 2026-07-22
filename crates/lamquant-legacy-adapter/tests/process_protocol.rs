use lamquant_legacy_adapter::{
    capability_manifest, convert_forensic, detect_format, import_semantic, ConvertRequest,
    LegacyError, LegacyFormat, SemanticImportRequest,
};
use lamquant_legacy_ir::{Bcs1Header, BCS1_VERSION_MAJOR, BCS1_VERSION_MINOR, CODEC_LML_53};
use std::fs;

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
        (b"LQTP\x02payload", LegacyFormat::Lqtp2),
        (b"LQTP\x03payload", LegacyFormat::Lqtp3),
    ];
    for (bytes, expected) in cases {
        assert_eq!(detect_format(bytes).unwrap(), *expected);
    }
}

#[test]
fn forensic_conversion_is_exact_non_destructive_and_idempotent() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("input.lml");
    let output = temp.path().join("output");
    let original = b"LML1\x01\x02\x03";
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
fn unsupported_profile_does_not_overstate_semantic_import() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("input.lma");
    let output = temp.path().join("semantic");
    fs::write(&source, b"LMA1payload").unwrap();
    let error = import_semantic(&SemanticImportRequest {
        source,
        destination: output.clone(),
        accept_fidelity: true,
        max_source_bytes: 1024,
        max_decoded_bytes: 1024,
    })
    .unwrap_err();
    assert_eq!(error, LegacyError::SemanticImportUnsupported);
    assert!(!output.exists());
    let manifest = capability_manifest();
    assert!(
        manifest
            .capabilities
            .iter()
            .find(|value| value.profile == "legacy.bcs1.v1")
            .unwrap()
            .semantic_import
    );
    assert!(
        !manifest
            .capabilities
            .iter()
            .find(|value| value.profile == "legacy.lma.v1")
            .unwrap()
            .semantic_import
    );
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
    assert_eq!(
        matrix["semantic_profiles"]["legacy.bcs1.v1"]["status"],
        "PASS_PROJECTED"
    );
    assert_eq!(
        matrix["semantic_profiles"]["legacy.lma.v1"]["status"],
        "NOT_CLAIMED"
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
