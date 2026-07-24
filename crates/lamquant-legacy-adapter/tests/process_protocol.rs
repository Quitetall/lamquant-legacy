use lamquant_legacy_adapter::{
    capability_manifest, convert_forensic, detect_format, export_semantic, import_semantic,
    inspect, ConvertRequest, ExportPayload, LegacyError, LegacyFormat, SemanticExportRequest,
    SemanticImportRequest,
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
        (b"LQTP\x02payload", LegacyFormat::Lqtp2),
        (b"LQTP\x03payload", LegacyFormat::Lqtp3),
    ];
    for (bytes, expected) in cases {
        assert_eq!(detect_format(bytes).unwrap(), *expected);
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
fn unsupported_profile_does_not_overstate_semantic_import() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("input.lma");
    let output = temp.path().join("semantic");
    // Both LMA generations now import semantically, so this fixture must use a
    // profile that is still genuinely unsupported for the assertion to mean
    // anything.
    fs::write(&source, b"LMQCpayload").unwrap();
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
            .find(|value| value.profile == "legacy.lmqc.v1")
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
    // LMA v1 imports semantically but cannot yet re-emit, so it must NOT carry
    // a reverse_export claim.
    assert!(matrix["semantic_profiles"]["legacy.lma.v1"]
        .get("reverse_export")
        .is_none());
    assert_eq!(
        matrix["semantic_profiles"]["legacy.lmqc.v1"]["status"],
        "NOT_CLAIMED"
    );
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

#[test]
fn lma_v1_semantic_import_emits_one_recording_per_archived_signal() {
    // A real LMA v1 archive: two synthetic EDFs, which the packer encodes as
    // `Method::Lml` entries, plus one non-signal sibling that must NOT become a
    // recording.
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("tree");
    fs::create_dir(&input).unwrap();
    let samples: Vec<i16> = (0..256).map(|value| (value % 97) as i16).collect();
    for name in ["a.edf", "b.edf"] {
        fs::write(
            input.join(name),
            lamquant_lml_archive::ingest::edf_synth::synth_single_channel_edf(&samples, 256.0),
        )
        .unwrap();
    }
    fs::write(input.join("notes.txt"), b"clinical sidecar, not a signal").unwrap();

    let archive = temp.path().join("input.lma");
    lamquant_lml_archive::lma::pack_archive(&input, &archive, 3, false, None).unwrap();
    // `pack_archive` emits the v2 streaming layout; the same reader resolves
    // v1, so both generations share the importer.
    assert!(fs::read(&archive).unwrap().starts_with(b"LMA2"));

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
    // Two signals in, two recordings out — the archive is never flattened.
    assert_eq!(receipt.decoded_channels, 2);
    assert!(receipt.exact_source_restoration);
    assert!(!receipt.semantic_equivalence);
    assert_eq!(receipt.semantic_coverage, "projected-semantic");

    // One payload file per archived signal, and none for the text sibling.
    let payloads = fs::read_dir(&output)
        .unwrap()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("payload-"))
        .count();
    assert_eq!(payloads, 2);

    let dataset: serde_json::Value =
        serde_json::from_slice(&fs::read(output.join("dataset.json")).unwrap()).unwrap();
    let text = dataset.to_string();
    // Both archived signals must survive as distinct recordings, each bound to
    // its own archive-entry digest.
    assert!(
        text.matches("legacy-lma-entry-sha256").count() >= 2,
        "dataset did not carry one source key per archived signal"
    );
    assert!(fs::read(output.join("source.bin")).is_ok());
}
