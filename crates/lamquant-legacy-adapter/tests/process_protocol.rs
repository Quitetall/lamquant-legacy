use lamquant_legacy_adapter::{
    capability_manifest, convert_forensic, detect_format, ConvertRequest, LegacyError, LegacyFormat,
};
use std::fs;

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
