//! Schema parity test — validates the canonical JSON Schema against fixtures
//! and exercises round-trips through Rust serialization.
//!
//! Per `specs/ui-parity.md`, the OpEvent wire format is canonical at
//! `specs/op-events.schema.json`. Rust, Python, and TS all generate or
//! validate against that file. This test:
//!
//!   1. Loads the canonical schema.
//!   2. Round-trips one fixture per OpEvent variant through the Rust enum.
//!   3. Validates the serialized output against the JSON Schema.
//!
//! Drift (e.g. a Rust struct field added without updating the schema) is
//! caught here. Python and TS validation are exercised by their own
//! per-language test scripts; this Rust test is the gate for `cargo test`.

use lamquant_ops_legacy::OpEvent;

const SCHEMA_PATH: &str = "specs/op-events.schema.json";

fn load_schema() -> serde_json::Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(SCHEMA_PATH);
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {}", path.display(), e))
}

fn assert_valid(validator: &jsonschema::Validator, value: &serde_json::Value, label: &str) {
    let errors: Vec<_> = validator.iter_errors(value).collect();
    if !errors.is_empty() {
        for e in &errors {
            eprintln!("  - {}: {}", label, e);
        }
        panic!(
            "schema validation failed for {}: {} error(s)",
            label,
            errors.len()
        );
    }
}

#[test]
fn schema_parses() {
    let schema = load_schema();
    let _ = jsonschema::validator_for(&schema).unwrap_or_else(|e| panic!("compile schema: {}", e));
}

#[test]
fn rust_emit_matches_schema() {
    let schema = load_schema();
    let validator =
        jsonschema::validator_for(&schema).unwrap_or_else(|e| panic!("compile schema: {}", e));

    let fixtures = vec![
        OpEvent::Started {
            ts_ms: 1_730_000_000_000,
            op_id: "encode".into(),
            total: Some(42),
        },
        OpEvent::Started {
            ts_ms: 1_730_000_000_000,
            op_id: "info".into(),
            total: None,
        },
        OpEvent::Progress {
            ts_ms: 1_730_000_000_500,
            current: 5,
            total: 42,
            message: "file 5/42".into(),
        },
        OpEvent::FileDone {
            ts_ms: 1_730_000_001_000,
            path: "data/a.lml".into(),
            success: true,
            ms: 120,
            cr: Some(2.5),
            bytes_in: Some(1_048_576),
            bytes_out: Some(419_430),
            samples: Some(5_250_000),
            duration_s: Some(1000.0),
            n_channels: Some(21),
            sample_rate: Some(250.0),
            sha256: Some("abcd1234".into()),
            n_windows: Some(100),
        },
        OpEvent::FileDone {
            ts_ms: 1_730_000_001_000,
            path: "data/b.lml".into(),
            success: false,
            ms: 7,
            cr: None,
            bytes_in: None,
            bytes_out: None,
            samples: None,
            duration_s: None,
            n_channels: None,
            sample_rate: None,
            sha256: None,
            n_windows: None,
        },
        OpEvent::Done {
            ts_ms: 1_730_000_002_000,
            message: "42 files in 9.4s".into(),
        },
        OpEvent::Error {
            ts_ms: 1_730_000_002_500,
            message: "out of disk space".into(),
        },
        OpEvent::Log {
            ts_ms: 1_730_000_002_700,
            message: "[stderr] WARN: mains pickup detected".into(),
        },
    ];

    for ev in &fixtures {
        let line = ev.to_json_line();
        let value: serde_json::Value = serde_json::from_str(&line)
            .unwrap_or_else(|e| panic!("parse own emission: {} ({})", e, line));
        assert_valid(&validator, &value, &line);
    }
}

#[test]
fn rust_round_trip_preserves_variants() {
    // Every variant must round-trip without losing fields. If you add a
    // field to OpEvent without updating the schema (or vice versa), the
    // schema validator above will catch the schema side; this test catches
    // the Rust side.
    let cases: Vec<OpEvent> = vec![
        OpEvent::Started {
            ts_ms: 1,
            op_id: "x".into(),
            total: None,
        },
        OpEvent::Started {
            ts_ms: 2,
            op_id: "y".into(),
            total: Some(10),
        },
        OpEvent::Progress {
            ts_ms: 3,
            current: 1,
            total: 10,
            message: "m".into(),
        },
        OpEvent::FileDone {
            ts_ms: 4,
            path: "p".into(),
            success: true,
            ms: 5,
            cr: Some(1.5),
            bytes_in: None,
            bytes_out: None,
            samples: None,
            duration_s: None,
            n_channels: None,
            sample_rate: None,
            sha256: None,
            n_windows: None,
        },
        OpEvent::FileDone {
            ts_ms: 5,
            path: "p".into(),
            success: false,
            ms: 0,
            cr: None,
            bytes_in: None,
            bytes_out: None,
            samples: None,
            duration_s: None,
            n_channels: None,
            sample_rate: None,
            sha256: None,
            n_windows: None,
        },
        OpEvent::Done {
            ts_ms: 6,
            message: "done".into(),
        },
        OpEvent::Error {
            ts_ms: 7,
            message: "fail".into(),
        },
        OpEvent::Log {
            ts_ms: 8,
            message: "log".into(),
        },
    ];
    for ev in cases {
        let line = ev.to_json_line();
        let back = OpEvent::from_json_line(&line)
            .unwrap_or_else(|e| panic!("round-trip parse failed: {} (line: {})", e, line));
        // Re-serialize and compare to ensure no fields silently dropped.
        let line2 = back.to_json_line();
        assert_eq!(line, line2, "round-trip mismatch");
    }
}

/// Sentinel test: tests/fixtures/op-events-sample.jsonl contains a hand-
/// written sample line per variant. Every line must validate against the
/// schema AND parse into our Rust enum. This pins the wire format so a
/// future schema edit can't silently break Python/TS readers.
#[test]
fn fixture_jsonl_validates() {
    let schema = load_schema();
    let validator =
        jsonschema::validator_for(&schema).unwrap_or_else(|e| panic!("compile schema: {}", e));

    let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/op-events-sample.jsonl");
    let text = std::fs::read_to_string(&fixture_path)
        .unwrap_or_else(|e| panic!("read {}: {}", fixture_path.display(), e));

    let mut count = 0;
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("line {}: invalid JSON: {}", i + 1, e));
        assert_valid(&validator, &value, &format!("line {}", i + 1));
        OpEvent::from_json_line(line)
            .unwrap_or_else(|e| panic!("line {}: Rust deserialize: {}", i + 1, e));
        count += 1;
    }
    assert!(
        count >= 6,
        "expected at least 6 fixture lines, got {}",
        count
    );
}
