//! Op specs — declarative description of how `lml <op>` is invoked.
//!
//! Front-ends look up an `OpSpec` by op id, collect input/output paths
//! from the user, and call `OpSpec::build_args` to produce the final argv
//! that's passed to `runner::spawn_lml`.

#[derive(Debug, Clone)]
pub struct OpSpec {
    pub cmd: &'static str,
    pub input: bool,
    pub output: bool,
    /// `"-o"` for named flag, `"POS"` for positional output, `""` if no output.
    pub output_flag: &'static str,
    pub recursive: bool,
    pub extra: &'static [&'static str],
}

impl OpSpec {
    /// Build the full argv (without the binary name) for `runner::spawn_lml`.
    ///
    /// Always prepends `--emit-json-events` so the lml binary streams
    /// JSON-line OpEvents back to the TUI runner. Without it, lml's
    /// plain stdout is wrapped as `OpEvent::Log` and no Progress /
    /// FileDone events reach the compress dashboard — the bar stays at
    /// 0/0 even while encoding is in flight.
    pub fn build_args(&self, input: Option<&str>, output: Option<&str>) -> Vec<String> {
        // `--emit-json-events` is a global flag on the lml CLI; it
        // must come BEFORE the subcommand (clap rejects it otherwise).
        let mut args: Vec<String> = vec!["--emit-json-events".to_string(), self.cmd.to_string()];
        if let Some(i) = input {
            args.push(i.to_string());
        }
        if let Some(o) = output {
            if self.output_flag == "POS" {
                args.push(o.to_string());
            } else if !self.output_flag.is_empty() {
                args.push(self.output_flag.to_string());
                args.push(o.to_string());
            }
        }
        for e in self.extra {
            args.push((*e).to_string());
        }
        if self.recursive {
            // Only add -r when input is a directory (the lml binary errors
            // out on -r against a single file).
            if let Some(p) = input {
                if std::path::Path::new(p).is_dir() {
                    args.push("-r".to_string());
                }
            }
        }
        args
    }
}

/// Look up an op spec by its canonical id. Op ids are frozen by
/// `specs/ui-parity.md::Op IDs`.
pub fn op_spec(op_id: &str) -> Option<OpSpec> {
    Some(match op_id {
        // `encode_neural` (LMQ) intentionally not wired here. The
        // host-side neural encoder is not built into the `lml` CLI in
        // this branch -- the previous arm aliased `cmd: "encode"`
        // (lossless) which silently shipped LML bytes labelled "LMQ
        // Neural" in the dashboard. The TUI's `start_op` short-circuits
        // `encode_neural` / `decode_neural` with a "not built" status
        // before reaching this lookup. See ADR 0018.
        "encode_lma" => OpSpec {
            // Default Lossless mode in the TUI. Packs the entire
            // output into per-recording .lma archives so siblings can't
            // be silently lost. The previously-defined "encode"
            // (bare-LML) op-id was removed in v1.1 -- the TUI never
            // wires to it anymore, and CLI users who want bare LML
            // pass --no-bundle / --bare-lml directly to `lml encode`.
            cmd: "encode",
            input: true,
            output: true,
            output_flag: "-o",
            recursive: true,
            extra: &["--verify", "--cross-validate", "--skip-existing", "--lma"],
        },
        "encode_lml_siblings" => OpSpec {
            // Sibling-preserving Lossless mode. Per-EEG-file .lml
            // outputs + non-EEG siblings COPIED verbatim alongside
            // (no archive container). Implemented by
            // `lamquant_core::lma::pack_lml_with_siblings`; the
            // lml CLI exposes it via `--lml-siblings`. Same safety
            // posture as --lma (refuses symlinks, hard-errors on
            // unsafe paths).
            cmd: "encode",
            input: true,
            output: true,
            output_flag: "-o",
            recursive: true,
            extra: &[
                "--verify",
                "--cross-validate",
                "--skip-existing",
                "--lml-siblings",
            ],
        },
        "decode" => OpSpec {
            // --to-edf reconstructs a byte-identical EDF/BDF (header +
            // data records + trailing) from the LML container, instead
            // of the legacy raw int32 LE sample dump. Required for the
            // "decompressed file = source file" guarantee — without
            // this flag, the decoder output is sample data only and
            // cannot SHA-match the original.
            cmd: "decode",
            input: true,
            output: true,
            output_flag: "-o",
            recursive: true,
            extra: &["--to-edf"],
        },
        "verify" => OpSpec {
            cmd: "verify",
            input: true,
            output: false,
            output_flag: "",
            recursive: true,
            extra: &[],
        },
        "info" => OpSpec {
            cmd: "info",
            input: true,
            output: false,
            output_flag: "",
            recursive: false,
            extra: &[],
        },
        "stats" => OpSpec {
            cmd: "stats",
            input: true,
            output: false,
            output_flag: "",
            recursive: true,
            extra: &[],
        },
        "bench" => OpSpec {
            cmd: "bench",
            input: true,
            output: false,
            output_flag: "",
            recursive: false,
            extra: &[],
        },
        "archive" => OpSpec {
            cmd: "archive",
            input: true,
            output: true,
            output_flag: "-o",
            recursive: false,
            extra: &[],
        },
        "extract" => OpSpec {
            cmd: "extract",
            input: true,
            output: true,
            output_flag: "-o",
            recursive: false,
            extra: &["--verify"],
        },
        "list_archive" => OpSpec {
            cmd: "list-archive",
            input: true,
            output: false,
            output_flag: "",
            recursive: false,
            extra: &[],
        },
        "verify_archive" => OpSpec {
            cmd: "verify-archive",
            input: true,
            output: false,
            output_flag: "",
            recursive: false,
            extra: &[],
        },
        "verify_manifest" => OpSpec {
            cmd: "verify-manifest",
            input: true,
            output: false,
            output_flag: "",
            recursive: false,
            extra: &[],
        },
        "export_csv" => OpSpec {
            cmd: "export",
            input: true,
            output: true,
            output_flag: "-o",
            recursive: false,
            extra: &["--format", "csv"],
        },
        "export_npy" => OpSpec {
            cmd: "export",
            input: true,
            output: true,
            output_flag: "-o",
            recursive: false,
            extra: &["--format", "npy"],
        },
        "export_raw" => OpSpec {
            cmd: "export",
            input: true,
            output: true,
            output_flag: "-o",
            recursive: false,
            extra: &["--format", "raw"],
        },
        "recover" => OpSpec {
            cmd: "recover",
            input: true,
            output: true,
            output_flag: "POS",
            recursive: false,
            extra: &[],
        },
        "diff" => OpSpec {
            cmd: "diff",
            input: true,
            output: false,
            output_flag: "",
            recursive: false,
            extra: &[],
        },
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_op_returns_none() {
        assert!(op_spec("not_a_real_op").is_none());
    }

    #[test]
    fn encode_dir_includes_recursive() {
        // The bare "encode" op-id was removed in v1.1 -- TUI flows
        // route through `encode_lma` (per-EDF archives) and
        // `encode_lml_siblings` (LML + copied non-EEG files). Both
        // should fold in `-r` when the input is a directory.
        for op_id in ["encode_lma", "encode_lml_siblings"] {
            let spec = op_spec(op_id).unwrap_or_else(|| panic!("missing op_spec: {}", op_id));
            // Use the workspace root as a known-existing directory.
            let args = spec.build_args(Some("."), Some("/tmp/out"));
            assert!(
                args.contains(&"-r".to_string()),
                "{} should include -r for directory input",
                op_id,
            );
            assert!(
                args.contains(&"--verify".to_string()),
                "{} should include --verify",
                op_id,
            );
        }
    }

    #[test]
    fn encode_lml_siblings_passes_through_flag() {
        let spec = op_spec("encode_lml_siblings").expect("encode_lml_siblings registered");
        let args = spec.build_args(Some("/tmp/src"), Some("/tmp/out"));
        // The CLI flag that triggers the lma::pack_lml_with_siblings
        // path. Without it the cmd_encode happy-path would run and
        // produce per-EDF .lma archives instead.
        assert!(
            args.contains(&"--lml-siblings".to_string()),
            "encode_lml_siblings argv missing --lml-siblings: {:?}",
            args,
        );
        // Mutually-exclusive flags must NOT be in the same argv.
        assert!(!args.contains(&"--lma".to_string()));
        assert!(!args.contains(&"--no-bundle".to_string()));
    }

    #[test]
    fn recover_uses_positional_output() {
        let spec = op_spec("recover").unwrap();
        let args = spec.build_args(Some("a.lml"), Some("a.out"));
        // Last two positional args: input, output (no -o flag).
        let p = args.iter().position(|a| a == "a.lml").unwrap();
        assert_eq!(args[p + 1], "a.out");
        assert!(!args.contains(&"-o".to_string()));
    }
}
