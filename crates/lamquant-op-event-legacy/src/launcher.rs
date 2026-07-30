//! Launcher table — external commands that aren't `lml` subcommands.
//!
//! Used for training, eagle validator, pytest, install scripts, GUI launch
//! — anything where the front-end shells out to a different binary. Keep
//! this table in sync with `specs/ui-parity.md::Op IDs`.
//!
//! ## BLUT recipe dispatch
//!
//! Training-tier launchers (`cockpit_data_prep`,
//! `cockpit_train_encoder`, etc.) resolve via [`blut_launcher`]
//! instead of [`launcher`], returning a `(recipe, args_json,
//! label)` triple. App-layer dispatch must check `blut_launcher`
//! FIRST and call `runner::spawn_blut` for matching IDs — that
//! path tails `status.jsonl` and translates `StageEvent`s into
//! `OpEvent`s for the dashboard, which the raw `spawn_command`
//! path cannot do.

/// Look up an external command spec by op id. Returns `(program, args, label)`.
pub fn launcher(id: &str) -> Option<(&'static str, Vec<&'static str>, &'static str)> {
    Some(match id {
        // Training (Python)
        "train_encoder" => ("python", vec!["lamquant.py", "train", "--mode=encoder"], "encoder training"),
        "train_snn"     => ("python", vec!["lamquant.py", "train", "--mode=snn"],     "SNN training"),
        "train_tnn"     => ("python", vec!["lamquant.py", "train", "--mode=tnn"],     "TNN training"),
        "train_resume"  => ("python", vec!["lamquant.py", "train", "--resume"],       "resume training"),

        // Eagle validator (Python entry point)
        "eagle_quick" => ("python", vec!["-m", "lamquant_codec.eagle", "--suite=quick"], "eagle: quick"),
        "eagle_full"  => ("python", vec!["-m", "lamquant_codec.eagle", "--suite=full"],  "eagle: full"),
        "eagle_bench" => ("python", vec!["-m", "lamquant_codec.eagle", "--suite=bench"], "eagle: bench"),
        "eagle_lqs_l" => ("python", vec!["-m", "lamquant_codec.eagle", "--lqs=L"],       "eagle: LQS-L"),
        "eagle_lqs_c" => ("python", vec!["-m", "lamquant_codec.eagle", "--lqs=C"],       "eagle: LQS-C"),
        "eagle_lqs_m" => ("python", vec!["-m", "lamquant_codec.eagle", "--lqs=M"],       "eagle: LQS-M"),
        "eagle_lqs_a" => ("python", vec!["-m", "lamquant_codec.eagle", "--lqs=A"],       "eagle: LQS-A"),
        "eagle_perf"  => ("python", vec!["-m", "lamquant_codec.eagle", "--suite=perf"],  "eagle: perf"),
        "eagle_rd"    => ("python", vec!["-m", "lamquant_codec.eagle", "--suite=rd"],    "eagle: rate-distortion"),
        "eagle_h2h"   => ("python", vec!["-m", "lamquant_codec.eagle", "--suite=h2h"],   "eagle: head-to-head"),

        // pytest menu
        "test_conformance" => ("pytest", vec!["tests/conformance/", "-q"],   "tests: conformance"),
        "test_full"        => ("pytest", vec!["tests/", "-q"],               "tests: full"),
        "test_paranoid"    => ("pytest", vec!["tests/", "-q", "--paranoid"], "tests: paranoid"),
        "test_codec"       => ("pytest", vec!["tests/codec/", "-q"],         "tests: codec"),

        // Setup / install
        "setup_pip"    => ("pip",   vec!["install", "-e", ".[dev]"],                       "pip install -e .[dev]"),
        "setup_extras" => ("pip",   vec!["install", "hypothesis", "prompt_toolkit", "zstandard"], "pip extras"),
        "setup_cargo"  => ("cargo", vec!["build", "--release", "--manifest-path", "lamquant-lossless/Cargo.toml", "--bin", "lml"], "cargo build lml"),
        "setup_musl"   => ("cargo", vec!["build", "--release", "--manifest-path", "lamquant-lossless/Cargo.toml", "--bin", "lml", "--target", "x86_64-unknown-linux-musl"], "static linux build"),
        "setup_windows"=> ("cargo", vec!["build", "--release", "--manifest-path", "lamquant-lossless/Cargo.toml", "--bin", "lml", "--target", "x86_64-pc-windows-gnu"], "windows build"),

        // GUI / visualization launchers — T5 pipe shape (ADR 0020).
        //
        // `$INPUT` in args is substituted at dispatch time by
        // `app.rs::start_launcher` with the path the user picked in
        // the visualization panel's file browser. Tools that genuinely
        // can't accept piped input (license-locked hardware GUIs,
        // network-only studios) live in the `viz_legacy_*` namespace
        // below — sequestered, not deleted.
        "gui"                => ("lamquant-gui",  vec!["$INPUT"],            "Vision GUI"),
        "viz_lamquant-gui"   => ("lamquant-gui",  vec!["$INPUT"],            "Vision GUI"),
        "viz_eeglab"         => ("eeglab",        vec!["$INPUT"],            "EEGLab"),
        // MNE-Python wrapper: load the file path the user picked,
        // open the standard raw browser. `$INPUT` is substituted
        // before exec so the `-c` payload sees a concrete path.
        "viz_mne"            => (
            "python3",
            vec![
                "-c",
                "import sys, mne; \
                 raw = mne.io.read_raw(sys.argv[1], preload=False); \
                 raw.plot(block=True)",
                "$INPUT",
            ],
            "MNE-Python viewer",
        ),

        // ── viz_legacy_* — sequestered: pipe-incompatible tools ──
        //
        // OpenBCIGUI, BVAnalyzer, BESA all open to their own
        // internal file pickers / hardware streams; they ignore
        // command-line file arguments. Kept callable (sequester-not-
        // delete) but moved to a separate namespace so the main
        // viz panel can render them in a "legacy launchers"
        // subsection with a clear note.
        "viz_legacy_OpenBCIGUI" => ("OpenBCIGUI", vec![], "OpenBCI GUI (legacy: no piped input)"),
        "viz_legacy_BVAnalyzer" => ("BVAnalyzer", vec![], "BrainVision Analyzer (legacy: no piped input)"),
        "viz_legacy_besa"       => ("besa",       vec![], "BESA (legacy: no piped input)"),
        // Original aliases kept callable so any stale launcher ID
        // out in the world still resolves. Each delegates to the
        // legacy entry above.
        "viz_OpenBCIGUI"     => ("OpenBCIGUI",    vec![], "OpenBCI GUI"),
        "viz_BVAnalyzer"     => ("BVAnalyzer",    vec![], "BrainVision Analyzer"),
        "viz_besa"           => ("besa",          vec![], "BESA"),

        // Auto-install launchers fired by the visualization panel's
        // [Enter]/[i] key on missing tools. Each runs the standard
        // package-manager install for that tool. Network access
        // required; failures show in the output panel and the
        // visualization panel's [r] re-probe surfaces success.
        // Vision GUI build — runs `cargo install` against the
        // gui/src-tauri crate. Requires repo-root cwd (so the
        // --path arg resolves) and the Tauri Linux webview deps:
        //   apt: libwebkit2gtk-4.1-dev libappindicator3-dev librsvg2-dev patchelf
        // Output panel surfaces compile errors verbatim so users
        // can diagnose missing system libs.
        "viz_install_lamquant_gui" => (
            "sh",
            vec![
                "-c",
                // 3-step build with strict per-step failure surfacing.
                // Each step writes a clear PASS/FAIL/STARTED marker so
                // even a wall of svelte/vite warnings can't bury the
                // final outcome. Logs persist at /tmp/lamquant-gui-install.log
                // for post-mortem inspection; tail -F that file in
                // another terminal to follow live.
                "LOG=/tmp/lamquant-gui-install.log; \
                 : > \"$LOG\"; \
                 mark()  { echo \"\"; echo \"========== $1 ==========\"; echo \"$1\" >> \"$LOG\"; }; \
                 fail()  { echo \"\"; echo \"########## INSTALL FAILED — $1 ##########\"; echo \"see $LOG\"; exit 1; }; \
                 if ! command -v npm >/dev/null 2>&1; then fail 'npm not found (install Node.js first)'; fi; \
                 if ! command -v cargo >/dev/null 2>&1; then fail 'cargo not found (install Rust toolchain)'; fi; \
                 if [ ! -d gui ]; then fail 'gui/ directory missing — run from repo root'; fi; \
                 mark '[1/4] cleanup stale install + caches'; \
                 BINPATH=\"$(command -v lamquant-gui 2>/dev/null || echo '')\"; \
                 if [ -n \"$BINPATH\" ]; then echo \"removing $BINPATH\"; rm -f \"$BINPATH\"; fi; \
                 cargo clean -p lamquant-gui 2>&1 | tee -a \"$LOG\" || true; \
                 rm -rf gui/src-tauri/target/release/build/lamquant-gui-* 2>/dev/null || true; \
                 rm -rf gui/build 2>/dev/null || true; \
                 mark '[2/4] npm install (frontend deps)'; \
                 (cd gui && npm install) 2>&1 | tee -a \"$LOG\" || fail 'step 2: npm install'; \
                 mark '[3/4] tauri build --no-bundle (Tauri CLI: embeds frontend via tauri-build)'; \
                 (cd gui && ./node_modules/.bin/tauri build --no-bundle) 2>&1 | tee -a \"$LOG\" || fail 'step 3: tauri build'; \
                 mark '[4/4] copy binary into ~/.cargo/bin/ and verify'; \
                 SRC=target/release/lamquant-gui; \
                 if [ ! -f \"$SRC\" ]; then SRC=gui/src-tauri/target/release/lamquant-gui; fi; \
                 if [ ! -f \"$SRC\" ]; then fail 'step 4: tauri build did not produce a binary in target/release/'; fi; \
                 mkdir -p \"$HOME/.cargo/bin\"; \
                 cp -f \"$SRC\" \"$HOME/.cargo/bin/lamquant-gui\"; \
                 chmod +x \"$HOME/.cargo/bin/lamquant-gui\"; \
                 NEWPATH=\"$HOME/.cargo/bin/lamquant-gui\"; \
                 echo \"copied $SRC → $NEWPATH\"; \
                 echo \"binary stat: $(stat -c %y \"$NEWPATH\")\"; \
                 EMBED_COUNT=$(strings \"$NEWPATH\" 2>/dev/null | grep -c _app/immutable || echo 0); \
                 echo \"embedded SvelteKit asset refs found: $EMBED_COUNT\"; \
                 if [ \"$EMBED_COUNT\" = 0 ]; then fail 'step 4: binary has 0 embedded frontend assets — build pipeline broken'; fi; \
                 echo \"\"; \
                 echo '########## INSTALL COMPLETE ##########'; \
                 echo \"binary: $NEWPATH\"; \
                 echo \"log:    $LOG\"; \
                 echo 're-launch via `lamquant --gui` or [r] in the viz panel'",
            ],
            "install: build + cargo install Vision GUI",
        ),
        // --force / --force-reinstall lets the same launcher serve
        // both first-install AND repair (re-fire on a working
        // install just rebuilds it). Companion viz_uninstall_<tool>
        // launchers below remove the tool.
        "viz_install_mne"        => ("pip",   vec!["install", "--force-reinstall", "mne"],     "install: pip install mne"),
        "viz_install_scope_tui"  => ("cargo", vec!["install", "--force", "scope-tui"],         "install: cargo install scope-tui"),
        "viz_install_bottom"     => ("cargo", vec!["install", "--force", "bottom"],            "install: cargo install bottom"),
        "viz_install_television" => ("cargo", vec!["install", "--force", "television"],        "install: cargo install television"),
        "viz_install_csvlens"    => ("cargo", vec!["install", "--force", "csvlens"],           "install: cargo install csvlens"),
        "viz_install_gitui"      => ("cargo", vec!["install", "--force", "gitui"],             "install: cargo install gitui"),

        // Uninstall companions — pair with viz_install_<tool> entries.
        // Vision GUI: cargo uninstall removes ~/.cargo/bin/lamquant-gui
        // but does NOT clean gui/build/ (frontend dist) — it's source
        // tree the user owns.
        "viz_uninstall_lamquant_gui" => ("cargo", vec!["uninstall", "lamquant-gui"], "uninstall: cargo uninstall lamquant-gui"),
        "viz_uninstall_mne"          => ("pip",   vec!["uninstall", "-y", "mne"],   "uninstall: pip uninstall mne"),
        "viz_uninstall_scope_tui"    => ("cargo", vec!["uninstall", "scope-tui"],   "uninstall: cargo uninstall scope-tui"),
        "viz_uninstall_bottom"       => ("cargo", vec!["uninstall", "bottom"],      "uninstall: cargo uninstall bottom"),
        "viz_uninstall_television"   => ("cargo", vec!["uninstall", "television"],  "uninstall: cargo uninstall television"),
        "viz_uninstall_csvlens"      => ("cargo", vec!["uninstall", "csvlens"],     "uninstall: cargo uninstall csvlens"),
        "viz_uninstall_gitui"        => ("cargo", vec!["uninstall", "gitui"],       "uninstall: cargo uninstall gitui"),

        // Cockpit utilities (Phase B.2 wiring of [r/c/m] keys).
        // Linux/macOS only — sh -c shell pipelines. Windows users see
        // the same status sidebar entry; the launcher itself fails fast.
        "cockpit_reset" => (
            "sh",
            vec![
                "-c",
                // Two destructive operations, each with honest exit
                // reporting. tmux missing-session is normal (no error);
                // tmux command-not-installed is reported. rm failures
                // are reported. Final exit code reflects whether either
                // step encountered a real error.
                "rc=0; \
                 if [ -e ~/.cache/lamquant ]; then \
                     if rm -rf ~/.cache/lamquant; then \
                         echo '✓ ~/.cache/lamquant cleared'; \
                     else \
                         echo '✗ rm -rf ~/.cache/lamquant failed (permissions?)'; \
                         rc=1; \
                     fi; \
                 else \
                     echo '— ~/.cache/lamquant did not exist; nothing to clear'; \
                 fi; \
                 if command -v tmux >/dev/null 2>&1; then \
                     out=$(tmux kill-session -t lamquant-train 2>&1); \
                     case \"$out\" in \
                         *\"can't find session\"*|'') \
                             echo '✓ tmux: no lamquant-train session running' ;; \
                         *) \
                             echo \"✗ tmux kill-session failed: $out\"; \
                             rc=1 ;; \
                     esac; \
                 else \
                     echo '— tmux not installed; skipped session kill'; \
                 fi; \
                 if [ \"$rc\" = 0 ]; then \
                     echo 'reset complete'; \
                 else \
                     echo 'reset finished with errors'; \
                 fi; \
                 exit $rc",
            ],
            "reset: cache + tmux",
        ),
        "cockpit_checkpoints" => (
            "sh",
            vec![
                "-c",
                // Tighter glob: -path 'runs/*/checkpoints/*' so we match
                // only files under a `checkpoints/` directory inside a
                // run, not arbitrary files with "checkpoint" in the name.
                "if [ -d runs ]; then \
                     hits=$(find runs -maxdepth 6 -path 'runs/*/checkpoints/*' -type f 2>/dev/null | sort -r | head -100); \
                     if [ -z \"$hits\" ]; then \
                         echo 'no checkpoints found under runs/*/checkpoints/'; \
                     else \
                         echo \"$hits\"; \
                     fi; \
                 else \
                     echo 'no runs/ directory in cwd ('\"$PWD\"')'; \
                 fi",
            ],
            "list checkpoints",
        ),
        "cockpit_metrics" => (
            "sh",
            vec![
                "-c",
                // Distinguishes "no runs/ at all" from "runs/ exists but
                // empty"; both are non-error exits since training just
                // hasn't started yet.
                "if [ ! -d runs ]; then \
                     echo 'no runs/ directory in cwd ('\"$PWD\"')'; \
                     exit 0; \
                 fi; \
                 latest=$(ls -td runs/*/ 2>/dev/null | head -1); \
                 if [ -z \"$latest\" ]; then \
                     echo 'no training runs found under runs/'; \
                     exit 0; \
                 fi; \
                 log=\"${latest}log.txt\"; \
                 if [ -f \"$log\" ]; then \
                     echo \"# tailing $log\"; \
                     tail -200 \"$log\"; \
                 else \
                     echo \"no log.txt in $latest (looked for $log)\"; \
                 fi",
            ],
            "tail latest log",
        ),

        // ── Firmware Hub (T6 / ADR 0019) ─────────────────────────
        //
        // Replaces the pre-T6 missing `scripts/firmware/build_*.sh`
        // scripts with `cargo build` + `probe-rs run` directly.
        // 4 Raytac module targets: RP2350 (Hazard3 RISC-V),
        // NRF54L15 (Cortex-M33 + BLE 6.0), ESP32-P4 (HP400 RISC-V),
        // STM32N6 (Cortex-M55 + NPU). ESP32-S3 sequestered to
        // `fw_legacy_esp32s3` (probe-rs upstream lacks ESP32-S3
        // support today; user-facing alias `fw_export` and panel
        // entries kept callable per sequester-not-delete).

        // Device enumeration — single source of truth for "what's
        // plugged in." The firmware panel parses this output to
        // populate target rows.
        "fw_list_devices" => ("probe-rs", vec!["list"], "list connected debug probes"),

        // Per-target cargo build. Decomposed workspace: each target
        // is its own crate under firmware/lamquant-firmware-<target>/.
        "fw_build_rp2350" => (
            "cargo",
            vec![
                "build", "-p", "lamquant-firmware-rp2350",
                "--target", "riscv32imac-unknown-none-elf",
                "--release",
            ],
            "build firmware: RP2350 (Raytac)",
        ),
        "fw_build_nrf54l15" => (
            "cargo",
            vec![
                "build", "-p", "lamquant-firmware-nrf54l15",
                "--target", "thumbv8m.main-none-eabihf",
                "--release",
            ],
            "build firmware: NRF54L15 (Raytac)",
        ),
        "fw_build_esp32p4" => (
            "cargo",
            vec![
                "build", "-p", "lamquant-firmware-esp32p4",
                "--target", "riscv32imafc-unknown-none-elf",
                "--release",
            ],
            "build firmware: ESP32-P4 (Raytac)",
        ),
        "fw_build_stm32n6" => (
            "cargo",
            vec![
                "build", "-p", "lamquant-firmware-stm32n6",
                "--target", "thumbv8m.main-none-eabihf",
                "--release",
            ],
            "build firmware: STM32N6 (Raytac)",
        ),

        // Per-target flash via probe-rs. Chip IDs match what
        // `probe-rs list-chips` reports. Binary paths follow the
        // decomposed crate naming: target/<triple>/release/<crate-name>.
        "fw_flash_rp2350" => (
            "probe-rs",
            vec![
                "run", "--chip", "RP2350",
                "target/riscv32imac-unknown-none-elf/release/lamquant-firmware-rp2350",
            ],
            "flash firmware: RP2350",
        ),
        "fw_flash_nrf54l15" => (
            "probe-rs",
            vec![
                "run", "--chip", "nRF54L15",
                "target/thumbv8m.main-none-eabihf/release/lamquant-firmware-nrf54l15",
            ],
            "flash firmware: NRF54L15",
        ),
        "fw_flash_esp32p4" => (
            "probe-rs",
            vec![
                "run", "--chip", "ESP32-P4",
                "target/riscv32imafc-unknown-none-elf/release/lamquant-firmware-esp32p4",
            ],
            "flash firmware: ESP32-P4 (probe-rs support pending upstream)",
        ),
        "fw_flash_stm32n6" => (
            "probe-rs",
            vec![
                "run", "--chip", "STM32N657NI",
                "target/thumbv8m.main-none-eabihf/release/lamquant-firmware-stm32n6",
            ],
            "flash firmware: STM32N6",
        ),

        // Per-target size report via arm-none-eabi-size / cargo size.
        "fw_size_rp2350" => (
            "arm-none-eabi-size",
            vec![
                "target/riscv32imac-unknown-none-elf/release/lamquant-firmware-rp2350",
            ],
            "size report: RP2350",
        ),
        "fw_size_stm32n6" => (
            "arm-none-eabi-size",
            vec![
                "target/thumbv8m.main-none-eabihf/release/lamquant-firmware-stm32n6",
            ],
            "size report: STM32N6",
        ),
        "fw_size_esp32p4" => (
            "arm-none-eabi-size",
            vec![
                "target/riscv32imafc-unknown-none-elf/release/lamquant-firmware-esp32p4",
            ],
            "size report: ESP32-P4",
        ),
        "fw_size_nrf54l15" => (
            "arm-none-eabi-size",
            vec![
                "target/thumbv8m.main-none-eabihf/release/lamquant-firmware-nrf54l15",
            ],
            "size report: NRF54L15",
        ),

        // Per-target cargo check (fast compile verification).
        "fw_check_rp2350" => (
            "cargo",
            vec!["check", "-p", "lamquant-firmware-rp2350", "--target", "riscv32imac-unknown-none-elf"],
            "check firmware: RP2350",
        ),
        "fw_check_stm32n6" => (
            "cargo",
            vec!["check", "-p", "lamquant-firmware-stm32n6", "--target", "thumbv8m.main-none-eabihf"],
            "check firmware: STM32N6",
        ),
        "fw_check_esp32p4" => (
            "cargo",
            vec!["check", "-p", "lamquant-firmware-esp32p4", "--target", "riscv32imafc-unknown-none-elf"],
            "check firmware: ESP32-P4",
        ),
        "fw_check_nrf54l15" => (
            "cargo",
            vec!["check", "-p", "lamquant-firmware-nrf54l15", "--target", "thumbv8m.main-none-eabihf"],
            "check firmware: NRF54L15",
        ),

        // Weight export — supports T6.3 $WEIGHTS / $OUTPUT
        // substitution following the T5 $INPUT pattern. The Python
        // script accepts `--weights <path> --out <path>`; explicit
        // file picking happens in the firmware panel.
        "fw_export" => (
            "python",
            vec![
                "scripts/export_weights.py",
                "--weights", "$WEIGHTS",
                "--out", "$OUTPUT",
            ],
            "export weights (training side)",
        ),

        // ── Sequestered: ESP32-S3 ────────────────────────────────
        //
        // ESP32-S3 dropped per the firmware-hub vision (4 Raytac
        // modules only). Sequester-not-delete policy keeps the
        // launcher entry callable — direct users still get a
        // working build path while the canonical hub focuses on
        // the 4 target modules.
        "fw_legacy_esp32s3" => (
            "cargo",
            vec![
                "build", "--manifest-path", "lamquant-firmware/Cargo.toml",
                "--target", "xtensa-esp32s3-none-elf",
                "--features", "target-esp32s3",
                "--release",
            ],
            "build firmware: ESP32-S3 (legacy: not part of T6 hub)",
        ),

        // Cockpit LQT integrations (T3.2 — Python cockpit parity).
        // `cockpit_jobs` lists LQT's per-job state via the `lqt jobs`
        // subcommand. Same data the Python cockpit's queue/history
        // screens used to compose by hand; LQT now owns the table.
        "cockpit_jobs" => ("lqt", vec!["jobs"], "LQT: list training jobs"),
        // `cockpit_export` mirrors the Python cockpit `_screen_export`
        // weight-export flow. Today it shells the same training-side
        // script `fw_export` uses; a follow-up sprint promotes this
        // to a typed BLUT stage with explicit `--weights` + `--out`
        // arguments (Track E in the plan also touches this path).
        "cockpit_export" => (
            "python",
            vec!["scripts/export_weights.py"],
            "Export weights (training side)",
        ),

        // Syscheck (handled in-process for the most part, but expose Python self-test)
        "syscheck_py" => ("python", vec!["-m", "lamquant_codec.cli.syscheck"], "syscheck (python)"),

        _ => return None,
    })
}

/// Resolve a cockpit-tier BLUT launcher id to `(recipe_name,
/// args_json, label)`. App-layer dispatch must call
/// [`runner::spawn_blut`](crate::runner::spawn_blut) for these
/// ids — only that path tails `status.jsonl` and translates
/// `StageEvent`s back to `OpEvent`s.
///
/// Defaults reflect repo-relative layout (`./data/lma`,
/// `runs/...`). Override per-launcher by env or by editing the
/// args JSON inline below — JSON is the only knob until a
/// recipe-args wizard panel ships (post-C3).
pub fn blut_launcher(id: &str) -> Option<(&'static str, &'static str, &'static str)> {
    Some(match id {
        // Full data prep — LMA conversion. `output_dir` is the
        // only required field; rest default in
        // `lamquant_data_prep::Args` (see
        // training/cookbooks/lamquant/src/recipes/lamquant_data_prep.rs).
        "cockpit_data_prep" => (
            "lamquant_data_prep",
            r#"{"output_dir":"./data/lma"}"#,
            "BLUT: data prep (LML → LMA)",
        ),
        // Encoder recipe — all fields default in
        // `lamquant_encoder::Args`, so `{}` accepts every default
        // (preset=fast, tier=0, seed=42, val_fraction=0.05,
        // mae_pretrain=false).
        "cockpit_train_encoder" => ("lamquant_encoder", "{}", "BLUT: encoder train"),
        "cockpit_train_snn" => ("lamquant_snn", "{}", "BLUT: SNN train"),
        "cockpit_train_oracle" => ("lamquant_oracle", "{}", "BLUT: oracle (teacher) train"),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_op_returns_some() {
        assert!(launcher("train_encoder").is_some());
        assert!(launcher("eagle_lqs_l").is_some());
    }

    #[test]
    fn unknown_op_returns_none() {
        assert!(launcher("not_a_real_op").is_none());
    }

    #[test]
    fn blut_launcher_resolves_cockpit_ids() {
        let (rec, json, _) = blut_launcher("cockpit_train_encoder").unwrap();
        assert_eq!(rec, "lamquant_encoder");
        // Args JSON must parse — guard against typos like a stray
        // trailing comma.
        let _: serde_json::Value = serde_json::from_str(json).unwrap();

        let (rec, json, _) = blut_launcher("cockpit_data_prep").unwrap();
        assert_eq!(rec, "lamquant_data_prep");
        let v: serde_json::Value = serde_json::from_str(json).unwrap();
        assert!(v.get("output_dir").is_some(), "output_dir is required");
    }

    #[test]
    fn blut_launcher_unknown_returns_none() {
        assert!(blut_launcher("not_a_real_blut_op").is_none());
        // Non-BLUT IDs (e.g. train_encoder) must NOT resolve via
        // the BLUT table — they live in `launcher()`.
        assert!(blut_launcher("train_encoder").is_none());
    }
}
