use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use lamquant_lma_training_legacy::{
    LegacyEnvironment, LegacyTrainer, ALLOWED_ENVIRONMENT, SOURCE_REPOSITORY, SOURCE_REVISION,
};

#[test]
fn source_and_trainer_allowlist_are_frozen() {
    assert_eq!(
        SOURCE_REPOSITORY,
        "https://github.com/Quitetall/blut-lamquant.git"
    );
    assert_eq!(SOURCE_REVISION, "64d4478deb2ea52193b9d9b108e9c46793701687");
    assert_eq!(
        LegacyTrainer::ALL
            .iter()
            .map(|trainer| (trainer.as_str(), trainer.script()))
            .collect::<Vec<_>>(),
        vec![
            ("pretrain_mae", "python/lamquant/student/pretrain_mae.py"),
            (
                "pretrain_ssl_tueg",
                "python/lamquant/snn/pretrain_ssl_tueg.py"
            ),
            (
                "train_4state_controller",
                "python/lamquant/snn/train_4state_controller.py"
            ),
            (
                "train_combined",
                "python/lamquant/decoder/train_combined.py"
            ),
            ("train_joint", "python/lamquant/student/train_joint.py"),
            (
                "train_l3_teacher",
                "python/lamquant/oracle/train_l3_teacher.py"
            ),
            ("train_teacher", "python/lamquant/oracle/train_teacher.py"),
            (
                "train_vocos_decoder",
                "python/lamquant/decoder/train_vocos_decoder.py"
            ),
        ]
    );
    assert!(ALLOWED_ENVIRONMENT.contains(&"CUDA_VISIBLE_DEVICES"));
    assert!(ALLOWED_ENVIRONMENT.contains(&"LOCAL_RANK"));
    assert!(ALLOWED_ENVIRONMENT.contains(&"WANDB_MODE"));
    assert!(!ALLOWED_ENVIRONMENT.contains(&"HF_HOME"));
    assert!(!ALLOWED_ENVIRONMENT.contains(&"TORCH_HOME"));
    assert!(!ALLOWED_ENVIRONMENT.contains(&"XDG_CACHE_HOME"));
    assert!(LegacyEnvironment::new("LD_PRELOAD", "forbidden").is_err());
    for trainer in LegacyTrainer::ALL {
        assert!(
            trainer
                .artifact_roots(Path::new("/private/legacy-run"))
                .iter()
                .all(|root| root.starts_with("/private/legacy-run")),
            "{trainer} output root escapes persistent workspace"
        );
    }
}

#[test]
fn cli_exposes_help_and_closed_trainer_discovery() {
    let binary = env!("CARGO_BIN_EXE_lamquant-lma-training-legacy");
    let help = Command::new(binary)
        .arg("--help")
        .output()
        .expect("run help");
    assert!(help.status.success());
    assert!(String::from_utf8(help.stdout)
        .expect("UTF-8 help")
        .contains("--workspace ABSOLUTE_PATH"));

    let listing = Command::new(binary)
        .arg("--list-trainers")
        .output()
        .expect("list trainers");
    assert!(listing.status.success());
    assert_eq!(
        String::from_utf8(listing.stdout)
            .expect("UTF-8 trainer list")
            .lines()
            .collect::<Vec<_>>(),
        LegacyTrainer::ALL
            .iter()
            .map(|trainer| trainer.as_str())
            .collect::<Vec<_>>()
    );
}

#[test]
fn cli_rejects_duplicate_singleton_options() {
    let binary = env!("CARGO_BIN_EXE_lamquant-lma-training-legacy");
    for option in [
        "--git",
        "--checkout",
        "--trainer",
        "--python",
        "--workspace",
    ] {
        let value = if option == "--trainer" {
            "train_joint"
        } else {
            "/tmp/value"
        };
        let output = Command::new(binary)
            .args([option, value, option, value])
            .output()
            .expect("run duplicate option probe");
        assert_eq!(output.status.code(), Some(2), "{option}");
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .lines()
                .next()
                .is_some_and(|line| line == format!("duplicate {option}")),
            "{option}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn cli_exits_with_signal_status_when_git_handshake_is_interrupted() {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    let parent = tempfile::tempdir().expect("CLI signal fixture");
    let git = parent.path().join("git");
    let ready = parent.path().join("git.ready");
    fs::write(&git, "#!/bin/sh\n: > \"$0.ready\"\nexec /bin/sleep 60\n")
        .expect("write blocking Git");
    fs::set_permissions(&git, fs::Permissions::from_mode(0o700))
        .expect("make blocking Git executable");
    let checkout = parent.path().join("checkout");
    let workspace = parent.path().join("workspace");
    fs::create_dir(&checkout).expect("create checkout placeholder");

    let binary = env!("CARGO_BIN_EXE_lamquant-lma-training-legacy");
    let mut child = Command::new(binary)
        .args(["--git"])
        .arg(&git)
        .args(["--checkout"])
        .arg(&checkout)
        .args(["--trainer", "train_joint", "--python", "/usr/bin/python3"])
        .args(["--workspace"])
        .arg(&workspace)
        .spawn()
        .expect("spawn legacy CLI");
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline && !ready.exists() {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(ready.exists(), "Git handshake never became ready");
    let pid = i32::try_from(child.id()).expect("CLI PID fits pid_t");
    let result = unsafe { libc::kill(pid, libc::SIGTERM) };
    assert_eq!(result, 0, "signal CLI: {}", std::io::Error::last_os_error());

    let status = child.wait().expect("wait for interrupted CLI");

    assert_eq!(status.code(), Some(128 + libc::SIGTERM));
}
