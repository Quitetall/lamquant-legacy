//! Process-isolated launcher for retired LMA-direct LamQuant trainers.
//!
//! Current LamQuant training accepts only governed ABIR snapshots. This crate
//! names the final source revision that still contained LMA-direct branches
//! and exposes a closed trainer allowlist for forensic rollback tooling.

#[cfg(not(target_os = "linux"))]
compile_error!("lamquant-lma-training-legacy supports Linux hosts only");

use core::fmt;
use core::str::FromStr;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{CString, OsStr, OsString};
use std::fs::{self, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

const ISOLATED_RUNNER: &str = concat!(
    "import os, sys\n",
    "python_root, source_script, virtual_script, *args = sys.argv[1:]\n",
    "for suffix in ('', 'lamquant', 'lamquant/common', 'lamquant/decoder', ",
    "'lamquant/oracle', 'lamquant/snn', 'lamquant/student'):\n",
    "    sys.path.insert(0, os.path.join(python_root, suffix))\n",
    "sys.argv = [virtual_script, *args]\n",
    "with open(source_script, 'rb') as source:\n",
    "    code = compile(source.read(), virtual_script, 'exec')\n",
    "scope = {'__name__': '__main__', '__file__': virtual_script, ",
    "'__package__': None, '__cached__': None}\n",
    "exec(code, scope)\n",
);
const PYTHON_HANDSHAKE: &str = concat!(
    "import os, sys\n",
    "assert sys.flags.isolated == 1\n",
    "assert sys.flags.no_user_site == 1\n",
    "assert sys.dont_write_bytecode\n",
    "print('LAMQUANT_LEGACY_PYTHON_V1')\n",
    "print(os.fsencode(os.path.realpath(sys.executable)).hex())\n",
    "print(f'{sys.version_info.major}.{sys.version_info.minor}')\n",
);
const DEPENDENCY_PREFLIGHT: &str = concat!(
    "import importlib, os, sys\n",
    "python_root, source_script, virtual_script, *modules = sys.argv[1:]\n",
    "for suffix in ('', 'lamquant', 'lamquant/common', 'lamquant/decoder', ",
    "'lamquant/oracle', 'lamquant/snn', 'lamquant/student'):\n",
    "    sys.path.insert(0, os.path.join(python_root, suffix))\n",
    "with open(source_script, 'rb') as source:\n",
    "    code = compile(source.read(), virtual_script, 'exec')\n",
    "scope = {'__name__': '__lamquant_dependency_preflight__', ",
    "'__file__': virtual_script, '__package__': None, '__cached__': None}\n",
    "exec(code, scope)\n",
    "for module in modules:\n",
    "    loaded = importlib.import_module(module)\n",
    "    version = getattr(loaded, '__version__', '<not-declared>')\n",
    "    print(f'dependency\\t{module}\\t{version}')\n",
    "print('LAMQUANT_LEGACY_PREFLIGHT_V1')\n",
);
const MAX_GIT_TEXT_BYTES: usize = 64 * 1024 * 1024;
const MAX_ARCHIVE_BYTES: u64 = 512 * 1024 * 1024;
const MAX_EXECUTABLE_BYTES: u64 = 256 * 1024 * 1024;
const MAX_VENV_CONFIG_BYTES: u64 = 64 * 1024;
const MAX_PREFLIGHT_OUTPUT_BYTES: usize = 64 * 1024;
const MAX_WORKSPACE_VALIDATION_ENTRIES: usize = 100_000;
const MAX_WORKSPACE_VALIDATION_DEPTH: usize = 64;
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
const PREFLIGHT_TIMEOUT: Duration = Duration::from_secs(120);
const GIT_COMMAND_TIMEOUT: Duration = Duration::from_secs(120);
const POLL_INTERVAL: Duration = Duration::from_millis(10);
const TERMINATION_GRACE: Duration = Duration::from_secs(10);
const DESCENDANT_GRACE: Duration = Duration::from_millis(500);
const SOURCE_MANIFEST: &str = "source-projection-v1.manifest";
const WORKSPACE_BINDING: &str = "workspace-binding-v1.manifest";
const DEPENDENCY_REPORT: &str = "dependency-preflight-v1.txt";
const SANDBOX_EXECUTABLE: &str = "/usr/bin/bwrap";
const PINNED_PYTHON_EXECUTABLE: &str = "/tmp/lamquant-legacy-python";
const IMPORTABLE_EXTENSIONS: [&str; 5] = ["py", "pyi", "pyc", "pyo", "so"];
static ACTIVE_CHILD_PROCESS_GROUP: AtomicI32 = AtomicI32::new(0);
static PENDING_TERMINATION_SIGNAL: AtomicI32 = AtomicI32::new(0);
static ATOMIC_WRITE_SEQUENCE: AtomicU64 = AtomicU64::new(0);
static PROCESS_SUPERVISION_LOCK: Mutex<()> = Mutex::new(());

/// Typed record of a termination signal received while supervising a bounded
/// launcher subprocess.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SignalInterruption {
    signal: libc::c_int,
    phase: &'static str,
}

impl SignalInterruption {
    fn new(signal: libc::c_int, phase: &'static str) -> Self {
        Self { signal, phase }
    }

    pub const fn signal(&self) -> libc::c_int {
        self.signal
    }

    pub const fn phase(&self) -> &'static str {
        self.phase
    }

    pub const fn exit_code(&self) -> libc::c_int {
        128 + self.signal
    }
}

impl fmt::Display for SignalInterruption {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} interrupted by signal {}",
            self.phase, self.signal
        )
    }
}

impl std::error::Error for SignalInterruption {}

fn signal_interruption(error: &io::Error) -> Option<SignalInterruption> {
    error
        .get_ref()
        .and_then(|source| source.downcast_ref::<SignalInterruption>())
        .cloned()
}

fn interrupted_io_error(signal: libc::c_int, phase: &'static str) -> io::Error {
    io::Error::new(
        io::ErrorKind::Interrupted,
        SignalInterruption::new(signal, phase),
    )
}

fn map_launch_io_error(
    error: io::Error,
    fallback: impl FnOnce(io::Error) -> LaunchError,
) -> LaunchError {
    signal_interruption(&error)
        .map(LaunchError::Interrupted)
        .unwrap_or_else(|| fallback(error))
}

extern "C" fn forward_termination_signal(signal: libc::c_int) {
    PENDING_TERMINATION_SIGNAL
        .compare_exchange(0, signal, Ordering::Relaxed, Ordering::Relaxed)
        .ok();
}

/// Install CLI termination handlers that forward HUP/INT/TERM to the owned
/// legacy trainer process group before exiting.
///
/// Embedders may omit this process-global policy. Handler records signal only;
/// supervision loop forwards it, permits bounded checkpoint cleanup, then
/// kills remaining sandbox descendants. Every normal Rust error path also
/// kills and reaps the owned process group.
pub fn install_termination_signal_forwarding() -> Result<(), io::Error> {
    for signal in [libc::SIGHUP, libc::SIGINT, libc::SIGTERM] {
        // SAFETY: handler performs only async-signal-safe operations.
        let previous = unsafe {
            libc::signal(
                signal,
                forward_termination_signal as *const () as libc::sighandler_t,
            )
        };
        if previous == libc::SIG_ERR {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

/// Environment controls accepted by the isolated legacy process.
///
/// Values must be supplied explicitly. Ambient host state is never inherited.
pub const ALLOWED_ENVIRONMENT: [&str; 17] = [
    "BLUT_AI_MODELS",
    "CUDA_DEVICE_ORDER",
    "CUDA_VISIBLE_DEVICES",
    "LAMQUANT_NEURAL",
    "LMA_NUM_WORKERS",
    "LMA_PREFETCH_FACTOR",
    "LOCAL_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "MKL_NUM_THREADS",
    "NCCL_DEBUG",
    "NCCL_P2P_DISABLE",
    "OMP_NUM_THREADS",
    "RANK",
    "SNN_SEIZURE_HEAD",
    "WANDB_MODE",
    "WORLD_SIZE",
];

/// Immutable source repository containing the retired trainers.
pub const SOURCE_REPOSITORY: &str = "https://github.com/Quitetall/blut-lamquant.git";

/// Final reviewed revision before production LMA-direct retirement.
pub const SOURCE_REVISION: &str = "64d4478deb2ea52193b9d9b108e9c46793701687";

/// Retired trainer entrypoints available to isolated rollback tooling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LegacyTrainer {
    PretrainMae,
    PretrainSslTueg,
    Train4StateController,
    TrainCombined,
    TrainJoint,
    TrainL3Teacher,
    TrainVocosDecoder,
}

impl LegacyTrainer {
    pub const ALL: [Self; 7] = [
        Self::PretrainMae,
        Self::PretrainSslTueg,
        Self::Train4StateController,
        Self::TrainCombined,
        Self::TrainJoint,
        Self::TrainL3Teacher,
        Self::TrainVocosDecoder,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PretrainMae => "pretrain_mae",
            Self::PretrainSslTueg => "pretrain_ssl_tueg",
            Self::Train4StateController => "train_4state_controller",
            Self::TrainCombined => "train_combined",
            Self::TrainJoint => "train_joint",
            Self::TrainL3Teacher => "train_l3_teacher",
            Self::TrainVocosDecoder => "train_vocos_decoder",
        }
    }

    pub const fn script(self) -> &'static str {
        match self {
            Self::PretrainMae => "python/lamquant/student/pretrain_mae.py",
            Self::PretrainSslTueg => "python/lamquant/snn/pretrain_ssl_tueg.py",
            Self::Train4StateController => "python/lamquant/snn/train_4state_controller.py",
            Self::TrainCombined => "python/lamquant/decoder/train_combined.py",
            Self::TrainJoint => "python/lamquant/student/train_joint.py",
            Self::TrainL3Teacher => "python/lamquant/oracle/train_l3_teacher.py",
            Self::TrainVocosDecoder => "python/lamquant/decoder/train_vocos_decoder.py",
        }
    }

    /// Durable roots that can receive outputs from this frozen trainer.
    pub fn artifact_roots(self, workspace: &Path) -> Vec<PathBuf> {
        let artifacts = workspace.join("artifacts");
        let python = workspace.join("run/python");
        match self {
            Self::PretrainMae | Self::PretrainSslTueg => vec![artifacts],
            Self::Train4StateController => vec![
                artifacts,
                python.join("weights/snn"),
                python.join("training_logs"),
            ],
            Self::TrainCombined => vec![
                python.join("lamquant/oracle"),
                python.join("lamquant/decoder"),
            ],
            Self::TrainJoint => vec![artifacts, python.join("training_logs")],
            Self::TrainL3Teacher => vec![python.join("ai_models/oracle")],
            Self::TrainVocosDecoder => vec![python.join("ai_models/decoder")],
        }
    }

    /// Directories provisioned writable inside the persistent workspace bind.
    fn writable_directories(self, workspace: &Path) -> BTreeSet<PathBuf> {
        let mut directories = self
            .artifact_roots(workspace)
            .into_iter()
            .collect::<BTreeSet<_>>();
        directories.extend([
            workspace.join("artifacts"),
            workspace.join("artifacts/checkpoints"),
            workspace.join("artifacts/recovery"),
            workspace.join("artifacts/wandb"),
            workspace.join("artifacts/cache/huggingface"),
            workspace.join("artifacts/cache/torch"),
            workspace.join("artifacts/cache/xdg"),
            workspace.join("home"),
        ]);
        directories
    }

    /// Frozen data resources resolved relative to virtual `__file__`.
    ///
    /// Sandbox bind-mounts each exact projected file read-only at its virtual
    /// location. Outputs remain under writable `run/`.
    const fn read_resources(self) -> &'static [&'static str] {
        match self {
            Self::TrainJoint => &["python/lamquant/snn/band_std.json"],
            _ => &[],
        }
    }

    fn dependency_modules(self, args: &[OsString]) -> Vec<&'static str> {
        let mut modules = match self {
            Self::PretrainMae => vec![
                "data_types",
                "lamquant.ingredients",
                "lamquant_codec.training",
                "streaming_dataset",
            ],
            Self::PretrainSslTueg => vec!["lamquant.ingredients"],
            Self::Train4StateController => vec![
                "blut_core",
                "blut_core.ingredients.checkpoint._specs",
                "blut_core.metric_log",
                "lamquant.ingredients",
            ],
            Self::TrainCombined => vec![
                "data_types",
                "lamquant.ingredients",
                "lamquant_codec.training",
            ],
            Self::TrainJoint => vec![
                "auraloss.freq",
                "blut_core.metric_log",
                "data_types",
                "lamquant.ingredients",
                "lamquant_neural.positions",
                "streaming_dataset",
                "training_diagnostics",
            ],
            Self::TrainL3Teacher => vec!["lamquant.ingredients"],
            Self::TrainVocosDecoder => {
                vec!["lamquant.ingredients", "lamquant_codec.training"]
            }
        };
        if argument_value(args, "--logger") == Some("wandb") {
            modules.push("wandb");
        }
        if self == Self::TrainJoint
            && argument_value(args, "--lr-schedule") == Some("schedule-free")
        {
            modules.push("schedulefree");
        }
        if self == Self::TrainVocosDecoder && has_argument(args, "--dac-init") {
            modules.push("dac");
        }
        if self == Self::Train4StateController
            && argument_value(args, "--target-source") == Some("recon_difficulty")
        {
            modules.push("lamquant_core");
        }
        modules
    }

    fn managed_arguments(self, workspace: &Path) -> Vec<OsString> {
        let artifacts = workspace.join("artifacts");
        match self {
            Self::PretrainMae => vec![
                OsString::from("--output"),
                artifacts.join("pretrained_mae.ckpt").into_os_string(),
            ],
            Self::PretrainSslTueg => vec![
                OsString::from("--out"),
                artifacts.join("pretrain_ssl_tueg.pt").into_os_string(),
            ],
            Self::Train4StateController => vec![
                OsString::from("--checkpoint"),
                artifacts.join("snn_4state_best.pt").into_os_string(),
            ],
            Self::TrainJoint => vec![
                OsString::from("--ckpt-dir"),
                artifacts.join("checkpoints").into_os_string(),
                OsString::from("--resume-dir"),
                artifacts.join("recovery").into_os_string(),
            ],
            Self::TrainCombined | Self::TrainL3Teacher | Self::TrainVocosDecoder => Vec::new(),
        }
    }

    const fn managed_flags(self) -> &'static [&'static str] {
        match self {
            Self::PretrainMae => &["--output"],
            Self::PretrainSslTueg => &["--out"],
            Self::Train4StateController => &["--checkpoint"],
            Self::TrainJoint => &["--ckpt-dir", "--resume-dir"],
            Self::TrainCombined | Self::TrainL3Teacher | Self::TrainVocosDecoder => &[],
        }
    }

    /// Frozen exact options that also prefix one protected long option.
    ///
    /// Python argparse gives an exact option precedence over abbreviation.
    /// Keep these source-bound exceptions narrow so `--l` and `--res` still
    /// fail closed while legitimate `--lr` and `--resume` remain usable.
    const fn exact_protected_prefix_options(self) -> &'static [&'static str] {
        match self {
            Self::PretrainMae
            | Self::PretrainSslTueg
            | Self::Train4StateController
            | Self::TrainJoint
            | Self::TrainL3Teacher
            | Self::TrainVocosDecoder => &["--lr"],
            Self::TrainCombined => &[],
        }
    }
}

impl fmt::Display for LegacyTrainer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for LegacyTrainer {
    type Err = ParseLegacyTrainerError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|trainer| trainer.as_str() == value)
            .ok_or_else(|| ParseLegacyTrainerError(value.to_owned()))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParseLegacyTrainerError(String);

impl fmt::Display for ParseLegacyTrainerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "unknown legacy trainer {:?}", self.0)
    }
}

impl std::error::Error for ParseLegacyTrainerError {}

/// One explicit environment value admitted to the isolated trainer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LegacyEnvironment {
    name: &'static str,
    value: OsString,
}

impl LegacyEnvironment {
    pub fn new(name: &str, value: impl Into<OsString>) -> Result<Self, String> {
        let name = ALLOWED_ENVIRONMENT
            .iter()
            .copied()
            .find(|allowed| *allowed == name)
            .ok_or_else(|| format!("legacy environment variable {name:?} is not allowlisted"))?;
        Ok(Self {
            name,
            value: value.into(),
        })
    }

    pub const fn name(&self) -> &'static str {
        self.name
    }

    pub fn value(&self) -> &OsStr {
        &self.value
    }
}

/// Exact, clean checkout admitted for one allowlisted trainer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedCheckout {
    git: PathBuf,
    root: PathBuf,
    revision: String,
    trainer: LegacyTrainer,
}

impl VerifiedCheckout {
    pub fn git(&self) -> &Path {
        &self.git
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn revision(&self) -> &str {
        &self.revision
    }

    pub fn trainer(&self) -> LegacyTrainer {
        self.trainer
    }

    pub fn script(&self) -> PathBuf {
        self.root.join(self.trainer.script())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VerificationFailure {
    Interrupted,
    InvalidGit,
    InvalidRevision,
    CheckoutPath,
    RepositoryRoot,
    RevisionMismatch,
    DirtyCheckout,
    ForbiddenIndex,
    SourceMismatch,
    Allowlist,
    GitCommand,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerificationError {
    failure: VerificationFailure,
    message: String,
    interruption: Option<SignalInterruption>,
}

impl VerificationError {
    fn new(failure: VerificationFailure, message: impl Into<String>) -> Self {
        Self {
            failure,
            message: message.into(),
            interruption: None,
        }
    }

    fn from_io(failure: VerificationFailure, context: impl fmt::Display, error: io::Error) -> Self {
        if let Some(interruption) = signal_interruption(&error) {
            return Self {
                failure: VerificationFailure::Interrupted,
                message: format!("{context}: {interruption}"),
                interruption: Some(interruption),
            };
        }
        Self::new(failure, format!("{context}: {error}"))
    }

    pub const fn failure(&self) -> VerificationFailure {
        self.failure
    }

    pub fn interrupted_signal(&self) -> Option<libc::c_int> {
        self.interruption.as_ref().map(SignalInterruption::signal)
    }
}

impl fmt::Display for VerificationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for VerificationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.interruption
            .as_ref()
            .map(|interruption| interruption as &(dyn std::error::Error + 'static))
    }
}

/// Verify a checkout against Package 31's frozen source revision.
pub fn verify_checkout(
    git: impl AsRef<OsStr>,
    checkout: impl AsRef<Path>,
    trainer: LegacyTrainer,
) -> Result<VerifiedCheckout, VerificationError> {
    let git = validate_git_executable(git.as_ref())?;
    verify_checkout_at(&git, checkout, trainer, SOURCE_REVISION)
}

/// Verify a clean Git checkout at an explicit full commit identity.
///
/// Private so callers cannot substitute a different revision for Package 31's
/// frozen [`SOURCE_REVISION`]. Tests use it for hermetic fixture commits.
fn verify_checkout_at(
    git: &Path,
    checkout: impl AsRef<Path>,
    trainer: LegacyTrainer,
    required_revision: &str,
) -> Result<VerifiedCheckout, VerificationError> {
    if required_revision.len() != 40
        || !required_revision
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(VerificationError::new(
            VerificationFailure::InvalidRevision,
            "required revision must be a full 40-hex Git commit",
        ));
    }

    let root = checkout.as_ref().canonicalize().map_err(|error| {
        VerificationError::new(
            VerificationFailure::CheckoutPath,
            format!(
                "canonicalize training checkout {}: {error}",
                checkout.as_ref().display()
            ),
        )
    })?;
    if !root.is_dir() {
        return Err(VerificationError::new(
            VerificationFailure::CheckoutPath,
            format!("training checkout is not a directory: {}", root.display()),
        ));
    }

    let mut top_level = git_output_bytes(git, &root, &["rev-parse", "--show-toplevel"], None)?;
    while matches!(top_level.last(), Some(b'\n' | b'\r')) {
        top_level.pop();
    }
    let top_level = PathBuf::from(OsString::from_vec(top_level))
        .canonicalize()
        .map_err(|error| {
            VerificationError::new(
                VerificationFailure::RepositoryRoot,
                format!("canonicalize Git top-level: {error}"),
            )
        })?;
    if top_level != root {
        return Err(VerificationError::new(
            VerificationFailure::RepositoryRoot,
            format!(
                "checkout must name repository root: expected {}, got {}",
                top_level.display(),
                root.display()
            ),
        ));
    }

    let revision = git_output(git, &root, &["rev-parse", "--verify", "HEAD^{commit}"])?;
    if revision != required_revision {
        return Err(VerificationError::new(
            VerificationFailure::RevisionMismatch,
            format!(
                "training checkout revision mismatch: required {required_revision}, got {revision}"
            ),
        ));
    }

    let worktree_status = git_output(
        git,
        &root,
        &[
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignored=matching",
        ],
    )?;
    if !worktree_status.is_empty() {
        return Err(VerificationError::new(
            VerificationFailure::DirtyCheckout,
            "training checkout contains tracked, untracked, or ignored files",
        ));
    }
    verify_index_flags(git, &root)?;
    verify_python_source_tree(git, &root)?;

    let script = root.join(trainer.script());
    let metadata = script.symlink_metadata().map_err(|error| {
        VerificationError::new(
            VerificationFailure::Allowlist,
            format!("inspect allowlisted trainer {}: {error}", script.display()),
        )
    })?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(VerificationError::new(
            VerificationFailure::Allowlist,
            format!(
                "allowlisted trainer must be a regular file: {}",
                script.display()
            ),
        ));
    }
    git_output(
        git,
        &root,
        &["ls-files", "--error-unmatch", "--", trainer.script()],
    )?;

    Ok(VerifiedCheckout {
        git: git.to_owned(),
        root,
        revision,
        trainer,
    })
}

fn validate_absolute_executable(executable: &OsStr, label: &str) -> Result<PathBuf, String> {
    let requested = Path::new(executable);
    if !requested.is_absolute() {
        return Err(format!("{label} executable must be an absolute path"));
    }
    let canonical = requested.canonicalize().map_err(|error| {
        format!(
            "canonicalize {label} executable {}: {error}",
            requested.display()
        )
    })?;
    let metadata = canonical.metadata().map_err(|error| {
        format!(
            "inspect {label} executable {}: {error}",
            canonical.display()
        )
    })?;
    if !metadata.is_file() {
        return Err(format!(
            "{label} executable is not a regular file: {}",
            canonical.display()
        ));
    }
    if metadata.permissions().mode() & 0o111 == 0 {
        return Err(format!(
            "{label} executable is not executable: {}",
            canonical.display()
        ));
    }
    Ok(canonical)
}

fn configure_owned_process_group(command: &mut Command) {
    let parent = unsafe { libc::getpid() };
    // SAFETY: closure executes after fork and before exec. It calls only
    // process-control syscalls and returns the OS error directly.
    unsafe {
        command.pre_exec(move || {
            if libc::setpgid(0, 0) != 0 {
                return Err(io::Error::last_os_error());
            }
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                return Err(io::Error::last_os_error());
            }
            if libc::getppid() != parent {
                libc::_exit(127);
            }
            Ok(())
        });
    }
}

struct SupervisedChild {
    child: Child,
    process_group: libc::pid_t,
    active: bool,
    _process_lock: MutexGuard<'static, ()>,
}

impl SupervisedChild {
    fn spawn(mut command: Command) -> Result<Self, io::Error> {
        let process_lock = PROCESS_SUPERVISION_LOCK
            .lock()
            .map_err(|_| io::Error::other("legacy subprocess supervision lock is poisoned"))?;
        configure_owned_process_group(&mut command);
        let mut child = command.spawn()?;
        let process_group = child
            .id()
            .try_into()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "child PID exceeds pid_t"))?;
        if ACTIVE_CHILD_PROCESS_GROUP
            .compare_exchange(0, process_group, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            unsafe {
                libc::kill(-process_group, libc::SIGKILL);
            }
            let _ = child.wait();
            return Err(io::Error::other(
                "legacy subprocess ownership invariant violated",
            ));
        }
        Ok(Self {
            child,
            process_group,
            active: true,
            _process_lock: process_lock,
        })
    }

    fn try_wait(&mut self) -> Result<Option<ExitStatus>, io::Error> {
        self.child.try_wait()
    }

    fn wait(mut self) -> Result<ExitStatus, io::Error> {
        let mut forwarded_signal = None;
        let mut termination_deadline = None;
        loop {
            if forwarded_signal.is_none() {
                let signal = PENDING_TERMINATION_SIGNAL.swap(0, Ordering::AcqRel);
                if signal != 0 {
                    self.forward_signal(signal);
                    forwarded_signal = Some(signal);
                    termination_deadline = Instant::now().checked_add(TERMINATION_GRACE);
                }
            }
            if let Some(status) = self.child.try_wait()? {
                self.finish_group();
                return Ok(forwarded_signal
                    .map_or(status, |signal| ExitStatus::from_raw((128 + signal) << 8)));
            }
            if termination_deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                self.terminate_and_wait();
                let signal = forwarded_signal.expect("termination deadline requires signal");
                return Ok(ExitStatus::from_raw((128 + signal) << 8));
            }
            std::thread::sleep(POLL_INTERVAL);
        }
    }

    fn terminate_and_wait(&mut self) {
        if self.active {
            self.forward_signal(libc::SIGKILL);
            let _ = self.child.wait();
            self.finish_group();
        }
    }

    fn complete(mut self, status: ExitStatus) -> ExitStatus {
        self.finish_group();
        status
    }

    fn forward_signal(&self, signal: libc::c_int) {
        unsafe {
            libc::kill(-self.process_group, signal);
        }
    }

    fn finish_group(&mut self) {
        if !self.active {
            return;
        }
        if process_group_exists(self.process_group) {
            self.forward_signal(libc::SIGTERM);
            let deadline = Instant::now() + DESCENDANT_GRACE;
            while Instant::now() < deadline && process_group_exists(self.process_group) {
                std::thread::sleep(POLL_INTERVAL);
            }
            if process_group_exists(self.process_group) {
                self.forward_signal(libc::SIGKILL);
                let deadline = Instant::now() + DESCENDANT_GRACE;
                while Instant::now() < deadline && process_group_exists(self.process_group) {
                    std::thread::sleep(POLL_INTERVAL);
                }
            }
        }
        self.release();
    }

    fn release(&mut self) {
        if self.active {
            let _ = ACTIVE_CHILD_PROCESS_GROUP.compare_exchange(
                self.process_group,
                0,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
            self.active = false;
        }
    }
}

fn process_group_exists(process_group: libc::pid_t) -> bool {
    let result = unsafe { libc::kill(-process_group, 0) };
    result == 0 || io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

impl Drop for SupervisedChild {
    fn drop(&mut self) {
        self.terminate_and_wait();
    }
}

fn supervised_status(command: Command) -> Result<ExitStatus, io::Error> {
    SupervisedChild::spawn(command)?.wait()
}

fn validate_git_executable(git: &OsStr) -> Result<PathBuf, VerificationError> {
    let git = validate_absolute_executable(git, "Git")
        .map_err(|message| VerificationError::new(VerificationFailure::InvalidGit, message))?;
    let mut command = git_base_command(&git);
    command.arg("--version");
    let (status, stdout, _stderr) =
        bounded_command_output(command, 4096, HANDSHAKE_TIMEOUT, "Git handshake").map_err(
            |error| {
                VerificationError::from_io(
                    VerificationFailure::InvalidGit,
                    format_args!("run Git handshake {}", git.display()),
                    error,
                )
            },
        )?;
    let version = String::from_utf8_lossy(&stdout);
    if !status.success() || !version.starts_with("git version ") {
        return Err(VerificationError::new(
            VerificationFailure::InvalidGit,
            format!("Git handshake failed for {}", git.display()),
        ));
    }
    Ok(git)
}

fn git_base_command(git: &Path) -> Command {
    let mut command = Command::new(git);
    command
        .env_clear()
        .env("GIT_ATTR_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_SYSTEM", "/dev/null")
        .env("GIT_PAGER", "cat")
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("LC_ALL", "C");
    command
}

fn git_command(git: &Path, root: &Path) -> Command {
    let mut command = git_base_command(git);
    command
        .arg("--no-optional-locks")
        .args(["-c", "core.fsmonitor=false"])
        .args(["-c", "core.hooksPath=/dev/null"])
        .args(["-c", "diff.external="])
        .arg("-C")
        .arg(root);
    command
}

fn git_output(git: &Path, root: &Path, args: &[&str]) -> Result<String, VerificationError> {
    let output = git_output_bytes(git, root, args, None)?;
    String::from_utf8(output)
        .map(|value| value.trim().to_owned())
        .map_err(|error| {
            VerificationError::new(
                VerificationFailure::GitCommand,
                format!("Git output is not UTF-8: {error}"),
            )
        })
}

fn verify_index_flags(git: &Path, root: &Path) -> Result<(), VerificationError> {
    let flags = git_output(git, root, &["ls-files", "-v"])?;
    if flags.is_empty() {
        return Err(VerificationError::new(
            VerificationFailure::ForbiddenIndex,
            "training checkout contains no tracked files",
        ));
    }
    if let Some(line) = flags.lines().find(|line| !line.starts_with("H ")) {
        return Err(VerificationError::new(
            VerificationFailure::ForbiddenIndex,
            format!("training checkout uses forbidden index flag: {line}"),
        ));
    }
    Ok(())
}

fn verify_python_source_tree(git: &Path, root: &Path) -> Result<(), VerificationError> {
    let index = git_output_bytes(
        git,
        root,
        &["ls-files", "-s", "-z", "--", "python/lamquant"],
        None,
    )?;
    let mut paths = Vec::new();
    let mut expected = Vec::new();
    for record in index
        .split(|byte| *byte == 0)
        .filter(|record| !record.is_empty())
    {
        let record = std::str::from_utf8(record).map_err(|error| {
            VerificationError::new(
                VerificationFailure::SourceMismatch,
                format!("tracked source path is not UTF-8: {error}"),
            )
        })?;
        let (metadata, path) = record.split_once('\t').ok_or_else(|| {
            VerificationError::new(
                VerificationFailure::SourceMismatch,
                "malformed Git ls-files source record",
            )
        })?;
        let mut metadata = metadata.split_ascii_whitespace();
        let mode = metadata.next().unwrap_or_default();
        let object_id = metadata.next().unwrap_or_default();
        let stage = metadata.next().unwrap_or_default();
        if !matches!(mode, "100644" | "100755")
            || object_id.is_empty()
            || stage != "0"
            || metadata.next().is_some()
        {
            return Err(VerificationError::new(
                VerificationFailure::SourceMismatch,
                format!("unsupported tracked source record: {record}"),
            ));
        }
        if path.contains('\n') {
            return Err(VerificationError::new(
                VerificationFailure::SourceMismatch,
                "tracked Python source path contains newline",
            ));
        }
        paths.push(path);
        expected.push(object_id);
    }
    if paths.is_empty() {
        return Err(VerificationError::new(
            VerificationFailure::SourceMismatch,
            "training checkout has no tracked python/lamquant source",
        ));
    }

    let mut stdin = paths.join("\n");
    stdin.push('\n');
    let hashes = git_output_bytes(
        git,
        root,
        &["hash-object", "--no-filters", "--stdin-paths"],
        Some(stdin.as_bytes()),
    )?;
    let actual = std::str::from_utf8(&hashes)
        .map_err(|error| {
            VerificationError::new(
                VerificationFailure::SourceMismatch,
                format!("Git hash-object output is not UTF-8: {error}"),
            )
        })?
        .lines()
        .collect::<Vec<_>>();
    if actual.len() != expected.len() {
        return Err(VerificationError::new(
            VerificationFailure::SourceMismatch,
            format!(
                "tracked Python source hash count mismatch: expected {}, got {}",
                expected.len(),
                actual.len()
            ),
        ));
    }
    if let Some((path, _, _)) = paths
        .iter()
        .zip(expected.iter())
        .zip(actual.iter())
        .find_map(|((path, expected), actual)| {
            (expected != actual).then_some((*path, *expected, *actual))
        })
    {
        return Err(VerificationError::new(
            VerificationFailure::SourceMismatch,
            format!("tracked Python source does not match HEAD: {path}"),
        ));
    }
    Ok(())
}

fn git_output_bytes(
    git: &Path,
    root: &Path,
    args: &[&str],
    stdin: Option<&[u8]>,
) -> Result<Vec<u8>, VerificationError> {
    let mut command = git_command(git, root);
    command.args(args);
    let (status, stdout, stderr) = supervised_command_output(
        command,
        MAX_GIT_TEXT_BYTES,
        64 * 1024,
        GIT_COMMAND_TIMEOUT,
        stdin.map(ToOwned::to_owned),
        "Git command",
    )
    .map_err(|error| {
        VerificationError::from_io(
            VerificationFailure::GitCommand,
            format_args!("run Git {args:?}"),
            error,
        )
    })?;
    if !status.success() {
        return Err(VerificationError::new(
            VerificationFailure::GitCommand,
            format!(
                "Git {args:?} failed ({status}): {}",
                String::from_utf8_lossy(&stderr).trim()
            ),
        ));
    }
    Ok(stdout)
}

fn read_bounded(
    reader: impl Read,
    limit: usize,
    label: &'static str,
) -> Result<Vec<u8>, io::Error> {
    let mut bytes = Vec::with_capacity(limit.min(64 * 1024));
    reader
        .take(limit.saturating_add(1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{label} exceeds {limit} bytes"),
        ));
    }
    Ok(bytes)
}

fn read_private_regular_file(
    path: &Path,
    limit: usize,
    label: &'static str,
) -> Result<Vec<u8>, io::Error> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file()
        || metadata.nlink() != 1
        || metadata.uid() != unsafe { libc::geteuid() }
        || metadata.permissions().mode() & 0o277 != 0
        || metadata.len() > limit as u64
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{label} is not a bounded private read-only regular file"),
        ));
    }
    read_bounded(file, limit, label)
}

#[derive(Debug)]
pub enum LaunchError {
    Interrupted(SignalInterruption),
    Verification(VerificationError),
    ReservedArgument(OsString),
    InvalidArguments(String),
    InvalidEnvironment(String),
    InvalidPythonInterpreter(String),
    InvalidSandbox(String),
    DependencyPreflight(String),
    Workspace(io::Error),
    Spawn(std::io::Error),
}

impl fmt::Display for LaunchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Interrupted(interruption) => write!(formatter, "{interruption}"),
            Self::Verification(error) => write!(formatter, "{error}"),
            Self::ReservedArgument(argument) => write!(
                formatter,
                "legacy launcher owns reserved argument {:?}",
                argument
            ),
            Self::InvalidArguments(message) => formatter.write_str(message),
            Self::InvalidEnvironment(message) => formatter.write_str(message),
            Self::InvalidPythonInterpreter(message) => formatter.write_str(message),
            Self::InvalidSandbox(message) => formatter.write_str(message),
            Self::DependencyPreflight(message) => {
                write!(formatter, "legacy dependency preflight failed: {message}")
            }
            Self::Workspace(error) => {
                write!(formatter, "prepare persistent legacy workspace: {error}")
            }
            Self::Spawn(error) => write!(formatter, "launch legacy trainer: {error}"),
        }
    }
}

impl std::error::Error for LaunchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Interrupted(interruption) => Some(interruption),
            Self::Verification(error) => Some(error),
            Self::Workspace(error) => Some(error),
            Self::Spawn(error) => Some(error),
            Self::ReservedArgument(_)
            | Self::InvalidArguments(_)
            | Self::InvalidEnvironment(_)
            | Self::InvalidPythonInterpreter(_)
            | Self::InvalidSandbox(_)
            | Self::DependencyPreflight(_) => None,
        }
    }
}

impl LaunchError {
    pub fn interrupted_signal(&self) -> Option<libc::c_int> {
        match self {
            Self::Interrupted(interruption) => Some(interruption.signal()),
            Self::Verification(error) => error.interrupted_signal(),
            Self::Workspace(error) | Self::Spawn(error) => {
                signal_interruption(error).map(|interruption| interruption.signal())
            }
            Self::ReservedArgument(_)
            | Self::InvalidArguments(_)
            | Self::InvalidEnvironment(_)
            | Self::InvalidPythonInterpreter(_)
            | Self::InvalidSandbox(_)
            | Self::DependencyPreflight(_) => None,
        }
    }

    pub fn exit_code(&self) -> libc::c_int {
        match self {
            Self::Interrupted(interruption) => interruption.exit_code(),
            Self::Verification(error) => error
                .interruption
                .as_ref()
                .map_or(1, SignalInterruption::exit_code),
            Self::Workspace(error) | Self::Spawn(error) => signal_interruption(error)
                .as_ref()
                .map_or(1, SignalInterruption::exit_code),
            Self::ReservedArgument(_)
            | Self::InvalidArguments(_)
            | Self::InvalidEnvironment(_)
            | Self::InvalidPythonInterpreter(_)
            | Self::InvalidSandbox(_)
            | Self::DependencyPreflight(_) => 1,
        }
    }
}

impl From<VerificationError> for LaunchError {
    fn from(error: VerificationError) -> Self {
        Self::Verification(error)
    }
}

/// Verify Package 31's frozen checkout and launch one retired trainer.
pub fn launch(
    git: impl AsRef<OsStr>,
    checkout: impl AsRef<Path>,
    trainer: LegacyTrainer,
    python: impl AsRef<OsStr>,
    workspace: impl AsRef<Path>,
    environment: &[LegacyEnvironment],
    args: &[OsString],
) -> Result<ExitStatus, LaunchError> {
    validate_legacy_arguments(trainer, args)?;
    validate_environment(environment)?;
    let verified = verify_checkout(git, checkout, trainer)?;
    launch_verified(&verified, python, workspace, environment, args)
}

/// Launch an already verified checkout after repeating verification.
fn launch_verified(
    verified: &VerifiedCheckout,
    python: impl AsRef<OsStr>,
    workspace: impl AsRef<Path>,
    environment: &[LegacyEnvironment],
    args: &[OsString],
) -> Result<ExitStatus, LaunchError> {
    launch_verified_at(
        verified,
        SOURCE_REVISION,
        python,
        workspace,
        environment,
        args,
    )
}

fn launch_verified_at(
    verified: &VerifiedCheckout,
    required_revision: &str,
    python: impl AsRef<OsStr>,
    workspace: impl AsRef<Path>,
    environment: &[LegacyEnvironment],
    args: &[OsString],
) -> Result<ExitStatus, LaunchError> {
    let verified = verify_checkout_at(
        verified.git(),
        verified.root(),
        verified.trainer(),
        required_revision,
    )?;
    validate_legacy_arguments(verified.trainer(), args)?;
    validate_environment(environment)?;

    let python = validate_python_interpreter(python.as_ref())?;
    let sandbox = validate_sandbox_executable()?;
    let workspace = FrozenWorkspace::open_or_create(&verified, workspace.as_ref())
        .map_err(map_workspace_error)?;
    let mut managed_args = verified.trainer().managed_arguments(workspace.root());
    managed_args.extend_from_slice(args);
    let dependency_modules = if verified.revision() == SOURCE_REVISION {
        verified.trainer().dependency_modules(&managed_args)
    } else {
        Vec::new()
    };
    preflight_dependency_environment(
        &workspace,
        &verified,
        &sandbox,
        &python,
        environment,
        dependency_modules,
    )?;
    sandbox.revalidate()?;
    python.revalidate()?;
    let status = supervised_status(python_command(
        &workspace,
        &sandbox,
        &python,
        environment,
        &managed_args,
    ))
    .map_err(map_spawn_error)?;
    workspace
        .revalidate(&verified)
        .map_err(map_workspace_error)?;
    validate_managed_destinations(&workspace).map_err(LaunchError::Workspace)?;
    Ok(status)
}

fn map_workspace_error(error: io::Error) -> LaunchError {
    map_launch_io_error(error, LaunchError::Workspace)
}

fn map_spawn_error(error: io::Error) -> LaunchError {
    map_launch_io_error(error, LaunchError::Spawn)
}

fn validate_legacy_arguments(trainer: LegacyTrainer, args: &[OsString]) -> Result<(), LaunchError> {
    let dependency_flags = ["--logger", "--lr-schedule", "--target-source", "--dac-init"];
    let required_input_flags = ["--lma-root", "--split-manifest"];
    let mut dependency_counts = BTreeMap::<&str, usize>::new();
    let mut required_input_counts = BTreeMap::<&str, usize>::new();
    for argument in args {
        let Some(argument) = argument.to_str() else {
            continue;
        };
        let head = argument.split_once('=').map_or(argument, |(head, _)| head);
        if !head.starts_with("--") {
            continue;
        }
        if head.starts_with("--training-") || "--training-".starts_with(head) {
            return Err(LaunchError::InvalidArguments(format!(
                "ABIR snapshot option {head:?} is forbidden in LMA-direct rollback mode"
            )));
        }
        for flag in required_input_flags {
            if flag.starts_with(head) {
                if head != flag {
                    return Err(LaunchError::InvalidArguments(format!(
                        "abbreviated LMA-direct input option {head:?} is forbidden; use {flag}"
                    )));
                }
                let count = required_input_counts.entry(flag).or_default();
                *count += 1;
                if *count > 1 {
                    return Err(LaunchError::InvalidArguments(format!(
                        "duplicate LMA-direct input option {flag}"
                    )));
                }
            }
        }
        let exact_prefix_option = trainer.exact_protected_prefix_options().contains(&head)
            || (trainer == LegacyTrainer::TrainJoint && head == "--resume");
        if exact_prefix_option {
            continue;
        }
        if let Some(flag) = trainer
            .managed_flags()
            .iter()
            .find(|flag| flag.starts_with(head))
        {
            return Err(LaunchError::ReservedArgument(OsString::from(format!(
                "{head} (resolves to launcher-owned {flag})"
            ))));
        }
        for flag in dependency_flags {
            if flag.starts_with(head) {
                if head != flag {
                    return Err(LaunchError::InvalidArguments(format!(
                        "abbreviated dependency-affecting option {head:?} is forbidden; use {flag}"
                    )));
                }
                let count = dependency_counts.entry(flag).or_default();
                *count += 1;
                if *count > 1 {
                    return Err(LaunchError::InvalidArguments(format!(
                        "duplicate dependency-affecting option {flag}"
                    )));
                }
            }
        }
    }
    for flag in required_input_flags {
        if required_input_counts.get(flag) != Some(&1) || !argument_has_non_option_value(args, flag)
        {
            return Err(LaunchError::InvalidArguments(format!(
                "{trainer} rollback requires exactly one {flag} with a non-empty value"
            )));
        }
    }
    Ok(())
}

fn argument_has_non_option_value(args: &[OsString], flag: &str) -> bool {
    args.iter().enumerate().any(|(index, argument)| {
        let Some(argument) = argument.to_str() else {
            return false;
        };
        if argument == flag {
            return args
                .get(index + 1)
                .and_then(|value| value.to_str())
                .is_some_and(|value| !value.is_empty() && !value.starts_with("--"));
        }
        argument
            .strip_prefix(flag)
            .and_then(|tail| tail.strip_prefix('='))
            .is_some_and(|value| !value.is_empty())
    })
}

fn validate_managed_destinations(workspace: &FrozenWorkspace) -> Result<(), io::Error> {
    for root in workspace.trainer.writable_directories(workspace.root()) {
        ensure_private_directory(workspace.root(), &root)?;
    }
    // Bubblewrap exposes the whole workspace through one writable bind, then
    // overlays frozen source resources read-only. Scan that exact writable
    // closure once so home, caches, run-derived outputs, and trainer-specific
    // artifact roots cannot hide preplanted links or special files.
    let mut entries = 0;
    validate_private_output_tree(workspace.root(), workspace.root(), 0, &mut entries)?;
    let arguments = workspace.trainer.managed_arguments(workspace.root());
    for pair in arguments.chunks_exact(2) {
        let flag = pair[0].to_string_lossy();
        let destination = Path::new(&pair[1]);
        let parent = destination
            .parent()
            .ok_or_else(|| io::Error::other("managed destination has no parent"))?;
        ensure_private_directory(workspace.root(), parent)?;
        match destination.symlink_metadata() {
            Ok(metadata) if flag == "--ckpt-dir" || flag == "--resume-dir" => {
                if !metadata.is_dir() || metadata.file_type().is_symlink() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "managed output directory is not a real directory: {}",
                            destination.display()
                        ),
                    ));
                }
            }
            Ok(metadata) => {
                if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.nlink() != 1
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "managed output file is not a private regular file: {}",
                            destination.display()
                        ),
                    ));
                }
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn validate_private_output_tree(
    workspace: &Path,
    path: &Path,
    depth: usize,
    entries: &mut usize,
) -> Result<(), io::Error> {
    if depth > MAX_WORKSPACE_VALIDATION_DEPTH {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "legacy output tree exceeds maximum validation depth",
        ));
    }
    path.strip_prefix(workspace).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("legacy output tree escaped workspace: {}", path.display()),
        )
    })?;
    let root_metadata = path.symlink_metadata()?;
    if !root_metadata.is_dir() || root_metadata.file_type().is_symlink() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "legacy output root is not a real directory: {}",
                path.display()
            ),
        ));
    }
    for entry in fs::read_dir(path)? {
        *entries = entries
            .checked_add(1)
            .ok_or_else(|| io::Error::other("legacy output entry count overflow"))?;
        if *entries > MAX_WORKSPACE_VALIDATION_ENTRIES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "legacy output tree exceeds maximum validation entry count",
            ));
        }
        let entry = entry?;
        let path = entry.path();
        let metadata = path.symlink_metadata()?;
        if metadata.file_type().is_symlink() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("legacy output tree contains symlink: {}", path.display()),
            ));
        }
        if metadata.is_dir() {
            validate_private_output_tree(workspace, &path, depth + 1, entries)?;
        } else if !metadata.is_file() || metadata.nlink() != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "legacy output tree contains non-private entry: {}",
                    path.display()
                ),
            ));
        }
    }
    Ok(())
}

fn validate_environment(environment: &[LegacyEnvironment]) -> Result<(), LaunchError> {
    let mut names = BTreeSet::new();
    for variable in environment {
        if !ALLOWED_ENVIRONMENT.contains(&variable.name()) {
            return Err(LaunchError::InvalidEnvironment(format!(
                "legacy environment variable {:?} is not allowlisted",
                variable.name()
            )));
        }
        if !names.insert(variable.name()) {
            return Err(LaunchError::InvalidEnvironment(format!(
                "duplicate legacy environment variable {:?}",
                variable.name()
            )));
        }
    }
    Ok(())
}

#[derive(Debug)]
struct ValidatedPythonEnvironment {
    invocation_root: PathBuf,
    target_root: PathBuf,
    target_device: u64,
    target_inode: u64,
    config_device: u64,
    config_inode: u64,
    config_sha256: String,
    root: fs::File,
}

#[derive(Debug)]
struct ValidatedPythonInterpreter {
    invocation: PathBuf,
    target: PathBuf,
    target_device: u64,
    target_inode: u64,
    target_sha256: String,
    sealed_image: fs::File,
    environment: Option<ValidatedPythonEnvironment>,
}

#[derive(Debug)]
struct ValidatedSandboxExecutable {
    invocation: PathBuf,
    target: PathBuf,
    target_device: u64,
    target_inode: u64,
    target_sha256: String,
    executable: fs::File,
}

impl ValidatedSandboxExecutable {
    fn command(&self) -> Command {
        Command::new(format!("/proc/self/fd/{}", self.executable.as_raw_fd()))
    }

    fn revalidate(&self) -> Result<(), LaunchError> {
        let target = self.invocation.canonicalize().map_err(|error| {
            LaunchError::InvalidSandbox(format!(
                "canonicalize bubblewrap sandbox {}: {error}",
                self.invocation.display()
            ))
        })?;
        if target != self.target {
            return Err(LaunchError::InvalidSandbox(format!(
                "bubblewrap sandbox target changed: {}",
                self.invocation.display()
            )));
        }
        let path_executable = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(&target)
            .map_err(|error| {
                LaunchError::InvalidSandbox(format!(
                    "open bubblewrap sandbox {}: {error}",
                    target.display()
                ))
            })?;
        let metadata = path_executable.metadata().map_err(|error| {
            LaunchError::InvalidSandbox(format!(
                "inspect bubblewrap sandbox {}: {error}",
                target.display()
            ))
        })?;
        if !metadata.is_file()
            || metadata.permissions().mode() & 0o111 == 0
            || metadata.len() == 0
            || metadata.len() > MAX_EXECUTABLE_BYTES
            || metadata.dev() != self.target_device
            || metadata.ino() != self.target_inode
            || metadata.uid() != 0
            || metadata.permissions().mode() & 0o022 != 0
        {
            return Err(LaunchError::InvalidSandbox(format!(
                "bubblewrap sandbox identity, ownership, or permissions changed: {}",
                target.display()
            )));
        }
        let mut executable = self.executable.try_clone().map_err(|error| {
            LaunchError::InvalidSandbox(format!(
                "duplicate bubblewrap sandbox descriptor {}: {error}",
                target.display()
            ))
        })?;
        let held_metadata = executable.metadata().map_err(|error| {
            LaunchError::InvalidSandbox(format!(
                "inspect held bubblewrap sandbox {}: {error}",
                target.display()
            ))
        })?;
        if held_metadata.dev() != self.target_device || held_metadata.ino() != self.target_inode {
            return Err(LaunchError::InvalidSandbox(format!(
                "held bubblewrap sandbox identity changed: {}",
                target.display()
            )));
        }
        let target_sha256 = file_sha256_from(&mut executable).map_err(|error| {
            LaunchError::InvalidSandbox(format!(
                "hash bubblewrap sandbox {}: {error}",
                target.display()
            ))
        })?;
        if target_sha256 != self.target_sha256 {
            return Err(LaunchError::InvalidSandbox(format!(
                "bubblewrap sandbox content changed: {}",
                target.display()
            )));
        }
        Ok(())
    }
}

fn validate_sandbox_executable() -> Result<ValidatedSandboxExecutable, LaunchError> {
    let invocation = PathBuf::from(SANDBOX_EXECUTABLE);
    let target = validate_absolute_executable(invocation.as_os_str(), "bubblewrap sandbox")
        .map_err(LaunchError::InvalidSandbox)?;
    let mut executable = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(&target)
        .map_err(|error| {
            LaunchError::InvalidSandbox(format!(
                "open bubblewrap sandbox {}: {error}",
                target.display()
            ))
        })?;
    let metadata = executable.metadata().map_err(|error| {
        LaunchError::InvalidSandbox(format!(
            "inspect bubblewrap sandbox {}: {error}",
            target.display()
        ))
    })?;
    if metadata.uid() != 0 || metadata.permissions().mode() & 0o022 != 0 {
        return Err(LaunchError::InvalidSandbox(format!(
            "bubblewrap sandbox must be root-owned and not group/other writable: {}",
            target.display()
        )));
    }
    if metadata.len() == 0 || metadata.len() > MAX_EXECUTABLE_BYTES {
        return Err(LaunchError::InvalidSandbox(format!(
            "bubblewrap sandbox exceeds executable size limit: {}",
            target.display()
        )));
    }
    let target_device = metadata.dev();
    let target_inode = metadata.ino();
    let target_sha256 = file_sha256_from(&mut executable).map_err(|error| {
        LaunchError::InvalidSandbox(format!(
            "hash bubblewrap sandbox {}: {error}",
            SANDBOX_EXECUTABLE
        ))
    })?;
    let validated = ValidatedSandboxExecutable {
        invocation,
        target,
        target_device,
        target_inode,
        target_sha256,
        executable,
    };
    validated.revalidate()?;
    Ok(validated)
}

impl ValidatedPythonInterpreter {
    fn invocation(&self) -> &Path {
        &self.invocation
    }

    fn sealed_fd(&self) -> libc::c_int {
        self.sealed_image.as_raw_fd()
    }

    fn environment(&self) -> Option<&ValidatedPythonEnvironment> {
        self.environment.as_ref()
    }

    fn direct_command(&self) -> Command {
        let fd = self.sealed_fd();
        let mut command = Command::new(format!("/proc/self/fd/{fd}"));
        command.arg0(&self.invocation);
        inherit_fd_across_exec(&mut command, fd);
        command
    }

    fn revalidate(&self) -> Result<(), LaunchError> {
        let target = self.invocation.canonicalize().map_err(|error| {
            LaunchError::InvalidPythonInterpreter(format!(
                "canonicalize Python interpreter {}: {error}",
                self.invocation.display()
            ))
        })?;
        if target != self.target {
            return Err(LaunchError::InvalidPythonInterpreter(format!(
                "Python interpreter target changed: {}",
                self.invocation.display()
            )));
        }
        let mut executable = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(&target)
            .map_err(|error| {
                LaunchError::InvalidPythonInterpreter(format!(
                    "open Python interpreter {}: {error}",
                    target.display()
                ))
            })?;
        let metadata = executable.metadata().map_err(|error| {
            LaunchError::InvalidPythonInterpreter(format!(
                "inspect Python interpreter {}: {error}",
                target.display()
            ))
        })?;
        if !metadata.is_file()
            || metadata.permissions().mode() & 0o111 == 0
            || metadata.len() == 0
            || metadata.len() > MAX_EXECUTABLE_BYTES
            || metadata.dev() != self.target_device
            || metadata.ino() != self.target_inode
        {
            return Err(LaunchError::InvalidPythonInterpreter(format!(
                "Python interpreter target identity changed or is no longer executable: {}",
                target.display()
            )));
        }
        let target_sha256 = file_sha256_from(&mut executable).map_err(|error| {
            LaunchError::InvalidPythonInterpreter(format!(
                "hash Python interpreter {}: {error}",
                target.display()
            ))
        })?;
        if target_sha256 != self.target_sha256 {
            return Err(LaunchError::InvalidPythonInterpreter(format!(
                "Python interpreter content changed: {}",
                target.display()
            )));
        }
        let mut sealed_image = self.sealed_image.try_clone().map_err(|error| {
            LaunchError::InvalidPythonInterpreter(format!(
                "duplicate sealed Python image descriptor: {error}"
            ))
        })?;
        let sealed_sha256 = file_sha256_from(&mut sealed_image).map_err(|error| {
            LaunchError::InvalidPythonInterpreter(format!("hash sealed Python image: {error}"))
        })?;
        if sealed_sha256 != self.target_sha256 {
            return Err(LaunchError::InvalidPythonInterpreter(
                "sealed Python image content changed".to_owned(),
            ));
        }
        if let Some(environment) = &self.environment {
            environment.revalidate()?;
        }
        Ok(())
    }
}

impl ValidatedPythonEnvironment {
    fn root_fd(&self) -> libc::c_int {
        self.root.as_raw_fd()
    }

    fn invocation_root(&self) -> &Path {
        &self.invocation_root
    }

    fn revalidate(&self) -> Result<(), LaunchError> {
        let target_root = self.invocation_root.canonicalize().map_err(|error| {
            LaunchError::InvalidPythonInterpreter(format!(
                "canonicalize Python environment {}: {error}",
                self.invocation_root.display()
            ))
        })?;
        if target_root != self.target_root {
            return Err(LaunchError::InvalidPythonInterpreter(format!(
                "Python environment target changed: {}",
                self.invocation_root.display()
            )));
        }
        let root = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_CLOEXEC | libc::O_DIRECTORY | libc::O_NOFOLLOW)
            .open(&target_root)
            .map_err(|error| {
                LaunchError::InvalidPythonInterpreter(format!(
                    "open Python environment {}: {error}",
                    target_root.display()
                ))
            })?;
        let root_metadata = root.metadata().map_err(|error| {
            LaunchError::InvalidPythonInterpreter(format!(
                "inspect Python environment {}: {error}",
                target_root.display()
            ))
        })?;
        let held_metadata = self.root.metadata().map_err(|error| {
            LaunchError::InvalidPythonInterpreter(format!(
                "inspect held Python environment {}: {error}",
                target_root.display()
            ))
        })?;
        if !root_metadata.is_dir()
            || root_metadata.dev() != self.target_device
            || root_metadata.ino() != self.target_inode
            || held_metadata.dev() != self.target_device
            || held_metadata.ino() != self.target_inode
        {
            return Err(LaunchError::InvalidPythonInterpreter(format!(
                "Python environment identity changed: {}",
                target_root.display()
            )));
        }

        let config = self.invocation_root.join("pyvenv.cfg");
        let mut config_file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(&config)
            .map_err(|error| {
                LaunchError::InvalidPythonInterpreter(format!(
                    "open Python environment config {}: {error}",
                    config.display()
                ))
            })?;
        let metadata = config_file.metadata().map_err(|error| {
            LaunchError::InvalidPythonInterpreter(format!(
                "inspect Python environment config {}: {error}",
                config.display()
            ))
        })?;
        if !metadata.is_file()
            || metadata.len() == 0
            || metadata.len() > MAX_VENV_CONFIG_BYTES
            || metadata.dev() != self.config_device
            || metadata.ino() != self.config_inode
        {
            return Err(LaunchError::InvalidPythonInterpreter(format!(
                "Python environment config identity changed: {}",
                config.display()
            )));
        }
        let sha256 = file_sha256_from(&mut config_file).map_err(|error| {
            LaunchError::InvalidPythonInterpreter(format!(
                "hash Python environment config {}: {error}",
                config.display()
            ))
        })?;
        if sha256 != self.config_sha256 {
            return Err(LaunchError::InvalidPythonInterpreter(format!(
                "Python environment config changed: {}",
                config.display()
            )));
        }
        Ok(())
    }
}

fn validate_python_environment(
    requested: &Path,
) -> Result<Option<ValidatedPythonEnvironment>, LaunchError> {
    let parent = requested.parent();
    let roots = [parent, parent.and_then(Path::parent)];
    for root in roots.into_iter().flatten() {
        let config = root.join("pyvenv.cfg");
        let link_metadata = match config.symlink_metadata() {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(LaunchError::InvalidPythonInterpreter(format!(
                    "inspect Python environment config {}: {error}",
                    config.display()
                )))
            }
        };
        if link_metadata.file_type().is_symlink() || !link_metadata.is_file() {
            return Err(LaunchError::InvalidPythonInterpreter(format!(
                "Python environment config must be a regular non-symlink file: {}",
                config.display()
            )));
        }
        let target_root = root.canonicalize().map_err(|error| {
            LaunchError::InvalidPythonInterpreter(format!(
                "canonicalize Python environment {}: {error}",
                root.display()
            ))
        })?;
        let root_file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_CLOEXEC | libc::O_DIRECTORY | libc::O_NOFOLLOW)
            .open(&target_root)
            .map_err(|error| {
                LaunchError::InvalidPythonInterpreter(format!(
                    "open Python environment {}: {error}",
                    target_root.display()
                ))
            })?;
        let root_metadata = root_file.metadata().map_err(|error| {
            LaunchError::InvalidPythonInterpreter(format!(
                "inspect Python environment {}: {error}",
                target_root.display()
            ))
        })?;
        if !root_metadata.is_dir() {
            return Err(LaunchError::InvalidPythonInterpreter(format!(
                "Python environment root is not a directory: {}",
                target_root.display()
            )));
        }
        let mut config_file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(&config)
            .map_err(|error| {
                LaunchError::InvalidPythonInterpreter(format!(
                    "open Python environment config {}: {error}",
                    config.display()
                ))
            })?;
        let config_metadata = config_file.metadata().map_err(|error| {
            LaunchError::InvalidPythonInterpreter(format!(
                "inspect Python environment config {}: {error}",
                config.display()
            ))
        })?;
        if config_metadata.len() == 0 || config_metadata.len() > MAX_VENV_CONFIG_BYTES {
            return Err(LaunchError::InvalidPythonInterpreter(format!(
                "Python environment config exceeds size limit: {}",
                config.display()
            )));
        }
        let config_sha256 = file_sha256_from(&mut config_file).map_err(|error| {
            LaunchError::InvalidPythonInterpreter(format!(
                "hash Python environment config {}: {error}",
                config.display()
            ))
        })?;
        let environment = ValidatedPythonEnvironment {
            invocation_root: root.to_owned(),
            target_root,
            target_device: root_metadata.dev(),
            target_inode: root_metadata.ino(),
            config_device: config_metadata.dev(),
            config_inode: config_metadata.ino(),
            config_sha256,
            root: root_file,
        };
        environment.revalidate()?;
        return Ok(Some(environment));
    }
    Ok(None)
}

fn inherit_fd_across_exec(command: &mut Command, fd: libc::c_int) {
    // SAFETY: fcntl is async-signal-safe. Closure changes only inherited
    // descriptor flags after fork and before exec.
    unsafe {
        command.pre_exec(move || {
            let flags = libc::fcntl(fd, libc::F_GETFD);
            if flags < 0 {
                return Err(io::Error::last_os_error());
            }
            if libc::fcntl(fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC) < 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

fn sealed_executable_snapshot(
    source: &mut fs::File,
    label: &'static str,
) -> Result<fs::File, io::Error> {
    let source_bytes = source.metadata()?.len();
    if source_bytes == 0 || source_bytes > MAX_EXECUTABLE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{label} exceeds executable size limit"),
        ));
    }
    let name = CString::new(format!("lamquant-{label}"))
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid memfd label"))?;
    let fd =
        unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    let mut snapshot = unsafe { fs::File::from_raw_fd(fd) };
    source.seek(SeekFrom::Start(0))?;
    let copied = io::copy(
        &mut source.take(MAX_EXECUTABLE_BYTES.saturating_add(1)),
        &mut snapshot,
    )?;
    if copied != source_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{label} changed size while being sealed"),
        ));
    }
    source.seek(SeekFrom::Start(0))?;
    snapshot.flush()?;
    if unsafe { libc::fchmod(fd, 0o500) } != 0 {
        return Err(io::Error::last_os_error());
    }
    let seals = libc::F_SEAL_SEAL | libc::F_SEAL_SHRINK | libc::F_SEAL_GROW | libc::F_SEAL_WRITE;
    if unsafe { libc::fcntl(fd, libc::F_ADD_SEALS, seals) } < 0 {
        return Err(io::Error::last_os_error());
    }
    snapshot.seek(SeekFrom::Start(0))?;
    Ok(snapshot)
}

fn validate_python_interpreter(python: &OsStr) -> Result<ValidatedPythonInterpreter, LaunchError> {
    let requested = Path::new(python);
    if !requested.is_absolute() {
        return Err(LaunchError::InvalidPythonInterpreter(
            "--python must name an absolute interpreter path".to_owned(),
        ));
    }
    let canonical = requested.canonicalize().map_err(|error| {
        LaunchError::InvalidPythonInterpreter(format!(
            "canonicalize Python interpreter {}: {error}",
            requested.display()
        ))
    })?;
    let mut executable = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(&canonical)
        .map_err(|error| {
            LaunchError::InvalidPythonInterpreter(format!(
                "open Python interpreter {}: {error}",
                canonical.display()
            ))
        })?;
    let metadata = executable.metadata().map_err(|error| {
        LaunchError::InvalidPythonInterpreter(format!(
            "inspect Python interpreter {}: {error}",
            canonical.display()
        ))
    })?;
    if !metadata.is_file() {
        return Err(LaunchError::InvalidPythonInterpreter(format!(
            "Python interpreter is not a regular file: {}",
            canonical.display()
        )));
    }
    if metadata.permissions().mode() & 0o111 == 0 {
        return Err(LaunchError::InvalidPythonInterpreter(format!(
            "Python interpreter is not executable: {}",
            canonical.display()
        )));
    }
    if metadata.len() == 0 || metadata.len() > MAX_EXECUTABLE_BYTES {
        return Err(LaunchError::InvalidPythonInterpreter(format!(
            "Python interpreter exceeds executable size limit: {}",
            canonical.display()
        )));
    }
    let target_device = metadata.dev();
    let target_inode = metadata.ino();
    let target_sha256 = file_sha256_from(&mut executable).map_err(|error| {
        LaunchError::InvalidPythonInterpreter(format!(
            "hash Python interpreter {}: {error}",
            canonical.display()
        ))
    })?;
    let sealed_image =
        sealed_executable_snapshot(&mut executable, "legacy-python").map_err(|error| {
            LaunchError::InvalidPythonInterpreter(format!(
                "seal Python interpreter {}: {error}",
                canonical.display()
            ))
        })?;
    let environment = validate_python_environment(requested)?;
    let validated = ValidatedPythonInterpreter {
        invocation: requested.to_owned(),
        target: canonical.clone(),
        target_device,
        target_inode,
        target_sha256,
        sealed_image,
        environment,
    };
    validated.revalidate()?;
    // Execute immutable bytes while preserving the operator-supplied argv[0],
    // which CPython uses to discover pyvenv.cfg and venv site-packages.
    let mut command = validated.direct_command();
    command
        .env_clear()
        .args(["-I", "-B", "-c", PYTHON_HANDSHAKE])
        .stdin(Stdio::null());
    let (status, stdout, stderr) =
        bounded_command_output(command, 4096, HANDSHAKE_TIMEOUT, "Python handshake").map_err(
            |error| {
                map_launch_io_error(error, |error| {
                    LaunchError::InvalidPythonInterpreter(format!(
                        "run Python handshake {}: {error}",
                        requested.display()
                    ))
                })
            },
        )?;
    let stdout = String::from_utf8(stdout).map_err(|error| {
        LaunchError::InvalidPythonInterpreter(format!(
            "Python handshake output is not UTF-8: {error}"
        ))
    })?;
    let mut lines = stdout.lines();
    let sentinel = lines.next();
    let executable = lines.next();
    let version = lines.next();
    let canonical_hex = hex_digest(canonical.as_os_str().as_bytes());
    if !status.success()
        || sentinel != Some("LAMQUANT_LEGACY_PYTHON_V1")
        || executable != Some(canonical_hex.as_str())
        || lines.next().is_some()
    {
        return Err(LaunchError::InvalidPythonInterpreter(format!(
            "Python handshake failed for {}: {}",
            requested.display(),
            String::from_utf8_lossy(&stderr).trim()
        )));
    }
    let (major, minor) = version
        .and_then(|value| value.split_once('.'))
        .and_then(|(major, minor)| Some((major.parse::<u8>().ok()?, minor.parse::<u8>().ok()?)))
        .ok_or_else(|| {
            LaunchError::InvalidPythonInterpreter(
                "Python handshake returned an invalid version".to_owned(),
            )
        })?;
    if (major, minor) < (3, 10) {
        return Err(LaunchError::InvalidPythonInterpreter(format!(
            "Python 3.10 or newer required, got {major}.{minor}"
        )));
    }
    validated.revalidate()?;
    Ok(validated)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ProjectedFile {
    bytes: u64,
    sha256: String,
}

struct SourceProjection {
    archive: Vec<u8>,
    files: BTreeMap<PathBuf, ProjectedFile>,
    manifest: Vec<u8>,
    trainer: LegacyTrainer,
}

impl SourceProjection {
    fn from_verified(verified: &VerifiedCheckout) -> Result<Self, io::Error> {
        let archive = source_archive(verified)?;
        let files = inspect_source_archive(&archive)?;
        if !files.contains_key(Path::new(verified.trainer().script())) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "frozen trainer is absent from source projection",
            ));
        }
        let archive_sha256 = hex_digest(Sha256::digest(&archive));
        let mut manifest = String::new();
        manifest.push_str("LAMQUANT_LEGACY_SOURCE_PROJECTION_V1\n");
        manifest.push_str(&format!("repository\t{SOURCE_REPOSITORY}\n"));
        manifest.push_str(&format!("revision\t{}\n", verified.revision()));
        manifest.push_str(&format!("trainer\t{}\n", verified.trainer()));
        manifest.push_str(&format!("archive_sha256\t{archive_sha256}\n"));
        manifest.push_str("projection\tpython/lamquant source files; tracked bytecode excluded\n");
        manifest.push_str(&format!("file_count\t{}\n", files.len()));
        for (path, file) in &files {
            let path = path.to_str().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "frozen source path is not UTF-8",
                )
            })?;
            if path.contains(['\n', '\r', '\t']) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "frozen source path contains a manifest delimiter",
                ));
            }
            manifest.push_str(&format!(
                "file\t{}\t{}\t{}\n",
                file.sha256, file.bytes, path
            ));
        }
        Ok(Self {
            archive,
            files,
            manifest: manifest.into_bytes(),
            trainer: verified.trainer(),
        })
    }
}

struct WorkspaceLock {
    file: fs::File,
    path: PathBuf,
    device: u64,
    inode: u64,
}

impl WorkspaceLock {
    fn acquire(root: &Path) -> Result<Self, io::Error> {
        let parent = root
            .parent()
            .ok_or_else(|| io::Error::other("legacy workspace has no parent"))?;
        let file_name = root
            .file_name()
            .ok_or_else(|| io::Error::other("legacy workspace has no file name"))?;
        let mut lock_name = OsString::from(".");
        lock_name.push(file_name);
        lock_name.push(".lock");
        let lock_path = parent.join(lock_name);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .mode(0o600)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(&lock_path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file()
            || metadata.permissions().mode() & 0o077 != 0
            || metadata.nlink() != 1
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!(
                    "legacy workspace lock must be a private regular file: {}",
                    lock_path.display()
                ),
            ));
        }
        // SAFETY: descriptor remains owned by `WorkspaceLock` until trainer exit.
        let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if result != 0 {
            let error = io::Error::last_os_error();
            return Err(
                if error
                    .raw_os_error()
                    .is_some_and(|code| code == libc::EWOULDBLOCK || code == libc::EAGAIN)
                {
                    io::Error::new(
                        io::ErrorKind::WouldBlock,
                        format!(
                            "legacy workspace is already owned by another launcher: {}",
                            root.display()
                        ),
                    )
                } else {
                    error
                },
            );
        }
        let lock = Self {
            file,
            path: lock_path,
            device: metadata.dev(),
            inode: metadata.ino(),
        };
        lock.validate_identity()?;
        Ok(lock)
    }

    fn validate_identity(&self) -> Result<(), io::Error> {
        let path_metadata = self.path.symlink_metadata()?;
        let file_metadata = self.file.metadata()?;
        if !path_metadata.is_file()
            || path_metadata.file_type().is_symlink()
            || path_metadata.permissions().mode() & 0o077 != 0
            || path_metadata.nlink() != 1
            || path_metadata.dev() != self.device
            || path_metadata.ino() != self.inode
            || file_metadata.dev() != self.device
            || file_metadata.ino() != self.inode
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!(
                    "legacy workspace lock identity changed: {}",
                    self.path.display()
                ),
            ));
        }
        Ok(())
    }
}

impl Drop for WorkspaceLock {
    fn drop(&mut self) {
        // SAFETY: unlock applies to this process-owned descriptor.
        unsafe {
            libc::flock(self.file.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

struct FrozenWorkspace {
    root: PathBuf,
    revision: String,
    trainer: LegacyTrainer,
    _lock: WorkspaceLock,
}

impl FrozenWorkspace {
    fn open_or_create(verified: &VerifiedCheckout, requested: &Path) -> Result<Self, io::Error> {
        let root = validated_workspace_path(verified.root(), requested)?;
        let lock = WorkspaceLock::acquire(&root)?;
        let locked_root = validated_workspace_path(verified.root(), requested)?;
        if root != locked_root {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "legacy workspace identity changed while acquiring its lock",
            ));
        }
        let projection = SourceProjection::from_verified(verified)?;
        if root.exists() {
            validate_existing_workspace(&root, &projection)?;
        } else {
            create_workspace(&root, &projection)?;
        }
        for directory in verified.trainer().writable_directories(&root) {
            ensure_private_directory(&root, &directory)?;
        }
        let workspace = Self {
            root,
            revision: verified.revision().to_owned(),
            trainer: verified.trainer(),
            _lock: lock,
        };
        workspace._lock.validate_identity()?;
        validate_managed_destinations(&workspace)?;
        Ok(workspace)
    }

    fn root(&self) -> &Path {
        &self.root
    }

    fn script(&self) -> PathBuf {
        self.root().join("source").join(self.trainer.script())
    }

    fn virtual_script(&self) -> PathBuf {
        self.root().join("run").join(self.trainer.script())
    }

    fn execution_root(&self) -> PathBuf {
        self.root().join("run")
    }

    fn manifest(&self) -> PathBuf {
        self.root().join(SOURCE_MANIFEST)
    }

    fn revalidate(&self, verified: &VerifiedCheckout) -> Result<(), io::Error> {
        self._lock.validate_identity()?;
        let projection = SourceProjection::from_verified(verified)?;
        validate_existing_workspace(self.root(), &projection)
    }
}

fn ensure_private_directory(root: &Path, directory: &Path) -> Result<(), io::Error> {
    let relative = directory.strip_prefix(root).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "legacy writable directory escaped workspace: {}",
                directory.display()
            ),
        )
    })?;
    let mut current = root.to_owned();
    for component in relative.components() {
        let std::path::Component::Normal(component) = component else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "legacy writable directory is not normalized",
            ));
        };
        current.push(component);
        match current.symlink_metadata() {
            Ok(metadata) => {
                if !metadata.is_dir() || metadata.file_type().is_symlink() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "legacy writable directory is not a real directory: {}",
                            current.display()
                        ),
                    ));
                }
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                fs::create_dir(&current)?;
            }
            Err(error) => return Err(error),
        }
        fs::set_permissions(&current, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

fn source_archive(verified: &VerifiedCheckout) -> Result<Vec<u8>, io::Error> {
    let mut command = git_command(verified.git(), verified.root());
    command.args([
        "archive",
        "--format=tar",
        verified.revision(),
        "--",
        "python/lamquant",
    ]);
    let (status, archive, stderr) = supervised_command_output(
        command,
        MAX_ARCHIVE_BYTES as usize,
        64 * 1024,
        GIT_COMMAND_TIMEOUT,
        None,
        "Git source archive",
    )?;
    if !status.success() {
        return Err(io::Error::other(format!(
            "Git archive failed ({status}): {}",
            String::from_utf8_lossy(&stderr).trim()
        )));
    }
    Ok(archive)
}

fn inspect_source_archive(archive: &[u8]) -> Result<BTreeMap<PathBuf, ProjectedFile>, io::Error> {
    let mut files = BTreeMap::new();
    let mut archive = tar::Archive::new(archive);
    for entry in archive.entries()? {
        let mut entry = entry?;
        let entry_type = entry.header().entry_type();
        if archive_metadata_entry(entry_type) {
            continue;
        }
        if !entry_type.is_file() && !entry_type.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "frozen Python source contains unsupported archive entry",
            ));
        }
        let path = validated_projection_path(&entry.path()?)?;
        if excluded_projection_path(&path) || entry_type.is_dir() {
            continue;
        }
        let bytes = entry.header().size()?;
        let mut hasher = Sha256::new();
        let mut observed = 0_u64;
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let count = entry.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            observed = observed
                .checked_add(count as u64)
                .ok_or_else(|| io::Error::other("source entry size overflow"))?;
            hasher.update(&buffer[..count]);
        }
        if observed != bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("source entry size mismatch: {}", path.display()),
            ));
        }
        if files
            .insert(
                path.clone(),
                ProjectedFile {
                    bytes,
                    sha256: hex_digest(hasher.finalize()),
                },
            )
            .is_some()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("duplicate source entry: {}", path.display()),
            ));
        }
    }
    if files.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "frozen source projection contains no files",
        ));
    }
    Ok(files)
}

fn archive_metadata_entry(entry_type: tar::EntryType) -> bool {
    entry_type.is_pax_global_extensions()
        || entry_type.is_pax_local_extensions()
        || entry_type.is_gnu_longname()
        || entry_type.is_gnu_longlink()
}

fn validated_projection_path(path: &Path) -> Result<PathBuf, io::Error> {
    if path
        .components()
        .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("frozen source path is not normalized: {}", path.display()),
        ));
    }
    if path != Path::new("python") && !path.starts_with("python/lamquant") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("frozen source escaped allowlisted tree: {}", path.display()),
        ));
    }
    Ok(path.to_owned())
}

fn excluded_projection_path(path: &Path) -> bool {
    path.components()
        .any(|component| component.as_os_str() == OsStr::new("__pycache__"))
        || matches!(
            path.extension().and_then(OsStr::to_str),
            Some("pyc" | "pyo")
        )
}

fn validated_workspace_path(checkout: &Path, requested: &Path) -> Result<PathBuf, io::Error> {
    if !requested.is_absolute() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--workspace must name an absolute path",
        ));
    }
    let requested_parent = requested.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "--workspace must have an existing parent",
        )
    })?;
    let parent = requested_parent.canonicalize()?;
    validate_private_workspace_parent(&parent)?;
    let root = if requested.exists() {
        if requested.symlink_metadata()?.file_type().is_symlink() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "legacy workspace path must not be a symlink",
            ));
        }
        requested.canonicalize()?
    } else {
        let name = requested.file_name().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "--workspace must have a final path component",
            )
        })?;
        parent.join(name)
    };
    if root.starts_with(checkout) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "legacy workspace must be outside the verified checkout",
        ));
    }
    if root.exists() {
        let metadata = root.symlink_metadata()?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "legacy workspace must be a real directory",
            ));
        }
        if metadata.permissions().mode() & 0o077 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "legacy workspace must not grant group or other permissions",
            ));
        }
    }
    Ok(root)
}

fn validate_private_workspace_parent(parent: &Path) -> Result<(), io::Error> {
    let metadata = parent.symlink_metadata()?;
    if !metadata.is_dir()
        || metadata.file_type().is_symlink()
        || metadata.uid() != unsafe { libc::geteuid() }
        || metadata.permissions().mode() & 0o022 != 0
    {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!(
                "legacy workspace parent must be owned by effective UID and not group/other writable: {}",
                parent.display()
            ),
        ));
    }
    Ok(())
}

fn create_workspace(root: &Path, projection: &SourceProjection) -> Result<(), io::Error> {
    let parent = root
        .parent()
        .ok_or_else(|| io::Error::other("legacy workspace has no parent"))?;
    let file_name = root
        .file_name()
        .ok_or_else(|| io::Error::other("legacy workspace has no file name"))?;
    let mut staging_name = OsString::from(".");
    staging_name.push(file_name);
    staging_name.push(".staging");
    let staging = parent.join(staging_name);
    let binding = workspace_binding(root, projection);
    if staging.exists() {
        let metadata = staging.symlink_metadata()?;
        if !metadata.is_dir()
            || metadata.file_type().is_symlink()
            || metadata.permissions().mode() & 0o077 != 0
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "abandoned legacy staging path is not a private directory: {}",
                    staging.display()
                ),
            ));
        }
        let observed = read_private_regular_file(
            &staging.join(WORKSPACE_BINDING),
            4096,
            "legacy staging binding",
        )?;
        if observed != binding {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!(
                    "abandoned legacy staging directory is not bound to requested workspace: {}",
                    staging.display()
                ),
            ));
        }
        remove_private_tree(&staging)?;
    }
    fs::create_dir(&staging)?;
    fs::set_permissions(&staging, fs::Permissions::from_mode(0o700))?;
    let result = (|| {
        let binding_path = staging.join(WORKSPACE_BINDING);
        let mut binding_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(&binding_path)?;
        binding_file.write_all(&binding)?;
        binding_file.sync_all()?;
        fs::set_permissions(&binding_path, fs::Permissions::from_mode(0o400))?;
        let source_root = staging.join("source");
        fs::create_dir(&source_root)?;
        fs::set_permissions(&source_root, fs::Permissions::from_mode(0o700))?;
        let mut archive = tar::Archive::new(projection.archive.as_slice());
        for entry in archive.entries()? {
            let mut entry = entry?;
            let entry_type = entry.header().entry_type();
            if archive_metadata_entry(entry_type) {
                continue;
            }
            let path = validated_projection_path(&entry.path()?)?;
            if excluded_projection_path(&path) {
                continue;
            }
            let destination = source_root.join(&path);
            if entry_type.is_dir() {
                fs::create_dir_all(&destination)?;
                fs::set_permissions(&destination, fs::Permissions::from_mode(0o700))?;
                continue;
            }
            let parent = destination
                .parent()
                .ok_or_else(|| io::Error::other("source file has no parent"))?;
            fs::create_dir_all(parent)?;
            fs::set_permissions(parent, fs::Permissions::from_mode(0o700))?;
            let mut output = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&destination)?;
            io::copy(&mut entry, &mut output)?;
            output.sync_all()?;
            fs::set_permissions(&destination, fs::Permissions::from_mode(0o400))?;
        }
        set_source_tree_read_only(&source_root)?;
        let virtual_script = staging.join("run").join(projection.trainer.script());
        let virtual_parent = virtual_script
            .parent()
            .ok_or_else(|| io::Error::other("virtual trainer has no parent"))?;
        fs::create_dir_all(virtual_parent)?;
        fs::set_permissions(virtual_parent, fs::Permissions::from_mode(0o700))?;
        fs::copy(
            source_root.join(projection.trainer.script()),
            &virtual_script,
        )?;
        fs::set_permissions(&virtual_script, fs::Permissions::from_mode(0o400))?;
        fs::File::open(&virtual_script)?.sync_all()?;
        for resource in projection.trainer.read_resources() {
            let resource = Path::new(resource);
            if !projection.files.contains_key(resource) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "frozen trainer resource is absent from source projection: {}",
                        resource.display()
                    ),
                ));
            }
            let virtual_resource = staging.join("run").join(resource);
            let virtual_parent = virtual_resource
                .parent()
                .ok_or_else(|| io::Error::other("virtual resource has no parent"))?;
            fs::create_dir_all(virtual_parent)?;
            fs::set_permissions(virtual_parent, fs::Permissions::from_mode(0o700))?;
            fs::copy(source_root.join(resource), &virtual_resource)?;
            fs::set_permissions(&virtual_resource, fs::Permissions::from_mode(0o400))?;
            fs::File::open(virtual_resource)?.sync_all()?;
        }
        let manifest = staging.join(SOURCE_MANIFEST);
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&manifest)?;
        output.write_all(&projection.manifest)?;
        output.sync_all()?;
        fs::set_permissions(manifest, fs::Permissions::from_mode(0o400))?;
        validate_existing_workspace_at(&staging, root, projection)?;
        let mut entries = 0;
        sync_directory_tree(&staging, 0, &mut entries)?;
        fs::rename(&staging, root)?;
        fs::File::open(parent)?.sync_all()
    })();
    if result.is_err() {
        let _ = remove_private_tree(&staging);
    }
    result
}

fn workspace_binding(root: &Path, projection: &SourceProjection) -> Vec<u8> {
    format!(
        "LAMQUANT_LEGACY_WORKSPACE_BINDING_V1\nroot_sha256\t{}\nsource_manifest_sha256\t{}\ntrainer\t{}\n",
        hex_digest(Sha256::digest(root.as_os_str().as_bytes())),
        hex_digest(Sha256::digest(&projection.manifest)),
        projection.trainer,
    )
    .into_bytes()
}

fn remove_private_tree(path: &Path) -> Result<(), io::Error> {
    let metadata = path.symlink_metadata()?;
    if metadata.file_type().is_symlink() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "refusing to remove symlink staging path: {}",
                path.display()
            ),
        ));
    }
    if metadata.is_dir() {
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
        for entry in fs::read_dir(path)? {
            remove_private_tree(&entry?.path())?;
        }
        fs::remove_dir(path)
    } else if metadata.is_file() {
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
        fs::remove_file(path)
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "refusing to remove unsupported staging entry: {}",
                path.display()
            ),
        ))
    }
}

fn sync_directory_tree(path: &Path, depth: usize, entries: &mut usize) -> Result<(), io::Error> {
    if depth > MAX_WORKSPACE_VALIDATION_DEPTH {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "legacy staging tree exceeds maximum depth",
        ));
    }
    for entry in fs::read_dir(path)? {
        *entries = entries
            .checked_add(1)
            .ok_or_else(|| io::Error::other("legacy staging entry count overflow"))?;
        if *entries > MAX_WORKSPACE_VALIDATION_ENTRIES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "legacy staging tree exceeds maximum entry count",
            ));
        }
        let path = entry?.path();
        let metadata = path.symlink_metadata()?;
        if metadata.is_dir() {
            sync_directory_tree(&path, depth + 1, entries)?;
        }
    }
    fs::File::open(path)?.sync_all()
}

fn set_source_tree_read_only(path: &Path) -> Result<(), io::Error> {
    let metadata = path.symlink_metadata()?;
    if metadata.file_type().is_symlink() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("source projection contains symlink: {}", path.display()),
        ));
    }
    if metadata.is_dir() {
        for entry in fs::read_dir(path)? {
            set_source_tree_read_only(&entry?.path())?;
        }
        fs::set_permissions(path, fs::Permissions::from_mode(0o500))
    } else if metadata.is_file() {
        fs::set_permissions(path, fs::Permissions::from_mode(0o400))
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "source projection contains unsupported entry: {}",
                path.display()
            ),
        ))
    }
}

fn validate_existing_workspace(
    root: &Path,
    projection: &SourceProjection,
) -> Result<(), io::Error> {
    validate_existing_workspace_at(root, root, projection)
}

fn validate_existing_workspace_at(
    root: &Path,
    bound_root: &Path,
    projection: &SourceProjection,
) -> Result<(), io::Error> {
    let binding = read_private_regular_file(
        &root.join(WORKSPACE_BINDING),
        4096,
        "legacy workspace binding",
    )?;
    if binding != workspace_binding(bound_root, projection) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "legacy workspace binding does not match requested root, revision, and trainer",
        ));
    }
    let manifest = read_private_regular_file(
        &root.join(SOURCE_MANIFEST),
        MAX_GIT_TEXT_BYTES,
        "source projection manifest",
    )?;
    if manifest != projection.manifest {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "legacy workspace source manifest does not match requested revision and trainer",
        ));
    }
    for (path, expected) in &projection.files {
        let path = root.join("source").join(path);
        let metadata = path.symlink_metadata()?;
        if !metadata.is_file()
            || metadata.file_type().is_symlink()
            || metadata.permissions().mode() & 0o222 != 0
            || metadata.len() != expected.bytes
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "legacy workspace source metadata changed: {}",
                    path.display()
                ),
            ));
        }
        let actual = file_sha256(&path)?;
        if actual != expected.sha256 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "legacy workspace source content changed: {}",
                    path.display()
                ),
            ));
        }
    }
    let virtual_script = root.join("run").join(projection.trainer.script());
    let source_script = root.join("source").join(projection.trainer.script());
    let virtual_metadata = virtual_script.symlink_metadata()?;
    if !virtual_metadata.is_file()
        || virtual_metadata.file_type().is_symlink()
        || virtual_metadata.permissions().mode() & 0o222 != 0
        || file_sha256(&virtual_script)? != file_sha256(&source_script)?
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "legacy virtual trainer differs from frozen source",
        ));
    }
    for resource in projection.trainer.read_resources() {
        let source = root.join("source").join(resource);
        let virtual_resource = root.join("run").join(resource);
        let metadata = virtual_resource.symlink_metadata()?;
        if !metadata.is_file()
            || metadata.file_type().is_symlink()
            || metadata.permissions().mode() & 0o222 != 0
            || file_sha256(&virtual_resource)? != file_sha256(&source)?
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "legacy virtual read resource differs from frozen source: {}",
                    virtual_resource.display()
                ),
            ));
        }
    }
    let mut entries = 0;
    validate_workspace_entries(
        root,
        &root.join("source"),
        &projection.files,
        projection.trainer,
        true,
        0,
        &mut entries,
    )?;
    validate_workspace_entries(
        root,
        &root.join("run"),
        &projection.files,
        projection.trainer,
        false,
        0,
        &mut entries,
    )?;
    Ok(())
}

fn validate_workspace_entries(
    root: &Path,
    path: &Path,
    projected_files: &BTreeMap<PathBuf, ProjectedFile>,
    trainer: LegacyTrainer,
    immutable_source: bool,
    depth: usize,
    entries: &mut usize,
) -> Result<(), io::Error> {
    if depth > MAX_WORKSPACE_VALIDATION_DEPTH {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "legacy workspace exceeds maximum validation depth",
        ));
    }
    let root_metadata = path.symlink_metadata()?;
    if !root_metadata.is_dir() || root_metadata.file_type().is_symlink() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "legacy workspace validation root is not a real directory: {}",
                path.display()
            ),
        ));
    }
    if immutable_source && root_metadata.permissions().mode() & 0o222 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "legacy source projection directory became writable: {}",
                path.display()
            ),
        ));
    }
    for entry in fs::read_dir(path)? {
        *entries = entries
            .checked_add(1)
            .ok_or_else(|| io::Error::other("legacy workspace entry count overflow"))?;
        if *entries > MAX_WORKSPACE_VALIDATION_ENTRIES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "legacy workspace exceeds maximum validation entry count",
            ));
        }
        let path = entry?.path();
        let metadata = path.symlink_metadata()?;
        if metadata.file_type().is_symlink() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("legacy workspace contains symlink: {}", path.display()),
            ));
        }
        if metadata.is_dir() {
            if immutable_source && metadata.permissions().mode() & 0o222 != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "legacy source projection directory became writable: {}",
                        path.display()
                    ),
                ));
            }
            if path.file_name() == Some(OsStr::new("__pycache__")) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "legacy workspace contains forbidden import cache: {}",
                        path.display()
                    ),
                ));
            }
            validate_workspace_entries(
                root,
                &path,
                projected_files,
                trainer,
                immutable_source,
                depth + 1,
                entries,
            )?;
            continue;
        }
        if !metadata.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "legacy workspace contains unsupported entry: {}",
                    path.display()
                ),
            ));
        }
        let relative = path
            .strip_prefix(root)
            .map_err(|_| io::Error::other("workspace traversal escaped root"))?;
        if immutable_source
            && relative
                .strip_prefix("source")
                .is_ok_and(|source| projected_files.contains_key(source))
        {
            continue;
        }
        let virtual_script = Path::new("run").join(trainer.script());
        if !immutable_source && relative == virtual_script {
            continue;
        }
        if immutable_source
            || relative
                .extension()
                .and_then(OsStr::to_str)
                .is_some_and(|extension| IMPORTABLE_EXTENSIONS.contains(&extension))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "legacy workspace contains unbound source or importable file: {}",
                    path.display()
                ),
            ));
        }
    }
    Ok(())
}

fn file_sha256(path: &Path) -> Result<String, io::Error> {
    let mut input = fs::File::open(path)?;
    file_sha256_from(&mut input)
}

fn file_sha256_from(input: &mut fs::File) -> Result<String, io::Error> {
    input.seek(SeekFrom::Start(0))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    input.seek(SeekFrom::Start(0))?;
    Ok(hex_digest(hasher.finalize()))
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    let bytes = bytes.as_ref();
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use core::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn dependency_preflight(
    workspace: &FrozenWorkspace,
    sandbox: &ValidatedSandboxExecutable,
    python: &ValidatedPythonInterpreter,
    environment: &[LegacyEnvironment],
    modules: Vec<&str>,
) -> Result<Vec<u8>, LaunchError> {
    sandbox.revalidate()?;
    python.revalidate()?;
    let mut command = isolated_python_command(workspace, sandbox, python, environment);
    command
        .args(["-I", "-B", "-c", DEPENDENCY_PREFLIGHT])
        .arg(workspace.root().join("source/python"))
        .arg(workspace.script())
        .arg(workspace.virtual_script())
        .args(modules);
    let (status, stdout, stderr) = bounded_command_output(
        command,
        MAX_PREFLIGHT_OUTPUT_BYTES,
        PREFLIGHT_TIMEOUT,
        "dependency preflight",
    )
    .map_err(|error| {
        map_launch_io_error(error, |error| {
            LaunchError::DependencyPreflight(error.to_string())
        })
    })?;
    if !status.success()
        || !String::from_utf8_lossy(&stdout)
            .lines()
            .any(|line| line == "LAMQUANT_LEGACY_PREFLIGHT_V1")
    {
        return Err(LaunchError::DependencyPreflight(format!(
            "Python exited {status}: {}",
            String::from_utf8_lossy(&stderr).trim()
        )));
    }
    Ok(stdout)
}

fn preflight_dependency_environment(
    workspace: &FrozenWorkspace,
    verified: &VerifiedCheckout,
    sandbox: &ValidatedSandboxExecutable,
    python: &ValidatedPythonInterpreter,
    environment: &[LegacyEnvironment],
    modules: Vec<&str>,
) -> Result<(), LaunchError> {
    let report = dependency_preflight(workspace, sandbox, python, environment, modules)?;
    workspace
        .revalidate(verified)
        .map_err(map_workspace_error)?;
    validate_managed_destinations(workspace).map_err(LaunchError::Workspace)?;
    publish_dependency_report(workspace, &report)?;
    workspace.revalidate(verified).map_err(map_workspace_error)
}

fn publish_dependency_report(workspace: &FrozenWorkspace, bytes: &[u8]) -> Result<(), LaunchError> {
    let report = workspace.root().join("artifacts").join(DEPENDENCY_REPORT);
    atomic_private_write(&report, bytes).map_err(|error| {
        LaunchError::DependencyPreflight(format!(
            "publish dependency report {}: {error}",
            report.display()
        ))
    })
}

fn atomic_private_write(path: &Path, bytes: &[u8]) -> Result<(), io::Error> {
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::other("atomic output has no parent"))?;
    let file_name = path
        .file_name()
        .ok_or_else(|| io::Error::other("atomic output has no file name"))?;
    let parent_handle = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_DIRECTORY | libc::O_NOFOLLOW)
        .open(parent)?;
    let parent_fd = parent_handle.as_raw_fd();
    let final_name = path_component_cstring(file_name)?;

    let mut temporary_name = OsString::from(".");
    temporary_name.push(file_name);
    temporary_name.push(format!(
        ".{}.{}.tmp",
        std::process::id(),
        ATOMIC_WRITE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    let temporary_name = path_component_cstring(&temporary_name)?;
    // SAFETY: both names are NUL-free single path components. `parent_fd`
    // remains open for this function, so creation, rename, cleanup, and fsync
    // stay bound to one verified directory inode even if its pathname changes.
    let descriptor = unsafe {
        libc::openat(
            parent_fd,
            temporary_name.as_ptr(),
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            0o600,
        )
    };
    if descriptor < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `openat` returned a newly owned descriptor.
    let mut output = unsafe { fs::File::from_raw_fd(descriptor) };
    let result = (|| {
        output.write_all(bytes)?;
        output.sync_all()?;
        // Close the staging file before publishing its directory entry.
        drop(output);
        // SAFETY: source and destination are NUL-free names in the same held
        // directory. `renameat` replaces the destination entry, never follows it.
        if unsafe {
            libc::renameat(
                parent_fd,
                temporary_name.as_ptr(),
                parent_fd,
                final_name.as_ptr(),
            )
        } != 0
        {
            return Err(io::Error::last_os_error());
        }
        parent_handle.sync_all()
    })();
    if result.is_err() {
        // SAFETY: cleanup is restricted to the staging name in the held parent.
        unsafe {
            libc::unlinkat(parent_fd, temporary_name.as_ptr(), 0);
        }
    }
    result
}

fn path_component_cstring(value: &OsStr) -> Result<CString, io::Error> {
    if value.as_bytes().contains(&b'/') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "atomic output name must be one path component",
        ));
    }
    CString::new(value.as_bytes()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "atomic output name contains NUL",
        )
    })
}

fn bounded_command_output(
    command: Command,
    limit: usize,
    timeout: Duration,
    label: &'static str,
) -> Result<(ExitStatus, Vec<u8>, Vec<u8>), io::Error> {
    supervised_command_output(command, limit, limit, timeout, None, label)
}

fn supervised_command_output(
    mut command: Command,
    stdout_limit: usize,
    stderr_limit: usize,
    timeout: Duration,
    stdin: Option<Vec<u8>>,
    label: &'static str,
) -> Result<(ExitStatus, Vec<u8>, Vec<u8>), io::Error> {
    command
        .stdin(if stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = SupervisedChild::spawn(command)?;
    let mut stdin_pipe = if let Some(input) = stdin {
        let pipe = child
            .child
            .stdin
            .take()
            .ok_or_else(|| io::Error::other(format!("open {label} stdin")))?;
        set_nonblocking(pipe.as_raw_fd())?;
        Some((pipe, input, 0_usize))
    } else {
        None
    };
    let stdout = child
        .child
        .stdout
        .take()
        .ok_or_else(|| io::Error::other(format!("open {label} stdout")))?;
    set_nonblocking(stdout.as_raw_fd())?;
    let stderr = child
        .child
        .stderr
        .take()
        .ok_or_else(|| io::Error::other(format!("open {label} stderr")))?;
    set_nonblocking(stderr.as_raw_fd())?;
    let mut stdout_pipe = Some(stdout);
    let mut stderr_pipe = Some(stderr);
    let mut stdout = Vec::with_capacity(stdout_limit.min(64 * 1024));
    let mut stderr = Vec::with_capacity(stderr_limit.min(64 * 1024));
    let deadline = Instant::now().checked_add(timeout).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "subprocess timeout overflow")
    })?;
    let mut status = None;
    let mut forwarded_signal = None;
    let mut termination_deadline = None;
    loop {
        if forwarded_signal.is_none() {
            let signal = PENDING_TERMINATION_SIGNAL.swap(0, Ordering::AcqRel);
            if signal != 0 {
                child.forward_signal(signal);
                forwarded_signal = Some(signal);
                termination_deadline = Instant::now().checked_add(TERMINATION_GRACE);
            }
        }
        drain_nonblocking(&mut stdout_pipe, &mut stdout, stdout_limit, label)?;
        drain_nonblocking(&mut stderr_pipe, &mut stderr, stderr_limit, label)?;
        write_nonblocking(&mut stdin_pipe)?;
        if status.is_none() {
            status = child.try_wait()?;
            if status.is_some() {
                child.finish_group();
            }
        }
        if let Some(status) = status {
            if stdout_pipe.is_none() && stderr_pipe.is_none() && stdin_pipe.is_none() {
                let status = child.complete(status);
                if let Some(signal) = forwarded_signal {
                    return Err(interrupted_io_error(signal, label));
                }
                return Ok((status, stdout, stderr));
            }
        }
        let now = Instant::now();
        if termination_deadline.is_some_and(|deadline| now >= deadline) {
            child.terminate_and_wait();
            let signal = forwarded_signal.expect("termination deadline requires signal");
            return Err(interrupted_io_error(signal, label));
        }
        if now >= deadline {
            child.terminate_and_wait();
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!("{label} exceeded {} seconds", timeout.as_secs_f64()),
            ));
        }
        std::thread::sleep(POLL_INTERVAL.min(deadline.saturating_duration_since(now)));
    }
}

fn set_nonblocking(fd: libc::c_int) -> Result<(), io::Error> {
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 {
        return Err(io::Error::last_os_error());
    }
    if unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

fn drain_nonblocking<R: Read>(
    pipe: &mut Option<R>,
    output: &mut Vec<u8>,
    limit: usize,
    label: &'static str,
) -> Result<(), io::Error> {
    let Some(reader) = pipe.as_mut() else {
        return Ok(());
    };
    let mut buffer = [0_u8; 8192];
    loop {
        match reader.read(&mut buffer) {
            Ok(0) => {
                *pipe = None;
                return Ok(());
            }
            Ok(count) => {
                if output.len().saturating_add(count) > limit {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("{label} exceeds {limit} bytes"),
                    ));
                }
                output.extend_from_slice(&buffer[..count]);
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(()),
            Err(error) => return Err(error),
        }
    }
}

fn write_nonblocking(
    input: &mut Option<(std::process::ChildStdin, Vec<u8>, usize)>,
) -> Result<(), io::Error> {
    let Some((writer, bytes, offset)) = input.as_mut() else {
        return Ok(());
    };
    while *offset < bytes.len() {
        match writer.write(&bytes[*offset..]) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "write child stdin",
                ))
            }
            Ok(count) => *offset += count,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(()),
            Err(error) => return Err(error),
        }
    }
    *input = None;
    Ok(())
}

fn isolated_python_command(
    workspace: &FrozenWorkspace,
    sandbox: &ValidatedSandboxExecutable,
    python: &ValidatedPythonInterpreter,
    environment: &[LegacyEnvironment],
) -> Command {
    let mut command = sandbox.command();
    let python_fd = python.sealed_fd();
    inherit_fd_across_exec(&mut command, python_fd);
    command
        .env_clear()
        .args(["--ro-bind", "/", "/"])
        // Frozen GPU trainers need dynamically enumerated accelerator nodes.
        // Full host `/dev` exposure is an explicit same-device trust boundary,
        // documented in README; filesystem and PID isolation remain separate.
        .args(["--dev-bind", "/dev", "/dev"])
        .args(["--proc", "/proc"])
        .args(["--tmpfs", "/tmp"])
        .args(["--tmpfs", "/dev/shm"]);
    if let Some(environment) = python.environment() {
        let root_fd = environment.root_fd();
        inherit_fd_across_exec(&mut command, root_fd);
        command
            .arg("--ro-bind-fd")
            .arg(root_fd.to_string())
            .arg(environment.invocation_root());
    }
    command
        .args(["--perms", "0500"])
        .arg("--ro-bind-data")
        .arg(python_fd.to_string())
        .arg(PINNED_PYTHON_EXECUTABLE)
        .arg("--bind")
        .arg(workspace.root())
        .arg(workspace.root())
        .arg("--ro-bind")
        .arg(workspace.root().join("source"))
        .arg(workspace.root().join("source"))
        .arg("--ro-bind")
        .arg(workspace.script())
        .arg(workspace.virtual_script());
    for resource in workspace.trainer.read_resources() {
        command
            .arg("--ro-bind")
            .arg(workspace.root().join("source").join(resource))
            .arg(workspace.root().join("run").join(resource));
    }
    command
        .args(["--unshare-pid", "--die-with-parent"])
        .arg("--chdir")
        .arg(workspace.execution_root())
        .arg("--argv0")
        .arg(python.invocation())
        .arg("--")
        .arg(PINNED_PYTHON_EXECUTABLE)
        .env("HOME", workspace.root().join("home"))
        .env("LAMQUANT_LEGACY_MODE", "lma-direct-training")
        .env(
            "LAMQUANT_LEGACY_SOURCE_ROOT",
            workspace.root().join("source"),
        )
        .env("LAMQUANT_LEGACY_SOURCE_MANIFEST", workspace.manifest())
        .env("LAMQUANT_LEGACY_SOURCE_REVISION", &workspace.revision)
        .env("LAMQUANT_LEGACY_WORKSPACE", workspace.root())
        .env("WANDB_DIR", workspace.root().join("artifacts/wandb"))
        .env(
            "HF_HOME",
            workspace.root().join("artifacts/cache/huggingface"),
        )
        .env("TORCH_HOME", workspace.root().join("artifacts/cache/torch"))
        .env(
            "XDG_CACHE_HOME",
            workspace.root().join("artifacts/cache/xdg"),
        );
    for variable in environment {
        command.env(variable.name(), variable.value());
    }
    command
}

fn python_command(
    workspace: &FrozenWorkspace,
    sandbox: &ValidatedSandboxExecutable,
    python: &ValidatedPythonInterpreter,
    environment: &[LegacyEnvironment],
    args: &[OsString],
) -> Command {
    let mut command = isolated_python_command(workspace, sandbox, python, environment);
    command
        .args(["-I", "-B", "-c", ISOLATED_RUNNER])
        .arg(workspace.root().join("source/python"))
        .arg(workspace.script())
        .arg(workspace.virtual_script())
        .args(args);
    command
}

fn has_argument(args: &[OsString], flag: &str) -> bool {
    args.iter().any(|argument| {
        let argument = argument.to_string_lossy();
        argument == flag || argument.starts_with(&format!("{flag}="))
    })
}

fn argument_value<'a>(args: &'a [OsString], flag: &str) -> Option<&'a str> {
    for (index, argument) in args.iter().enumerate() {
        let Some(argument) = argument.to_str() else {
            continue;
        };
        if argument == flag {
            return args.get(index + 1)?.to_str();
        }
        if let Some(value) = argument
            .strip_prefix(flag)
            .and_then(|tail| tail.strip_prefix('='))
        {
            return Some(value);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::fs;

    fn executable_on_path(name: &str) -> PathBuf {
        std::env::var_os("PATH")
            .into_iter()
            .flat_map(|path| std::env::split_paths(&path).collect::<Vec<_>>())
            .map(|directory| directory.join(name))
            .find(|candidate| {
                candidate.metadata().is_ok_and(|metadata| {
                    metadata.is_file() && metadata.permissions().mode() & 0o111 != 0
                })
            })
            .unwrap_or_else(|| panic!("{name} executable exists on PATH"))
            .canonicalize()
            .expect("canonical executable path")
    }

    fn absolute_git() -> PathBuf {
        executable_on_path("git")
    }

    fn lma_direct_args(extra: &[&str]) -> Vec<OsString> {
        [
            "--lma-root",
            "/archive/lma",
            "--split-manifest",
            "/archive/split.json",
        ]
        .into_iter()
        .chain(extra.iter().copied())
        .map(OsString::from)
        .collect()
    }

    fn git(root: &Path, args: &[&str]) -> String {
        let output = Command::new(absolute_git())
            .arg("-C")
            .arg(root)
            .args(args)
            .output()
            .expect("run git");
        assert!(
            output.status.success(),
            "git {:?}: {}",
            args,
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout)
            .expect("git output is UTF-8")
            .trim()
            .to_owned()
    }

    fn verify_fixture_checkout_at(
        checkout: impl AsRef<Path>,
        trainer: LegacyTrainer,
        revision: &str,
    ) -> Result<VerifiedCheckout, VerificationError> {
        verify_checkout_at(&absolute_git(), checkout, trainer, revision)
    }

    fn fixture_checkout(script_source: &str) -> tempfile::TempDir {
        let checkout = tempfile::tempdir().expect("temporary checkout");
        git(checkout.path(), &["init", "-q"]);
        git(checkout.path(), &["config", "user.name", "Legacy Test"]);
        git(
            checkout.path(),
            &["config", "user.email", "legacy-test@example.invalid"],
        );
        for trainer in LegacyTrainer::ALL {
            let script = checkout.path().join(trainer.script());
            fs::create_dir_all(script.parent().expect("script parent"))
                .expect("create script parent");
            fs::write(&script, script_source).expect("write script");
        }
        fs::write(
            checkout.path().join("python/lamquant/legacy_probe.py"),
            "VALUE = 'tracked-probe'\n",
        )
        .expect("write import probe");
        fs::create_dir_all(checkout.path().join("python/lamquant/snn"))
            .expect("create frozen resource directory");
        fs::write(
            checkout.path().join("python/lamquant/snn/band_std.json"),
            "{}\n",
        )
        .expect("write frozen band statistics");
        git(checkout.path(), &["add", "."]);
        git(checkout.path(), &["commit", "-q", "-m", "fixture"]);
        checkout
    }

    #[test]
    fn checkout_verification_accepts_exact_clean_revision() {
        let checkout = fixture_checkout("print('legacy trainer')\n");
        let revision = git(checkout.path(), &["rev-parse", "HEAD"]);

        let verified =
            verify_fixture_checkout_at(checkout.path(), LegacyTrainer::TrainJoint, &revision)
                .expect("exact clean checkout");

        assert_eq!(verified.revision(), revision);
        assert_eq!(
            verified.script(),
            checkout
                .path()
                .canonicalize()
                .unwrap()
                .join("python/lamquant/student/train_joint.py")
        );
    }

    #[test]
    fn checkout_verification_requires_explicit_python_and_git_executables() {
        let checkout = fixture_checkout("print('legacy trainer')\n");
        let revision = git(checkout.path(), &["rev-parse", "HEAD"]);
        let git_error = verify_checkout("git", checkout.path(), LegacyTrainer::TrainJoint)
            .expect_err("relative Git path must fail");
        assert_eq!(git_error.failure(), VerificationFailure::InvalidGit);
        let git_error = verify_checkout(
            absolute_python(),
            checkout.path(),
            LegacyTrainer::TrainJoint,
        )
        .expect_err("non-Git executable must fail");
        assert_eq!(git_error.failure(), VerificationFailure::InvalidGit);

        let verified =
            verify_fixture_checkout_at(checkout.path(), LegacyTrainer::TrainJoint, &revision)
                .expect("verify interpreter fixture");
        let workspace_parent = tempfile::tempdir().expect("workspace parent");
        let workspace = workspace_parent.path().join("legacy-workspace");
        assert!(matches!(
            launch_verified_at(
                &verified,
                &revision,
                absolute_git(),
                &workspace,
                &[],
                &lma_direct_args(&[]),
            ),
            Err(LaunchError::InvalidPythonInterpreter(_))
        ));
    }

    #[test]
    fn invalid_arguments_fail_before_git_execution() {
        let parent = tempfile::tempdir().expect("fail-fast fixture");
        let marker = parent.path().join("git-ran");
        let git = parent.path().join("git");
        fs::write(
            &git,
            format!("#!/bin/sh\n: > '{}'\nexit 99\n", marker.display()),
        )
        .expect("write fake Git");
        fs::set_permissions(&git, fs::Permissions::from_mode(0o700))
            .expect("make fake Git executable");
        let error = launch(
            &git,
            parent.path().join("missing-checkout"),
            LegacyTrainer::TrainJoint,
            absolute_python(),
            parent.path().join("workspace"),
            &[],
            &[OsString::from("--training-snapshot")],
        )
        .expect_err("forbidden argument must fail before Git");

        assert!(matches!(error, LaunchError::InvalidArguments(_)));
        assert!(!marker.exists(), "invalid local arguments executed Git");
    }

    #[test]
    fn checkout_verification_rejects_wrong_revision_and_all_worktree_dirt() {
        let checkout = fixture_checkout("print('legacy trainer')\n");
        let wrong_revision = "0000000000000000000000000000000000000000";
        assert!(verify_fixture_checkout_at(
            checkout.path(),
            LegacyTrainer::TrainJoint,
            wrong_revision,
        )
        .is_err());

        let script = checkout
            .path()
            .join("python/lamquant/student/train_joint.py");
        let revision = git(checkout.path(), &["rev-parse", "HEAD"]);
        fs::write(&script, "print('modified')\n").expect("modify script");
        assert!(
            verify_fixture_checkout_at(checkout.path(), LegacyTrainer::TrainJoint, &revision)
                .is_err()
        );
        git(
            checkout.path(),
            &["checkout", "--", LegacyTrainer::TrainJoint.script()],
        );

        git(
            checkout.path(),
            &[
                "update-index",
                "--skip-worktree",
                LegacyTrainer::TrainJoint.script(),
            ],
        );
        fs::write(&script, "print('skip-worktree injection')\n")
            .expect("modify skip-worktree source");
        assert!(
            verify_fixture_checkout_at(checkout.path(), LegacyTrainer::TrainJoint, &revision)
                .is_err()
        );
        git(
            checkout.path(),
            &[
                "update-index",
                "--no-skip-worktree",
                LegacyTrainer::TrainJoint.script(),
            ],
        );
        git(
            checkout.path(),
            &["checkout", "--", LegacyTrainer::TrainJoint.script()],
        );

        git(
            checkout.path(),
            &[
                "update-index",
                "--assume-unchanged",
                LegacyTrainer::TrainJoint.script(),
            ],
        );
        fs::write(&script, "print('assume-unchanged injection')\n")
            .expect("modify assume-unchanged source");
        assert!(
            verify_fixture_checkout_at(checkout.path(), LegacyTrainer::TrainJoint, &revision)
                .is_err()
        );
        git(
            checkout.path(),
            &[
                "update-index",
                "--no-assume-unchanged",
                LegacyTrainer::TrainJoint.script(),
            ],
        );
        git(
            checkout.path(),
            &["checkout", "--", LegacyTrainer::TrainJoint.script()],
        );

        let untracked = checkout.path().join("python/sitecustomize.py");
        fs::write(&untracked, "raise RuntimeError('injected')\n").expect("write untracked");
        assert!(
            verify_fixture_checkout_at(checkout.path(), LegacyTrainer::TrainJoint, &revision)
                .is_err()
        );
        fs::remove_file(&untracked).expect("remove untracked");

        fs::write(
            checkout.path().join(".gitignore"),
            "python/sitecustomize.py\n",
        )
        .expect("write ignore rule");
        git(checkout.path(), &["add", ".gitignore"]);
        git(checkout.path(), &["commit", "-q", "-m", "ignore rule"]);
        let ignored_revision = git(checkout.path(), &["rev-parse", "HEAD"]);
        fs::write(&untracked, "raise RuntimeError('ignored injection')\n")
            .expect("write ignored file");
        assert!(verify_fixture_checkout_at(
            checkout.path(),
            LegacyTrainer::TrainJoint,
            &ignored_revision,
        )
        .is_err());
    }

    #[test]
    fn checkout_verification_does_not_execute_repository_fsmonitor() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            let checkout = fixture_checkout("print('legacy trainer')\n");
            let revision = git(checkout.path(), &["rev-parse", "HEAD"]);
            let hook = checkout.path().join(".git/fsmonitor-test");
            let marker = checkout.path().join(".git/fsmonitor-ran");
            fs::write(
                &hook,
                format!("#!/bin/sh\nprintf ran > '{}'\nexit 1\n", marker.display()),
            )
            .expect("write fsmonitor");
            let mut permissions = hook.metadata().expect("fsmonitor metadata").permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(&hook, permissions).expect("make fsmonitor executable");
            git(
                checkout.path(),
                &[
                    "config",
                    "core.fsmonitor",
                    hook.to_str().expect("UTF-8 hook path"),
                ],
            );

            verify_fixture_checkout_at(checkout.path(), LegacyTrainer::TrainJoint, &revision)
                .expect("verification ignores repository fsmonitor");
            assert!(!marker.exists());
        }
    }

    fn absolute_python() -> PathBuf {
        let python = executable_on_path("python3");
        let output = Command::new(&python)
            .args(["-c", "import sys; print(sys.executable)"])
            .output()
            .expect("locate Python");
        assert!(output.status.success());
        PathBuf::from(
            String::from_utf8(output.stdout)
                .expect("Python path is UTF-8")
                .trim(),
        )
        .canonicalize()
        .expect("canonical Python path")
    }

    fn make_tree_owner_writable(path: &Path) {
        let metadata = path.symlink_metadata().expect("tree metadata");
        if metadata.is_dir() {
            fs::set_permissions(path, fs::Permissions::from_mode(0o700))
                .expect("make directory writable");
            for entry in fs::read_dir(path).expect("read tree") {
                make_tree_owner_writable(&entry.expect("tree entry").path());
            }
        } else {
            fs::set_permissions(path, fs::Permissions::from_mode(0o600))
                .expect("make file writable");
        }
    }

    #[test]
    fn all_trainers_route_outputs_inside_persistent_workspace() {
        let workspace = Path::new("/tmp/lamquant-legacy-workspace-contract");
        let expected = [
            (
                LegacyTrainer::PretrainMae,
                vec![workspace.join("artifacts")],
            ),
            (
                LegacyTrainer::PretrainSslTueg,
                vec![workspace.join("artifacts")],
            ),
            (
                LegacyTrainer::Train4StateController,
                vec![
                    workspace.join("artifacts"),
                    workspace.join("run/python/weights/snn"),
                    workspace.join("run/python/training_logs"),
                ],
            ),
            (
                LegacyTrainer::TrainCombined,
                vec![
                    workspace.join("run/python/lamquant/oracle"),
                    workspace.join("run/python/lamquant/decoder"),
                ],
            ),
            (
                LegacyTrainer::TrainJoint,
                vec![
                    workspace.join("artifacts"),
                    workspace.join("run/python/training_logs"),
                ],
            ),
            (
                LegacyTrainer::TrainL3Teacher,
                vec![workspace.join("run/python/ai_models/oracle")],
            ),
            (
                LegacyTrainer::TrainVocosDecoder,
                vec![workspace.join("run/python/ai_models/decoder")],
            ),
        ];
        for (trainer, expected_roots) in expected {
            let roots = trainer.artifact_roots(workspace);
            assert_eq!(
                roots, expected_roots,
                "{trainer} frozen output roots drifted"
            );
            let managed = trainer.managed_arguments(workspace);
            for value in managed.iter().skip(1).step_by(2) {
                assert!(
                    Path::new(value).starts_with(workspace),
                    "{trainer} managed output escaped workspace: {value:?}"
                );
            }
        }
    }

    #[test]
    fn all_frozen_output_directories_exist_before_launch() {
        let checkout = fixture_checkout("if __name__ == '__main__': pass\n");
        let revision = git(checkout.path(), &["rev-parse", "HEAD"]);
        let parent = tempfile::tempdir().expect("workspace parent");
        for trainer in LegacyTrainer::ALL {
            let verified = verify_fixture_checkout_at(checkout.path(), trainer, &revision)
                .expect("verify trainer fixture");
            let workspace_path = parent.path().join(trainer.as_str());
            let workspace = FrozenWorkspace::open_or_create(&verified, &workspace_path)
                .expect("create frozen workspace");
            for directory in trainer.artifact_roots(&workspace_path) {
                let metadata = directory
                    .symlink_metadata()
                    .unwrap_or_else(|error| panic!("{}: {error}", directory.display()));
                assert!(metadata.is_dir());
                assert!(!metadata.file_type().is_symlink());
                assert_eq!(metadata.permissions().mode() & 0o077, 0);
            }
            drop(workspace);
        }
    }

    #[test]
    fn workspace_lock_rejects_concurrent_owners_and_recovers() {
        let parent = tempfile::tempdir().expect("workspace parent");
        let workspace = parent.path().join("legacy-workspace");
        let first = WorkspaceLock::acquire(&workspace).expect("first workspace lock");
        let error = WorkspaceLock::acquire(&workspace)
            .err()
            .expect("second workspace lock must fail");
        assert_eq!(error.kind(), io::ErrorKind::WouldBlock);
        drop(first);
        WorkspaceLock::acquire(&workspace).expect("workspace lock recovers after release");
    }

    #[test]
    fn only_bound_abandoned_staging_directory_is_recovered() {
        let checkout = fixture_checkout("if __name__ == '__main__': pass\n");
        let revision = git(checkout.path(), &["rev-parse", "HEAD"]);
        let verified =
            verify_fixture_checkout_at(checkout.path(), LegacyTrainer::TrainJoint, &revision)
                .expect("verify staging fixture");
        let parent = tempfile::tempdir().expect("workspace parent");
        let workspace = parent.path().join("legacy-workspace");
        let staging = parent.path().join(".legacy-workspace.staging");
        fs::create_dir(&staging).expect("create abandoned staging directory");
        fs::set_permissions(&staging, fs::Permissions::from_mode(0o700))
            .expect("protect staging directory");
        let projection = SourceProjection::from_verified(&verified).expect("staging projection");
        fs::write(
            staging.join(WORKSPACE_BINDING),
            workspace_binding(&workspace, &projection),
        )
        .expect("write staging binding");
        fs::set_permissions(
            staging.join(WORKSPACE_BINDING),
            fs::Permissions::from_mode(0o400),
        )
        .expect("protect staging binding");
        fs::write(staging.join("partial"), b"interrupted").expect("write partial staging data");

        FrozenWorkspace::open_or_create(&verified, &workspace)
            .expect("recover abandoned staging directory");

        assert!(workspace.is_dir());
        assert!(!staging.exists());
    }

    #[test]
    fn unbound_abandoned_staging_directory_fails_closed() {
        let checkout = fixture_checkout("if __name__ == '__main__': pass\n");
        let revision = git(checkout.path(), &["rev-parse", "HEAD"]);
        let verified =
            verify_fixture_checkout_at(checkout.path(), LegacyTrainer::TrainJoint, &revision)
                .expect("verify staging fixture");
        let parent = tempfile::tempdir().expect("workspace parent");
        let workspace = parent.path().join("legacy-workspace");
        let staging = parent.path().join(".legacy-workspace.staging");
        fs::create_dir(&staging).expect("create unbound staging directory");
        fs::set_permissions(&staging, fs::Permissions::from_mode(0o700))
            .expect("protect staging directory");
        fs::write(staging.join("unrelated"), b"must survive").expect("write unrelated data");

        assert!(FrozenWorkspace::open_or_create(&verified, &workspace).is_err());
        assert_eq!(
            fs::read(staging.join("unrelated")).expect("unbound data remains"),
            b"must survive"
        );
    }

    #[test]
    fn abandoned_staging_fifo_binding_fails_without_blocking() {
        let checkout = fixture_checkout("if __name__ == '__main__': pass\n");
        let revision = git(checkout.path(), &["rev-parse", "HEAD"]);
        let verified =
            verify_fixture_checkout_at(checkout.path(), LegacyTrainer::TrainJoint, &revision)
                .expect("verify FIFO staging fixture");
        let parent = tempfile::tempdir().expect("workspace parent");
        let workspace = parent.path().join("legacy-workspace");
        let staging = parent.path().join(".legacy-workspace.staging");
        fs::create_dir(&staging).expect("create staging directory");
        fs::set_permissions(&staging, fs::Permissions::from_mode(0o700))
            .expect("protect staging directory");
        let fifo = staging.join(WORKSPACE_BINDING);
        let fifo_c = CString::new(fifo.as_os_str().as_bytes()).expect("FIFO path");
        // SAFETY: path is NUL-free and points inside the private test directory.
        assert_eq!(unsafe { libc::mkfifo(fifo_c.as_ptr(), 0o400) }, 0);

        let started = Instant::now();
        assert!(FrozenWorkspace::open_or_create(&verified, &workspace).is_err());
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "FIFO metadata must fail before a blocking read"
        );
    }

    #[test]
    fn workspace_manifest_symlink_fails_closed() {
        use std::os::unix::fs::symlink;

        let checkout = fixture_checkout("if __name__ == '__main__': pass\n");
        let revision = git(checkout.path(), &["rev-parse", "HEAD"]);
        let verified =
            verify_fixture_checkout_at(checkout.path(), LegacyTrainer::TrainJoint, &revision)
                .expect("verify manifest-symlink fixture");
        let parent = tempfile::tempdir().expect("workspace parent");
        let workspace = parent.path().join("legacy-workspace");
        drop(
            FrozenWorkspace::open_or_create(&verified, &workspace)
                .expect("create manifest-symlink fixture"),
        );
        let manifest = workspace.join(SOURCE_MANIFEST);
        fs::remove_file(&manifest).expect("remove real manifest");
        let outside = parent.path().join("outside.manifest");
        fs::write(&outside, b"outside").expect("write outside manifest");
        fs::set_permissions(&outside, fs::Permissions::from_mode(0o400))
            .expect("protect outside manifest");
        symlink(&outside, &manifest).expect("replace manifest with symlink");

        assert!(FrozenWorkspace::open_or_create(&verified, &workspace).is_err());
    }

    #[test]
    fn argparse_abbreviations_cannot_override_managed_outputs() {
        let cases = [
            (LegacyTrainer::PretrainMae, "--outp"),
            (LegacyTrainer::PretrainMae, "--ou=/outside"),
            (LegacyTrainer::PretrainSslTueg, "--o"),
            (LegacyTrainer::Train4StateController, "--check"),
            (LegacyTrainer::TrainJoint, "--ckpt"),
            (LegacyTrainer::TrainJoint, "--resume-d=/outside"),
        ];
        for (trainer, argument) in cases {
            assert!(
                matches!(
                    validate_legacy_arguments(trainer, &[OsString::from(argument)]),
                    Err(LaunchError::ReservedArgument(_))
                ),
                "{trainer} accepted managed argparse abbreviation {argument}"
            );
        }
    }

    #[test]
    fn dependency_affecting_options_are_exact_and_singleton() {
        for abbreviated in ["--log", "--lr-s", "--target-s", "--dac-i"] {
            assert!(matches!(
                validate_legacy_arguments(
                    LegacyTrainer::TrainJoint,
                    &[
                        OsString::from(abbreviated),
                        OsString::from("--lma-root"),
                        OsString::from("/archive/lma"),
                        OsString::from("--split-manifest"),
                        OsString::from("/archive/split.json"),
                    ]
                ),
                Err(LaunchError::InvalidArguments(_))
            ));
        }
        for flag in ["--logger", "--lr-schedule", "--target-source", "--dac-init"] {
            assert!(matches!(
                validate_legacy_arguments(
                    LegacyTrainer::TrainJoint,
                    &[OsString::from(flag), OsString::from(flag)]
                ),
                Err(LaunchError::InvalidArguments(_))
            ));
        }
    }

    #[test]
    fn every_allowlisted_trainer_requires_exact_lma_direct_inputs() {
        for trainer in LegacyTrainer::ALL {
            validate_legacy_arguments(trainer, &lma_direct_args(&[]))
                .unwrap_or_else(|error| panic!("{trainer} rejected exact LMA inputs: {error}"));
            for args in [
                vec![
                    OsString::from("--split-manifest"),
                    OsString::from("/archive/split.json"),
                ],
                vec![OsString::from("--lma-root"), OsString::from("/archive/lma")],
                vec![
                    OsString::from("--lma-root="),
                    OsString::from("--split-manifest"),
                    OsString::from("/archive/split.json"),
                ],
                vec![
                    OsString::from("--lma-root"),
                    OsString::from("/archive/lma"),
                    OsString::from("--split-manifest"),
                ],
            ] {
                assert!(
                    matches!(
                        validate_legacy_arguments(trainer, &args),
                        Err(LaunchError::InvalidArguments(_))
                    ),
                    "{trainer} accepted incomplete LMA-direct inputs: {args:?}"
                );
            }
        }
        assert!(
            LegacyTrainer::from_str("train_teacher").is_err(),
            "NPZ/memmap-only train_teacher must remain outside rollback allowlist"
        );
    }

    #[test]
    fn frozen_exact_options_win_over_protected_prefixes() {
        for trainer in [
            LegacyTrainer::PretrainMae,
            LegacyTrainer::PretrainSslTueg,
            LegacyTrainer::Train4StateController,
            LegacyTrainer::TrainJoint,
            LegacyTrainer::TrainL3Teacher,
            LegacyTrainer::TrainVocosDecoder,
        ] {
            validate_legacy_arguments(trainer, &lma_direct_args(&["--lr"]))
                .unwrap_or_else(|error| panic!("{trainer} rejected exact --lr: {error}"));
        }
        validate_legacy_arguments(LegacyTrainer::TrainJoint, &lma_direct_args(&["--resume"]))
            .expect("train_joint exact --resume must not resolve to --resume-dir");
        assert!(matches!(
            validate_legacy_arguments(LegacyTrainer::TrainJoint, &[OsString::from("--res")]),
            Err(LaunchError::ReservedArgument(_))
        ));
        assert!(matches!(
            validate_legacy_arguments(LegacyTrainer::TrainJoint, &[OsString::from("--l")]),
            Err(LaunchError::InvalidArguments(_))
        ));
    }

    #[test]
    fn managed_output_symlink_is_rejected_before_launch() {
        use std::os::unix::fs::symlink;

        let checkout = fixture_checkout("if __name__ == '__main__': pass\n");
        let revision = git(checkout.path(), &["rev-parse", "HEAD"]);
        let verified =
            verify_fixture_checkout_at(checkout.path(), LegacyTrainer::PretrainMae, &revision)
                .expect("verify managed-output fixture");
        let parent = tempfile::tempdir().expect("workspace parent");
        let workspace_path = parent.path().join("legacy-workspace");
        let workspace = FrozenWorkspace::open_or_create(&verified, &workspace_path)
            .expect("create managed-output fixture");
        let outside = parent.path().join("outside.ckpt");
        fs::write(&outside, b"outside").expect("write outside target");
        symlink(
            &outside,
            workspace_path.join("artifacts/pretrained_mae.ckpt"),
        )
        .expect("create managed output symlink");

        assert!(validate_managed_destinations(&workspace).is_err());
        assert_eq!(
            fs::read(outside).expect("outside target remains"),
            b"outside"
        );
    }

    #[test]
    fn sandbox_enforces_source_immutability_during_execution() {
        let checkout = fixture_checkout(concat!(
            "import os, pathlib\n",
            "if __name__ == '__main__':\n",
            "    source = pathlib.Path(os.environ['LAMQUANT_LEGACY_SOURCE_ROOT']) / ",
            "'python/lamquant/student/train_joint.py'\n",
            "    try:\n",
            "        source.chmod(0o600)\n",
            "        source.write_text('tampered')\n",
            "    except OSError:\n",
            "        pass\n",
            "    else:\n",
            "        raise RuntimeError('source mount was writable')\n",
            "    workspace = pathlib.Path(os.environ['LAMQUANT_LEGACY_WORKSPACE'])\n",
            "    (workspace / 'artifacts/source-read-only').write_text('PASS')\n",
        ));
        let revision = git(checkout.path(), &["rev-parse", "HEAD"]);
        let verified =
            verify_fixture_checkout_at(checkout.path(), LegacyTrainer::TrainJoint, &revision)
                .expect("verify source-sandbox fixture");
        let parent = tempfile::tempdir().expect("workspace parent");
        let workspace = parent.path().join("legacy-workspace");

        let status = launch_verified_at(
            &verified,
            &revision,
            absolute_python(),
            &workspace,
            &[],
            &lma_direct_args(&[]),
        )
        .expect("launch source-sandbox fixture");

        assert!(status.success());
        assert_eq!(
            fs::read(workspace.join("artifacts/source-read-only")).expect("sandbox result"),
            b"PASS"
        );
    }

    #[test]
    fn pid_namespace_destroys_detached_descendants_before_unlock() {
        let checkout = fixture_checkout(concat!(
            "import os, pathlib, time\n",
            "if __name__ == '__main__':\n",
            "    if os.fork() == 0:\n",
            "        os.setsid()\n",
            "        time.sleep(0.3)\n",
            "        workspace = pathlib.Path(os.environ['LAMQUANT_LEGACY_WORKSPACE'])\n",
            "        (workspace / 'artifacts/detached-escaped').write_text('BAD')\n",
            "        os._exit(0)\n",
            "    os._exit(0)\n",
        ));
        let revision = git(checkout.path(), &["rev-parse", "HEAD"]);
        let verified =
            verify_fixture_checkout_at(checkout.path(), LegacyTrainer::TrainJoint, &revision)
                .expect("verify detached-child fixture");
        let parent = tempfile::tempdir().expect("workspace parent");
        let workspace = parent.path().join("legacy-workspace");

        let status = launch_verified_at(
            &verified,
            &revision,
            absolute_python(),
            &workspace,
            &[],
            &lma_direct_args(&[]),
        )
        .expect("launch detached-child fixture");

        assert!(status.success());
        std::thread::sleep(Duration::from_millis(500));
        assert!(!workspace.join("artifacts/detached-escaped").exists());
    }

    #[test]
    fn termination_signal_allows_bounded_trainer_cleanup() {
        let checkout = fixture_checkout(concat!(
            "import os, pathlib, signal, sys, time\n",
            "if __name__ == '__main__':\n",
            "    workspace = pathlib.Path(os.environ['LAMQUANT_LEGACY_WORKSPACE'])\n",
            "    def stop(_signal, _frame):\n",
            "        (workspace / 'artifacts/graceful-flush').write_text('PASS')\n",
            "        sys.exit(0)\n",
            "    signal.signal(signal.SIGTERM, stop)\n",
            "    (workspace / 'artifacts/ready').write_text('READY')\n",
            "    while True:\n",
            "        time.sleep(0.05)\n",
        ));
        let revision = git(checkout.path(), &["rev-parse", "HEAD"]);
        let verified =
            verify_fixture_checkout_at(checkout.path(), LegacyTrainer::TrainJoint, &revision)
                .expect("verify signal fixture");
        let parent = tempfile::tempdir().expect("workspace parent");
        let workspace_path = parent.path().join("legacy-workspace");
        let workspace = FrozenWorkspace::open_or_create(&verified, &workspace_path)
            .expect("create signal fixture");
        let ready = workspace_path.join("artifacts/ready");
        let sender = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(5);
            while Instant::now() < deadline && !ready.exists() {
                std::thread::sleep(POLL_INTERVAL);
            }
            assert!(ready.exists(), "trainer never became signal-ready");
            PENDING_TERMINATION_SIGNAL.store(libc::SIGTERM, Ordering::Release);
        });

        let sandbox = validate_sandbox_executable().expect("validate sandbox");
        let python = validate_python_interpreter(absolute_python().as_os_str())
            .expect("validate Python interpreter");
        let status = supervised_status(python_command(&workspace, &sandbox, &python, &[], &[]))
            .expect("signal-supervised trainer");
        sender.join().expect("signal sender");

        assert_eq!(status.code(), Some(128 + libc::SIGTERM));
        assert_eq!(
            fs::read(workspace_path.join("artifacts/graceful-flush"))
                .expect("graceful flush marker"),
            b"PASS"
        );
    }

    #[test]
    fn workspace_parent_must_be_private_and_owned() {
        let checkout = fixture_checkout("if __name__ == '__main__': pass\n");
        let parent = tempfile::tempdir().expect("workspace parent");
        fs::set_permissions(parent.path(), fs::Permissions::from_mode(0o770))
            .expect("make parent group-writable");

        let error = validated_workspace_path(checkout.path(), &parent.path().join("workspace"))
            .expect_err("group-writable workspace parent must fail");

        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
        fs::set_permissions(parent.path(), fs::Permissions::from_mode(0o700))
            .expect("restore parent for cleanup");
    }

    #[test]
    fn supervised_output_kills_overflowing_and_timed_out_children() {
        let python = absolute_python();
        let mut overflowing = Command::new(&python);
        overflowing.args([
            "-I",
            "-B",
            "-c",
            "import os\nwhile True: os.write(1, b'x' * 4096)\n",
        ]);
        let error =
            bounded_command_output(overflowing, 1024, Duration::from_secs(2), "overflow probe")
                .expect_err("overflowing child must be killed");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);

        let mut hanging = Command::new(python);
        hanging.args(["-I", "-B", "-c", "import time; time.sleep(60)"]);
        let started = Instant::now();
        let error =
            bounded_command_output(hanging, 1024, Duration::from_millis(100), "timeout probe")
                .expect_err("hanging child must be killed");
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn python_handshake_accepts_non_utf8_unix_executable_paths() {
        let parent = tempfile::tempdir().expect("non-UTF-8 executable parent");
        let component = OsString::from_vec(vec![b'p', b'y', 0xff]);
        let directory = parent.path().join(component);
        fs::create_dir(&directory).expect("create non-UTF-8 directory");
        let executable = directory.join("python3");
        let python = absolute_python();
        if fs::hard_link(&python, &executable).is_err() {
            fs::copy(&python, &executable).expect("copy Python executable");
        }

        let validated =
            validate_python_interpreter(executable.as_os_str()).expect("validate non-UTF-8 Python");

        assert_eq!(validated.invocation(), executable);
        assert_eq!(validated.target, executable.canonicalize().unwrap());
    }

    #[test]
    fn python_target_inode_replacement_fails_revalidation() {
        let parent = tempfile::tempdir().expect("interpreter parent");
        let executable = parent.path().join("python");
        fs::copy(absolute_python(), &executable).expect("copy Python interpreter");
        let validated =
            validate_python_interpreter(executable.as_os_str()).expect("validate copied Python");
        fs::rename(&executable, parent.path().join("python.original"))
            .expect("move validated interpreter inode");
        fs::copy(absolute_python(), &executable).expect("replace Python at same canonical path");

        assert!(matches!(
            validated.revalidate(),
            Err(LaunchError::InvalidPythonInterpreter(_))
        ));
    }

    #[test]
    fn python_in_place_mutation_fails_revalidation() {
        let parent = tempfile::tempdir().expect("interpreter parent");
        let executable = parent.path().join("python");
        fs::copy(absolute_python(), &executable).expect("copy Python interpreter");
        let validated =
            validate_python_interpreter(executable.as_os_str()).expect("validate copied Python");
        let inode = executable.metadata().expect("Python metadata").ino();
        let mut replacement = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&executable)
            .expect("open Python for in-place replacement");
        replacement
            .write_all(b"#!/bin/sh\nexit 0\n")
            .expect("replace Python bytes in place");
        replacement.sync_all().expect("sync Python replacement");
        assert_eq!(
            executable.metadata().expect("replacement metadata").ino(),
            inode,
            "test must preserve the validated inode"
        );

        assert!(matches!(
            validated.revalidate(),
            Err(LaunchError::InvalidPythonInterpreter(_))
        ));
    }

    #[test]
    fn oversized_sparse_python_fails_before_copy_or_handshake() {
        let parent = tempfile::tempdir().expect("interpreter parent");
        let executable = parent.path().join("python");
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&executable)
            .expect("create sparse Python");
        file.set_len(MAX_EXECUTABLE_BYTES + 1)
            .expect("size sparse Python");
        drop(file);
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o700))
            .expect("make sparse Python executable");

        assert!(matches!(
            validate_python_interpreter(executable.as_os_str()),
            Err(LaunchError::InvalidPythonInterpreter(_))
        ));
    }

    #[test]
    fn sandbox_in_place_mutation_fails_revalidation() {
        let parent = tempfile::tempdir().expect("sandbox parent");
        let executable = parent.path().join("bwrap");
        fs::copy(SANDBOX_EXECUTABLE, &executable).expect("copy sandbox executable");
        let target = executable
            .canonicalize()
            .expect("canonical sandbox executable");
        let mut input = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(&target)
            .expect("open copied sandbox");
        let metadata = input.metadata().expect("sandbox metadata");
        let validated = ValidatedSandboxExecutable {
            invocation: executable.clone(),
            target,
            target_device: metadata.dev(),
            target_inode: metadata.ino(),
            target_sha256: file_sha256_from(&mut input).expect("hash copied sandbox"),
            executable: input,
        };
        let mut replacement = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&executable)
            .expect("open sandbox for in-place replacement");
        replacement
            .write_all(b"#!/bin/sh\nexit 0\n")
            .expect("replace sandbox bytes in place");
        replacement.sync_all().expect("sync sandbox replacement");
        assert_eq!(
            executable.metadata().expect("replacement metadata").ino(),
            validated.target_inode,
            "test must preserve validated sandbox inode"
        );

        assert!(matches!(
            validated.revalidate(),
            Err(LaunchError::InvalidSandbox(_))
        ));
    }

    #[test]
    fn sealed_python_image_survives_post_validation_path_mutation() {
        let parent = tempfile::tempdir().expect("interpreter parent");
        let executable = parent.path().join("python");
        fs::copy(absolute_python(), &executable).expect("copy Python interpreter");
        let validated =
            validate_python_interpreter(executable.as_os_str()).expect("validate copied Python");
        let mut replacement = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&executable)
            .expect("open Python for post-validation replacement");
        replacement
            .write_all(b"#!/bin/sh\nexit 99\n")
            .expect("replace original Python path");
        replacement.sync_all().expect("sync Python replacement");

        let mut command = validated.direct_command();
        command.args(["-I", "-B", "-c", "print('PINNED_PYTHON_PASS')"]);
        let output = command.output().expect("execute sealed Python image");
        assert!(output.status.success());
        assert_eq!(output.stdout, b"PINNED_PYTHON_PASS\n");
    }

    #[test]
    fn real_venv_interpreter_keeps_venv_site_packages_during_launch() {
        let venv_parent = tempfile::Builder::new()
            .prefix(".lamquant-venv-test-")
            .tempdir()
            .expect("venv parent");
        let venv = venv_parent.path().join("venv");
        let status = Command::new(absolute_python())
            .args(["-m", "venv", "--without-pip"])
            .arg(&venv)
            .status()
            .expect("create real Python venv");
        assert!(status.success(), "python3 -m venv failed");
        let venv_python = venv.join("bin/python");
        let site_packages = Command::new(&venv_python)
            .args([
                "-I",
                "-B",
                "-c",
                "import site; print(site.getsitepackages()[0])",
            ])
            .output()
            .expect("locate venv site-packages");
        assert!(site_packages.status.success());
        let site_packages = PathBuf::from(
            String::from_utf8(site_packages.stdout)
                .expect("site-packages path is UTF-8")
                .trim(),
        );
        fs::write(
            site_packages.join("lamquant_venv_probe.py"),
            "VALUE = 'venv-site-packages'\n",
        )
        .expect("install venv-only dependency");

        let checkout = fixture_checkout(concat!(
            "import os, pathlib\n",
            "from lamquant_venv_probe import VALUE\n",
            "if __name__ == '__main__':\n",
            "    workspace = pathlib.Path(os.environ['LAMQUANT_LEGACY_WORKSPACE'])\n",
            "    (workspace / 'artifacts/venv-import.txt').write_text(VALUE)\n",
        ));
        let revision = git(checkout.path(), &["rev-parse", "HEAD"]);
        let verified =
            verify_fixture_checkout_at(checkout.path(), LegacyTrainer::TrainJoint, &revision)
                .expect("verify venv fixture");
        let workspace_parent = tempfile::tempdir().expect("workspace parent");
        let workspace = workspace_parent.path().join("legacy-workspace");

        let status = launch_verified_at(
            &verified,
            &revision,
            &venv_python,
            &workspace,
            &[],
            &lma_direct_args(&[]),
        )
        .expect("launch with real venv interpreter");

        assert!(status.success(), "venv-backed trainer exited {status}");
        assert_eq!(
            fs::read_to_string(workspace.join("artifacts/venv-import.txt"))
                .expect("venv import result"),
            "venv-site-packages"
        );
    }

    fn blocking_handshake_executable(parent: &Path, name: &str) -> (PathBuf, PathBuf) {
        let executable = parent.join(name);
        let ready = parent.join(format!("{name}.ready"));
        fs::write(
            &executable,
            format!("#!/bin/sh\n: > '{}'\nexec /bin/sleep 60\n", ready.display()),
        )
        .expect("write blocking handshake executable");
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o700))
            .expect("make blocking handshake executable");
        (executable, ready)
    }

    fn signal_when_ready(ready: PathBuf) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(5);
            while Instant::now() < deadline && !ready.exists() {
                std::thread::sleep(POLL_INTERVAL);
            }
            assert!(ready.exists(), "handshake child never became ready");
            PENDING_TERMINATION_SIGNAL.store(libc::SIGTERM, Ordering::Release);
        })
    }

    #[test]
    fn git_handshake_signal_is_preserved_as_typed_interruption() {
        let parent = tempfile::tempdir().expect("fake Git parent");
        let (git, ready) = blocking_handshake_executable(parent.path(), "git");
        let sender = signal_when_ready(ready);

        let error = validate_git_executable(git.as_os_str())
            .expect_err("signal must interrupt Git handshake");
        sender.join().expect("signal sender");

        assert_eq!(error.interrupted_signal(), Some(libc::SIGTERM));
    }

    #[test]
    fn python_handshake_signal_is_preserved_as_typed_interruption() {
        let parent = tempfile::tempdir().expect("fake Python parent");
        let (python, ready) = blocking_handshake_executable(parent.path(), "python");
        let sender = signal_when_ready(ready);

        let error = validate_python_interpreter(python.as_os_str())
            .expect_err("signal must interrupt Python handshake");
        sender.join().expect("signal sender");

        assert_eq!(error.interrupted_signal(), Some(libc::SIGTERM));
    }

    #[test]
    fn dependency_preflight_signal_is_preserved_as_typed_interruption() {
        let checkout = fixture_checkout("if __name__ == '__main__': pass\n");
        let revision = git(checkout.path(), &["rev-parse", "HEAD"]);
        let verified =
            verify_fixture_checkout_at(checkout.path(), LegacyTrainer::TrainJoint, &revision)
                .expect("verify preflight signal fixture");
        let workspace_parent = tempfile::tempdir().expect("workspace parent");
        let workspace_path = workspace_parent.path().join("legacy-workspace");
        let workspace = FrozenWorkspace::open_or_create(&verified, &workspace_path)
            .expect("create preflight signal workspace");
        let executable_parent = tempfile::Builder::new()
            .prefix(".lamquant-preflight-signal-")
            .tempdir_in(std::env::current_dir().expect("current directory"))
            .expect("fake Python parent");
        let python = executable_parent.path().join("python");
        fs::write(
            &python,
            concat!(
                "#!/bin/sh\n",
                ": > \"$LAMQUANT_LEGACY_WORKSPACE/artifacts/dependency-signal.ready\"\n",
                "exec /bin/sleep 60\n",
            ),
        )
        .expect("write blocking preflight Python");
        fs::set_permissions(&python, fs::Permissions::from_mode(0o700))
            .expect("make blocking preflight Python executable");
        let target = python.canonicalize().expect("canonical fake Python");
        let mut input = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(&target)
            .expect("open fake Python");
        let metadata = input.metadata().expect("fake Python metadata");
        let target_sha256 = file_sha256_from(&mut input).expect("hash fake Python");
        let sealed_image =
            sealed_executable_snapshot(&mut input, "preflight-signal").expect("seal fake Python");
        let python = ValidatedPythonInterpreter {
            target,
            target_device: metadata.dev(),
            target_inode: metadata.ino(),
            target_sha256,
            invocation: python,
            sealed_image,
            environment: None,
        };
        let sandbox = validate_sandbox_executable().expect("validate sandbox");
        let sender = signal_when_ready(workspace_path.join("artifacts/dependency-signal.ready"));

        let error = dependency_preflight(&workspace, &sandbox, &python, &[], Vec::new())
            .expect_err("signal must interrupt dependency preflight");
        sender.join().expect("signal sender");

        assert_eq!(error.interrupted_signal(), Some(libc::SIGTERM));
    }

    #[test]
    fn preflight_workspace_symlink_fails_before_host_report_write() {
        let parent = tempfile::tempdir().expect("workspace parent");
        let outside = parent.path().join("outside");
        fs::create_dir(&outside).expect("create outside directory");
        let checkout = fixture_checkout(&format!(
            "import os, pathlib, shutil\n\
             workspace = pathlib.Path(os.environ['LAMQUANT_LEGACY_WORKSPACE'])\n\
             shutil.rmtree(workspace / 'artifacts')\n\
             os.symlink({outside:?}, workspace / 'artifacts')\n\
             if __name__ == '__main__': pass\n",
            outside = outside.as_os_str(),
        ));
        let revision = git(checkout.path(), &["rev-parse", "HEAD"]);
        let verified =
            verify_fixture_checkout_at(checkout.path(), LegacyTrainer::TrainJoint, &revision)
                .expect("verify preflight-symlink fixture");
        let workspace_path = parent.path().join("legacy-workspace");
        let workspace = FrozenWorkspace::open_or_create(&verified, &workspace_path)
            .expect("create preflight-symlink workspace");
        let python =
            validate_python_interpreter(absolute_python().as_os_str()).expect("validate Python");
        let sandbox = validate_sandbox_executable().expect("validate sandbox");

        assert!(matches!(
            preflight_dependency_environment(
                &workspace,
                &verified,
                &sandbox,
                &python,
                &[],
                Vec::new()
            ),
            Err(LaunchError::Workspace(_))
        ));
        assert!(
            !outside.join(DEPENDENCY_REPORT).exists(),
            "host report write followed a preflight-created parent symlink"
        );
        fs::remove_file(workspace_path.join("artifacts")).expect("remove attack symlink");
        fs::create_dir(workspace_path.join("artifacts")).expect("restore artifacts directory");
        make_tree_owner_writable(&workspace_path);
    }

    #[test]
    fn workspace_reuse_rejects_artifact_symlink() {
        use std::os::unix::fs::symlink;

        let checkout = fixture_checkout("if __name__ == '__main__': pass\n");
        let revision = git(checkout.path(), &["rev-parse", "HEAD"]);
        let verified =
            verify_fixture_checkout_at(checkout.path(), LegacyTrainer::TrainJoint, &revision)
                .expect("verify artifact fixture");
        let parent = tempfile::tempdir().expect("workspace parent");
        let workspace = parent.path().join("legacy-workspace");
        drop(
            FrozenWorkspace::open_or_create(&verified, &workspace)
                .expect("create artifact fixture"),
        );
        fs::write(
            workspace.join("artifacts/checkpoints/model.pt"),
            b"checkpoint",
        )
        .expect("write durable checkpoint");
        symlink("checkpoints/model.pt", workspace.join("artifacts/latest"))
            .expect("create durable artifact symlink");

        assert!(
            FrozenWorkspace::open_or_create(&verified, &workspace).is_err(),
            "artifact symlinks can redirect derived trainer outputs"
        );
    }

    #[test]
    fn complete_writable_workspace_rejects_preplanted_hardlink() {
        let checkout = fixture_checkout("if __name__ == '__main__': pass\n");
        let revision = git(checkout.path(), &["rev-parse", "HEAD"]);
        let verified =
            verify_fixture_checkout_at(checkout.path(), LegacyTrainer::TrainJoint, &revision)
                .expect("verify hardlink fixture");
        let parent = tempfile::tempdir().expect("workspace parent");
        let workspace_path = parent.path().join("legacy-workspace");
        let workspace = FrozenWorkspace::open_or_create(&verified, &workspace_path)
            .expect("create hardlink fixture");
        let outside = parent.path().join("outside.ckpt");
        fs::write(&outside, b"outside").expect("write outside checkpoint");
        fs::hard_link(&outside, workspace_path.join("home/derived.ckpt"))
            .expect("preplant writable-home hardlink");

        assert!(validate_managed_destinations(&workspace).is_err());
        assert_eq!(
            fs::read(outside).expect("outside target remains"),
            b"outside"
        );
    }

    #[test]
    fn workspace_reuse_rejects_source_mutation_and_import_injection() {
        let checkout = fixture_checkout(concat!(
            "if __name__ == '__main__':\n",
            "    print('legacy trainer')\n",
        ));
        let revision = git(checkout.path(), &["rev-parse", "HEAD"]);
        let verified =
            verify_fixture_checkout_at(checkout.path(), LegacyTrainer::TrainJoint, &revision)
                .expect("verify workspace fixture");
        let parent = tempfile::tempdir().expect("workspace parent");
        let workspace = parent.path().join("legacy-workspace");
        FrozenWorkspace::open_or_create(&verified, &workspace).expect("create workspace");

        let source = workspace
            .join("source")
            .join(LegacyTrainer::TrainJoint.script());
        fs::set_permissions(&source, fs::Permissions::from_mode(0o600))
            .expect("make source writable for tamper probe");
        fs::write(&source, "raise RuntimeError('tampered')\n").expect("tamper source");
        assert!(FrozenWorkspace::open_or_create(&verified, &workspace).is_err());

        make_tree_owner_writable(&workspace);
        fs::remove_dir_all(&workspace).expect("remove tampered workspace");
        FrozenWorkspace::open_or_create(&verified, &workspace).expect("recreate workspace");
        fs::write(
            workspace.join("run/python/lamquant/injected.py"),
            "raise RuntimeError('injected')\n",
        )
        .expect("write import injection");
        assert!(FrozenWorkspace::open_or_create(&verified, &workspace).is_err());
        make_tree_owner_writable(&workspace);
    }

    #[test]
    fn isolated_launcher_preserves_argv_scrubs_code_paths_and_leaves_checkout_clean() {
        let checkout = fixture_checkout(concat!(
            "import os, pathlib, sys\n",
            "from lamquant.legacy_probe import VALUE\n",
            "assert VALUE == 'tracked-probe'\n",
            "assert 'BLUT_AI_MODELS' not in os.environ\n",
            "assert 'LD_PRELOAD' not in os.environ\n",
            "assert 'LD_LIBRARY_PATH' not in os.environ\n",
            "assert 'PATH' not in os.environ\n",
            "assert not (pathlib.Path(__file__).stat().st_mode & 0o222)\n",
            "assert os.environ.get('LAMQUANT_LEGACY_MODE') == ",
            "'lma-direct-training'\n",
            "if __name__ == '__main__':\n",
            "    workspace = pathlib.Path(os.environ['LAMQUANT_LEGACY_WORKSPACE'])\n",
            "    assert pathlib.Path(__file__).resolve().is_relative_to(workspace)\n",
            "    (workspace / 'artifacts' / 'argv.txt').write_text(",
            "'\\n'.join(sys.argv[1:]))\n",
        ));
        let revision = git(checkout.path(), &["rev-parse", "HEAD"]);
        let verified =
            verify_fixture_checkout_at(checkout.path(), LegacyTrainer::TrainJoint, &revision)
                .expect("verify argv fixture");
        let workspace_parent = tempfile::tempdir().expect("workspace parent");
        let workspace_path = workspace_parent.path().join("legacy-workspace");
        let workspace = FrozenWorkspace::open_or_create(&verified, &workspace_path)
            .expect("frozen fixture workspace");
        let sandbox = validate_sandbox_executable().expect("validate sandbox");
        let python = validate_python_interpreter(absolute_python().as_os_str())
            .expect("validate Python interpreter");
        let command = python_command(&workspace, &sandbox, &python, &[], &[]);
        assert!(command
            .get_envs()
            .all(|(name, _)| name != OsStr::new("BLUT_AI_MODELS")));
        drop(workspace);
        let args = vec![
            OsString::from("--lma-root"),
            OsString::from("/archive/lma"),
            OsString::from("--split-manifest"),
            OsString::from("/archive/split.json"),
        ];

        for snapshot_argument in [
            "--training-snapshot",
            "--training-snapshot=x",
            "--training-snap",
        ] {
            assert!(matches!(
                validate_legacy_arguments(
                    LegacyTrainer::TrainJoint,
                    &[OsString::from(snapshot_argument)],
                ),
                Err(LaunchError::InvalidArguments(_))
            ));
        }
        assert!(matches!(
            launch_verified_at(
                &verified,
                &revision,
                OsStr::new("python3"),
                &workspace_path,
                &[],
                &args,
            ),
            Err(LaunchError::InvalidPythonInterpreter(_))
        ));
        let python = absolute_python();
        let status = launch_verified_at(&verified, &revision, &python, &workspace_path, &[], &args)
            .expect("launch exact legacy trainer");

        assert!(status.success());
        let recorded =
            fs::read_to_string(workspace_path.join("artifacts/argv.txt")).expect("recorded argv");
        assert!(recorded.contains("--ckpt-dir"));
        assert!(recorded.contains(
            workspace_path
                .join("artifacts/checkpoints")
                .to_str()
                .expect("UTF-8 workspace")
        ));
        assert!(
            recorded.ends_with("--lma-root\n/archive/lma\n--split-manifest\n/archive/split.json")
        );
        assert!(workspace_path.join(SOURCE_MANIFEST).is_file());
        assert!(workspace_path
            .join("artifacts")
            .join(DEPENDENCY_REPORT)
            .is_file());
        verify_fixture_checkout_at(checkout.path(), LegacyTrainer::TrainJoint, &revision)
            .expect("launch leaves checkout clean");
        let status = launch_verified_at(&verified, &revision, &python, &workspace_path, &[], &args)
            .expect("repeat launch");
        assert!(status.success());
        assert!(!checkout.path().join("python/lamquant/__pycache__").exists());
        assert!(workspace_path.join("artifacts/argv.txt").is_file());
        make_tree_owner_writable(&workspace_path);
    }

    #[test]
    fn launcher_rejects_non_python_executable() {
        let checkout = fixture_checkout("raise AssertionError('must not run')\n");
        let revision = git(checkout.path(), &["rev-parse", "HEAD"]);
        let verified =
            verify_fixture_checkout_at(checkout.path(), LegacyTrainer::TrainJoint, &revision)
                .expect("verify interpreter fixture");
        let workspace_parent = tempfile::tempdir().expect("workspace parent");
        let workspace = workspace_parent.path().join("legacy-workspace");

        assert!(matches!(
            launch_verified_at(
                &verified,
                &revision,
                absolute_git(),
                workspace,
                &[],
                &lma_direct_args(&[]),
            ),
            Err(LaunchError::InvalidPythonInterpreter(_))
        ));
    }
}
