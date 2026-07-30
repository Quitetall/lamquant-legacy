"""
LamQuant configuration — TOML file + CLI flags + env vars.

Hierarchy (first wins): CLI flags > config file > env vars > defaults.
Locations: --config PATH > ./lamquant.toml > $XDG_CONFIG_HOME > /etc > built-in.
"""
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List


@dataclass
class OutputConfig:
    refresh_hz: float = 10.0
    color: str = "auto"           # auto | always | never
    charset: str = "auto"         # auto | unicode | ascii
    dashboard_width: int = 0      # 0 = auto
    show_spinner: bool = True
    spinner_style: str = "braille"
    verbose_per_file: bool = False
    truncate_filenames: bool = True
    show_banner: bool = True
    show_summary: bool = True
    json_summary: bool = False
    splash_duration: float = 0.5    # seconds, 0 = off
    autocomplete: bool = True       # prompt_toolkit tab-completion
    potato_mode: bool = False       # minimal UI: no splash, no banner, no spinner, ascii
    allow_root: bool = False        # allow running as root/sudo
    warn_root: bool = True          # show warning banner when running as root
    instant_nav: bool = False       # single keypress navigation (no Enter needed)


@dataclass
class LosslessConfig:
    entropy_coder: str = "golomb_rice"
    lpc_order: int = 2
    use_lifting: bool = True


@dataclass
class CodecConfig:
    default_mode: str = "lossless"
    input_bits: int = 16
    window_samples: int = 2500
    noise_bits: int = 0
    verification: str = "standard"  # paranoid | standard | fast
    lossless: LosslessConfig = field(default_factory=LosslessConfig)


@dataclass
class ComputeConfig:
    workers: int = 0              # 0 = auto (cpu_count - 1)
    memory_limit_gib: float = 0
    numba_cache_dir: str = "auto"


@dataclass
class IntegrityConfig:
    window_checksum: str = "crc32"
    file_checksum: str = "sha256"
    verify_after_write: bool = True
    verify_outliers: bool = True
    reject_corrupted_input: bool = True
    refuse_double_strip: bool = True
    fail_fast: bool = False


@dataclass
class ResumeConfig:
    enabled: bool = True
    state_file: str = ".lamquant_state.json"
    checkpoint_strategy: str = "per_file"
    on_existing_state: str = "auto"
    skip_existing_output: bool = True
    verify_skipped: bool = False
    quarantine_dir: str = "quarantine"
    max_retries: int = 2
    retry_backoff_s: float = 5.0


@dataclass
class LoggingConfig:
    audit_log: str = "audit.log"
    append_audit: bool = True
    stderr_level: str = "WARNING"
    file_log: str = ""
    include_tracebacks: bool = False
    manifest: str = "manifest.lml.json"
    manifest_include_files: bool = True


@dataclass
class InputConfig:
    extensions: List[str] = field(default_factory=lambda: ["edf", "bdf"])
    recursive: bool = True
    follow_symlinks: bool = False
    min_file_size: int = 1024
    max_file_size: int = 0
    exclude_patterns: List[str] = field(
        default_factory=lambda: ["**/test/**", "**/.git/**"])


@dataclass
class OutputFilesConfig:
    extension: str = "lml"
    preserve_structure: bool = True
    atomic_writes: bool = True
    fsync_on_write: bool = True


@dataclass
class BackendConfig:
    """Compression backend selection.

    The Python CLI dispatches to a configurable backend for encode/decode.
    Default: 'auto' (Rust if available, Python fallback).

    Backend contract: any executable that accepts:
      <backend> encode <input> --output <output> --recursive --verify --skip-existing --threads N
      <backend> decode <input> --output <output> --recursive --skip-existing --threads N
    And produces LML v1 files with CRC-32 + SHA-256.
    """
    mode: str = "auto"           # auto | rust | python | custom
    rust_binary: str = "lml"     # path or name (searched in PATH + target/release/)
    custom_binary: str = ""      # path to custom backend (when mode=custom)

    def resolve(self) -> str:
        """Return the active backend: 'rust', 'python', or 'custom'."""
        if self.mode == "custom" and self.custom_binary:
            return "custom"
        if self.mode == "python":
            return "python"
        if self.mode == "rust":
            return "rust"
        # auto: try Rust first
        if _find_rust_binary(self.rust_binary):
            return "rust"
        return "python"


def _find_rust_binary(name: str) -> Optional[str]:
    """Find the lml Rust binary. Checks: explicit path, PATH, target/release/."""
    import shutil
    if os.path.isfile(name) and os.access(name, os.X_OK):
        return name
    found = shutil.which(name)
    if found:
        return found
    # Check common build locations relative to repo root
    for candidate in [
        os.path.join(os.path.dirname(__file__), '..', '..', 'target', 'release', 'lml'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'target', 'debug', 'lml'),
    ]:
        candidate = os.path.normpath(candidate)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


@dataclass
class LamQuantConfig:
    """Complete LamQuant configuration."""
    schema_version: str = "1.0"
    instance_name: str = "default"
    output: OutputConfig = field(default_factory=OutputConfig)
    codec: CodecConfig = field(default_factory=CodecConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    integrity: IntegrityConfig = field(default_factory=IntegrityConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    input: InputConfig = field(default_factory=InputConfig)
    output_files: OutputFilesConfig = field(default_factory=OutputFilesConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)

    def hash(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def effective_workers(self) -> int:
        if self.compute.workers > 0:
            return self.compute.workers
        cores = max(1, (os.cpu_count() or 4) - 2)
        # Cap by available RAM: ~500 MiB per worker, leave 2 GiB for OS
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        avail_gib = int(line.split()[1]) / (1024**2)
                        max_by_ram = max(1, int((avail_gib - 2.0) / 0.5))
                        return min(cores, max_by_ram)
        except Exception:
            pass
        return cores


def _find_config_file(cli_path: Optional[str] = None) -> Optional[Path]:
    """Search hierarchy for config file."""
    candidates = []
    if cli_path:
        candidates.append(Path(cli_path))
    candidates.append(Path("lamquant.toml"))
    xdg = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    candidates.append(Path(xdg) / "lamquant" / "config.toml")
    candidates.append(Path("/etc/lamquant/config.toml"))
    for p in candidates:
        if p.exists():
            return p
    return None


def _apply_env(cfg: LamQuantConfig):
    """Override config from LAMQUANT_* environment variables."""
    env_map = {
        "LAMQUANT_OUTPUT_REFRESH_HZ": ("output", "refresh_hz", float),
        "LAMQUANT_OUTPUT_COLOR": ("output", "color", str),
        "LAMQUANT_COMPUTE_WORKERS": ("compute", "workers", int),
        "LAMQUANT_INTEGRITY_VERIFY_AFTER_WRITE": ("integrity", "verify_after_write",
                                                   lambda x: x.lower() in ("1", "true", "yes")),
    }
    for env_key, (section, field_name, conv) in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            obj = getattr(cfg, section)
            setattr(obj, field_name, conv(val))


def _apply_toml(cfg: LamQuantConfig, path: Path):
    """Merge TOML file into config."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            print(f"WARNING: {path} found but tomllib/tomli not installed. "
                  f"Using defaults.", file=sys.stderr)
            return

    with open(path, "rb") as f:
        data = tomllib.load(f)

    # Flatten and apply
    section_map = {
        "output": cfg.output,
        "codec": cfg.codec,
        "compute": cfg.compute,
        "integrity": cfg.integrity,
        "resume": cfg.resume,
        "logging": cfg.logging,
        "input": cfg.input,
        "output_files": cfg.output_files,
        "backend": cfg.backend,
    }
    for section_name, obj in section_map.items():
        section_data = data.get(section_name, {})
        for k, v in section_data.items():
            if hasattr(obj, k):
                setattr(obj, k, v)


def load_config(cli_path: Optional[str] = None,
                cli_overrides: Optional[dict] = None) -> LamQuantConfig:
    """Load config from file + env + CLI overrides."""
    cfg = LamQuantConfig()

    # TOML file
    config_file = _find_config_file(cli_path)
    if config_file:
        _apply_toml(cfg, config_file)

    # Environment variables
    _apply_env(cfg)

    # CLI overrides (highest priority)
    if cli_overrides:
        for dotkey, val in cli_overrides.items():
            parts = dotkey.split(".")
            obj = cfg
            for p in parts[:-1]:
                try:
                    obj = getattr(obj, p)
                except AttributeError:
                    raise ValueError(
                        f"Invalid config key '{dotkey}': section '{p}' does not exist")
            if not hasattr(obj, parts[-1]):
                raise ValueError(
                    f"Invalid config key '{dotkey}': field '{parts[-1]}' does not exist")
            setattr(obj, parts[-1], val)

    _validate_config(cfg)
    return cfg


def _validate_config(cfg: LamQuantConfig):
    """Validate config field types and ranges after loading."""
    if not isinstance(cfg.compute.workers, int) or cfg.compute.workers < 0:
        raise ValueError(f"compute.workers must be non-negative int, got {cfg.compute.workers!r}")
    if not isinstance(cfg.codec.noise_bits, int) or not (0 <= cfg.codec.noise_bits <= 15):
        raise ValueError(f"codec.noise_bits must be 0-15, got {cfg.codec.noise_bits!r}")
    if not isinstance(cfg.codec.window_samples, int) or cfg.codec.window_samples <= 0:
        raise ValueError(f"codec.window_samples must be positive int, got {cfg.codec.window_samples!r}")
    if cfg.output.refresh_hz <= 0:
        raise ValueError(f"output.refresh_hz must be positive, got {cfg.output.refresh_hz}")


def generate_default_config(cfg: Optional['LamQuantConfig'] = None) -> str:
    """Generate TOML config string from a config (or defaults)."""
    if cfg is None:
        cfg = LamQuantConfig()
    workers = cfg.effective_workers() if cfg.compute.workers == 0 else cfg.compute.workers
    return f"""# LamQuant Configuration v1.0
# Generated by: lamquant settings

[backend]
mode = "{cfg.backend.mode}"
rust_binary = "{cfg.backend.rust_binary}"
custom_binary = "{cfg.backend.custom_binary}"

[codec]
default_mode = "{cfg.codec.default_mode}"
input_bits = {cfg.codec.input_bits}
window_samples = {cfg.codec.window_samples}
noise_bits = {cfg.codec.noise_bits}
verification = "{cfg.codec.verification}"

[compute]
workers = {workers}

[integrity]
verify_after_write = {str(cfg.integrity.verify_after_write).lower()}
verify_outliers = {str(cfg.integrity.verify_outliers).lower()}
reject_corrupted_input = {str(cfg.integrity.reject_corrupted_input).lower()}
fail_fast = {str(cfg.integrity.fail_fast).lower()}

[resume]
enabled = {str(cfg.resume.enabled).lower()}
skip_existing_output = {str(cfg.resume.skip_existing_output).lower()}
max_retries = {cfg.resume.max_retries}

[output]
refresh_hz = {cfg.output.refresh_hz}
color = "{cfg.output.color}"
charset = "{cfg.output.charset}"
splash_duration = {cfg.output.splash_duration}
autocomplete = {str(cfg.output.autocomplete).lower()}
potato_mode = {str(cfg.output.potato_mode).lower()}
allow_root = {str(cfg.output.allow_root).lower()}
warn_root = {str(cfg.output.warn_root).lower()}
instant_nav = {str(cfg.output.instant_nav).lower()}
show_banner = {str(cfg.output.show_banner).lower()}
show_summary = {str(cfg.output.show_summary).lower()}

[logging]
audit_log = "{cfg.logging.audit_log}"
manifest = "{cfg.logging.manifest}"
stderr_level = "{cfg.logging.stderr_level}"

[input]
extensions = {list(cfg.input.extensions)}
recursive = {str(cfg.input.recursive).lower()}
follow_symlinks = {str(cfg.input.follow_symlinks).lower()}

[output_files]
extension = "{cfg.output_files.extension}"
preserve_structure = {str(cfg.output_files.preserve_structure).lower()}
atomic_writes = {str(cfg.output_files.atomic_writes).lower()}
fsync_on_write = {str(cfg.output_files.fsync_on_write).lower()}
"""
