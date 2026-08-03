"""
OpenHuman LamQuant — Interactive Menu

This module contains the menu rendering, input handling, history,
and terminal helpers. Zero heavy imports. Starts in <50ms.

The actual command implementations live in their own modules and
are imported lazily only when the user selects them.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from lamquant_codec._paths import REPO_ROOT as ROOT


# ────────────────────────────────────────────────────────────────────
# Terminal (no dependencies)
# ────────────────────────────────────────────────────────────────────

def _supports_color():
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty() and os.environ.get("TERM") != "dumb"

def _supports_unicode():
    try:
        "┌─┐│└─┘✓✗⠋".encode(sys.stdout.encoding or "utf-8")
        return True
    except (UnicodeEncodeError, LookupError):
        return False

_C = _supports_color()
_U = _supports_unicode()

DIM   = "\033[90m" if _C else ""
CYN   = "\033[36m" if _C else ""
GRN   = "\033[32m" if _C else ""
RED   = "\033[31m" if _C else ""
YEL   = "\033[33m" if _C else ""
BLD   = "\033[1m"  if _C else ""
RST   = "\033[0m"  if _C else ""

H  = "─" if _U else "-"
DOT = "·" if _U else "-"
OK = "✓" if _U else "OK"
NO = "✗" if _U else "X"


def clear(full=False):
    if sys.stdout.isatty():
        if full:
            # Clear screen + scrollback buffer, then home cursor
            sys.stdout.write("\033[3J\033[H\033[2J\033[H")
        else:
            sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()


# ────────────────────────────────────────────────────────────────────
# Prompt (optional prompt_toolkit)
# ────────────────────────────────────────────────────────────────────

try:
    from prompt_toolkit import prompt as _pt_prompt
    from prompt_toolkit.completion import PathCompleter, WordCompleter
    _HAS_PT = True
    _path_completer = PathCompleter(expanduser=True)
except ImportError:
    _HAS_PT = False

_autocomplete = True   # on by default (prompt_toolkit tab-completion)
_instant_nav = False   # single keypress, no Enter needed (default off)

def set_autocomplete(enabled: bool):
    global _autocomplete
    _autocomplete = enabled

def set_instant_nav(enabled: bool):
    global _instant_nav
    _instant_nav = enabled

def instant_input(msg="  > "):
    """Read a single keypress if instant_nav is on, else normal input()."""
    if not _instant_nav or not sys.stdout.isatty() or sys.platform == 'win32':
        return input(msg).strip().lower()
    import tty, termios
    sys.stdout.write(msg)
    sys.stdout.flush()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        sys.stdout.write(ch + "\n")
        sys.stdout.flush()
        return ch.lower()
    except (KeyboardInterrupt, EOFError):
        raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

def prompt(msg="  > ", completer=None):
    try:
        if completer and _HAS_PT and _autocomplete and sys.stdout.isatty():
            return _pt_prompt(msg, completer=completer).strip()
        return input(msg).strip()
    except (KeyboardInterrupt, EOFError):
        raise KeyboardInterrupt

def prompt_path(msg="  > "):
    if _HAS_PT and _autocomplete:
        return prompt(msg, completer=_path_completer)
    return input(msg).strip()

def prompt_menu(msg="  > ", options=None):
    if _HAS_PT and _autocomplete and options:
        return prompt(msg, completer=WordCompleter(options, ignore_case=True))
    return prompt(msg)


# ────────────────────────────────────────────────────────────────────
# Version (lazy, no heavy imports)
# ────────────────────────────────────────────────────────────────────

def version():
    try:
        from lamquant_codec import __version__
        return __version__
    except Exception:
        return "0.0.0"

def git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=2, cwd=ROOT)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"

def gen_tag():
    v = version()
    parts = v.split(".")
    return f"Gen {parts[0]}.{parts[1]}" if len(parts) >= 2 else f"v{v}"

def cli_version():
    try:
        from lamquant_codec import __cli_version__
        return __cli_version__
    except Exception:
        return "1.0.0"


# ────────────────────────────────────────────────────────────────────
# History (persistent, locked, atomic) — shared with Rust TUI + Tauri GUI
#
# On-disk schema is canonical at ``specs/history-schema.json``. The
# resolver mirrors the Rust ``crates/lamquant-history/src/lib.rs``
# precedence so all three front-ends read/write the same file.
# ────────────────────────────────────────────────────────────────────

CURRENT_HISTORY_SCHEMA = "2.0"
CURRENT_PARITY_VERSION = 2


class HistoryFormatError(ValueError):
    """history.json is malformed or violates the current structural contract."""


class HistoryParityError(ValueError):
    """history.json belongs to another front-end protocol generation."""


def _history_dir():
    """Resolve the per-OS history directory.

    Precedence (mirrors `lamquant_history::history_path` in Rust):
      1. ``LAMQUANT_HISTORY`` env (test override / advanced users).
      2. ``$XDG_CONFIG_HOME/lamquant/`` (Linux + explicit XDG).
      3. ``~/Library/Application Support/lamquant/`` (macOS).
      4. ``%APPDATA%\\lamquant\\`` (Windows).
      5. ``~/.config/lamquant/`` (Linux fallback).
    """
    override = os.environ.get("LAMQUANT_HISTORY")
    if override:
        return Path(override).parent
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "lamquant"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "lamquant"
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "lamquant"
    return Path.home() / ".config" / "lamquant"


def _history_path():
    """Full path to ``history.json`` (honours LAMQUANT_HISTORY override)."""
    override = os.environ.get("LAMQUANT_HISTORY")
    if override:
        return Path(override)
    return _history_dir() / "history.json"


def _empty_history():
    return {
        "schema_version": CURRENT_HISTORY_SCHEMA,
        "parity_version": CURRENT_PARITY_VERSION,
        "recent_operations": [],
        "recent_paths": {"inputs": [], "outputs": []},
        "interrupted": False,
        "last_op": None,
        "last_input": None,
        "last_output": None,
    }


def _validate_current_history(data):
    if not isinstance(data, dict):
        raise HistoryFormatError("history root must be an object")
    parity = data.get("parity_version", 1)
    if isinstance(parity, bool) or not isinstance(parity, int):
        raise HistoryFormatError("parity_version must be an integer")
    if parity != CURRENT_PARITY_VERSION:
        raise HistoryParityError(
            f"history parity mismatch: found {parity}, expected {CURRENT_PARITY_VERSION}; "
            "rerun setup wizard or downgrade"
        )
    required = {
        "schema_version",
        "parity_version",
        "recent_operations",
        "recent_paths",
    }
    optional = {"interrupted", "last_op", "last_input", "last_output"}
    missing = required - set(data)
    unknown = set(data) - required - optional
    if missing:
        raise HistoryFormatError(
            f"history missing required fields: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise HistoryFormatError(
            f"history contains unknown fields: {', '.join(sorted(unknown))}"
        )
    if data.get("schema_version") != CURRENT_HISTORY_SCHEMA:
        raise HistoryFormatError(
            f"history schema mismatch: found {data.get('schema_version')!r}, "
            f"expected {CURRENT_HISTORY_SCHEMA!r}"
        )
    operations = data["recent_operations"]
    paths = data["recent_paths"]
    if not isinstance(operations, list) or len(operations) > 50:
        raise HistoryFormatError("recent_operations must be an array of at most 50 entries")
    for index, operation in enumerate(operations):
        _validate_history_operation(operation, index)
    if not isinstance(paths, dict) or set(paths) != {"inputs", "outputs"}:
        raise HistoryFormatError("recent_paths must contain exactly inputs and outputs")
    for kind in ("inputs", "outputs"):
        values = paths[kind]
        if (
            not isinstance(values, list)
            or len(values) > 20
            or any(not isinstance(value, str) for value in values)
        ):
            raise HistoryFormatError(f"recent_paths.{kind} must be at most 20 strings")
    if not isinstance(data.get("interrupted", False), bool):
        raise HistoryFormatError("interrupted must be a boolean")
    for field in ("last_op", "last_input", "last_output"):
        value = data.get(field)
        if value is not None and not isinstance(value, str):
            raise HistoryFormatError(f"{field} must be a string or null")
    return data


def _validate_history_operation(operation, index=0):
    if not isinstance(operation, dict) or set(operation) != {
        "action",
        "target",
        "when",
        "result",
    }:
        raise HistoryFormatError(
            f"recent_operations[{index}] must contain exactly "
            "action, target, when, and result"
        )
    if (
        not isinstance(operation["action"], str)
        or not operation["action"]
        or not isinstance(operation["target"], str)
        or not isinstance(operation["when"], str)
        or not _is_rfc3339_date_time(operation["when"])
    ):
        raise HistoryFormatError(
            f"recent_operations[{index}] contains invalid strings or timestamp"
        )
    if operation["result"] not in {"ok", "error", "cancelled", "partial"}:
        raise HistoryFormatError(f"recent_operations[{index}].result is invalid")


def _is_rfc3339_date_time(value):
    import re

    match = re.fullmatch(
        r"(\d{4})-(\d{2})-(\d{2})[Tt]"
        r"(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?"
        r"(?:[Zz]|[+-](\d{2}):(\d{2}))",
        value,
    )
    if match is None:
        return False
    year, month, day, hour, minute, second, offset_hour, offset_minute = (
        int(part) if part is not None else None for part in match.groups()
    )
    if (
        year == 0
        or not 1 <= month <= 12
        or hour > 23
        or minute > 59
        or second > 60
        or (offset_hour is not None and offset_hour > 23)
        or (offset_minute is not None and offset_minute > 59)
    ):
        return False
    import calendar

    return 1 <= day <= calendar.monthrange(year, month)[1]


def _load_or_migrate(path: Path):
    """Read current history strictly; migration requires explicit setup."""
    if not path.exists():
        return _empty_history()
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise HistoryFormatError(f"cannot read valid history JSON: {error}") from error
    return _validate_current_history(data)


def migrate_history_to_current(path: Path | None = None):
    """Explicit setup-wizard migration preserving compatible history fields."""
    target = path or _history_path()
    with _history_lock(target):
        if not target.exists():
            migrated = _empty_history()
            _atomic_replace_unlocked(target, json.dumps(migrated, indent=2))
            return migrated
        try:
            data = json.loads(target.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise HistoryFormatError(
                f"cannot migrate malformed history JSON: {error}"
            ) from error
        if not isinstance(data, dict):
            raise HistoryFormatError("cannot migrate non-object history")
        if data.get("parity_version") == CURRENT_PARITY_VERSION:
            return _validate_current_history(data)
        recent_paths = data.get("recent_paths")
        if not isinstance(recent_paths, dict):
            recent_paths = {
                "inputs": data.get("recent_inputs", []),
                "outputs": data.get("recent_outputs", []),
            }
        recent_operations = []
        for index, operation in enumerate(data.get("recent_operations", [])):
            try:
                _validate_history_operation(operation, index)
            except HistoryFormatError as error:
                raise HistoryFormatError(
                    f"cannot migrate invalid recent_operations[{index}]: {error}; "
                    f"repair or remove {target} and rerun setup"
                ) from error
            recent_operations.append(operation)
        migrated = {
            "schema_version": CURRENT_HISTORY_SCHEMA,
            "parity_version": CURRENT_PARITY_VERSION,
            "recent_operations": recent_operations[:50],
            "recent_paths": {
                "inputs": list(recent_paths.get("inputs", []))[:20],
                "outputs": list(recent_paths.get("outputs", []))[:20],
            },
            "interrupted": data.get("interrupted", False),
            "last_op": data.get("last_op"),
            "last_input": data.get("last_input"),
            "last_output": data.get("last_output"),
        }
        _validate_current_history(migrated)
        _atomic_replace_unlocked(target, json.dumps(migrated, indent=2))
        return migrated


@contextmanager
def _history_lock(path: Path):
    """Hold the shared history lock for a complete read/merge/write transaction.

    POSIX uses ``flock``. Windows calls ``LockFileEx`` on byte zero of the
    sibling lock file so Python and Rust can share one blocking lock protocol.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".json.lock")
    lock_fh = open(lock_path, "a+b")
    if os.name == "nt":
        import ctypes
        import msvcrt
        from ctypes import wintypes

        class Overlapped(ctypes.Structure):
            _fields_ = [
                ("Internal", ctypes.c_size_t),
                ("InternalHigh", ctypes.c_size_t),
                ("Offset", wintypes.DWORD),
                ("OffsetHigh", wintypes.DWORD),
                ("hEvent", wintypes.HANDLE),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        overlapped = Overlapped()
        handle = msvcrt.get_osfhandle(lock_fh.fileno())
        lock_file_ex = kernel32.LockFileEx
        lock_file_ex.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(Overlapped),
        ]
        lock_file_ex.restype = wintypes.BOOL
        if not lock_file_ex(handle, 0x00000002, 0, 1, 0, ctypes.byref(overlapped)):
            lock_fh.close()
            raise OSError(ctypes.get_last_error(), "LockFileEx history lock failed")
        try:
            yield
        finally:
            unlock_file_ex = kernel32.UnlockFileEx
            unlock_file_ex.argtypes = [
                wintypes.HANDLE,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.DWORD,
                ctypes.POINTER(Overlapped),
            ]
            unlock_file_ex.restype = wintypes.BOOL
            if not unlock_file_ex(handle, 0, 1, 0, ctypes.byref(overlapped)):
                error = ctypes.get_last_error()
                lock_fh.close()
                raise OSError(error, "UnlockFileEx history lock failed")
            lock_fh.close()
    else:
        import fcntl

        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            try:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            finally:
                lock_fh.close()


def _atomic_replace_unlocked(path: Path, body: str):
    """Write and fsync one unique same-directory temporary, then replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary)
    replaced = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        replaced = True
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if not replaced:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _atomic_write_locked(path: Path, body: str):
    with _history_lock(path):
        _atomic_replace_unlocked(path, body)


def _mutate_history(path: Path, mutate):
    with _history_lock(path):
        history = _load_or_migrate(path)
        mutate(history)
        _validate_current_history(history)
        _atomic_replace_unlocked(path, json.dumps(history, indent=2))


def load_history():
    """Read the spec-format history JSON. Returns the canonical shape."""
    path = _history_path()
    return _load_or_migrate(path)


def update_history(action: str, target: str, result: str):
    """Append an op to the rolling 50-entry log, atomically + locked."""
    path = _history_path()
    try:
        def mutate(history):
            history["recent_operations"].insert(0, {
                "action": action, "target": target,
                "when": datetime.now(timezone.utc).isoformat(),
                "result": result,
            })
            history["recent_operations"] = history["recent_operations"][:50]

        _mutate_history(path, mutate)
    except Exception as e:
        print(f"warning: could not update history: {e}", file=sys.stderr)


def add_recent_path(kind: str, path: str):
    """Push ``path`` to the front of the appropriate recent list."""
    if kind not in ("inputs", "outputs", "input", "output"):
        return
    if kind in ("input", "output"):
        kind = kind + "s"
    target_path = _history_path()
    try:
        def mutate(history):
            paths = history["recent_paths"][kind]
            if path in paths:
                paths.remove(path)
            paths.insert(0, path)
            history["recent_paths"][kind] = paths[:20]

        _mutate_history(target_path, mutate)
    except Exception as e:
        print(f"warning: could not save recent path: {e}", file=sys.stderr)


# ────────────────────────────────────────────────────────────────────
# State detection
# ────────────────────────────────────────────────────────────────────

def find_interrupted_runs():
    candidates = []
    history = load_history()
    for path in history.get("recent_paths", {}).get("outputs", [])[:10]:
        sf = Path(path) / ".lamquant_state.json"
        if sf.exists():
            try:
                state = json.loads(sf.read_text())
                stats = state.get("statistics_so_far", {})
                if stats.get("files_remaining", 0) > 0:
                    candidates.append((path, state))
            except Exception:
                pass
    if Path(".lamquant_state.json").exists():
        try:
            state = json.loads(Path(".lamquant_state.json").read_text())
            candidates.append((str(Path.cwd()), state))
        except Exception:
            pass
    return candidates

def config_status():
    try:
        from lamquant_codec.cli.config import _find_config_file
        cf = _find_config_file()
        return str(cf) if cf else None
    except Exception:
        return None


# ────────────────────────────────────────────────────────────────────
# Error recovery
# ────────────────────────────────────────────────────────────────────

def save_crash_report(exc):
    crash_dir = Path(tempfile.gettempdir())
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = crash_dir / f"lamquant_crash_{ts}.txt"
    try:
        path.write_text(traceback.format_exc())
    except Exception:
        pass
    return str(path)

def run_safely(fn, args=None):
    try:
        return fn(args or [])
    except KeyboardInterrupt:
        print(f"\n  {YEL}Cancelled.{RST}")
        return None
    except SystemExit:
        raise  # let x/exit propagate — don't swallow it
    except Exception as e:
        crash = save_crash_report(e)
        print(f"\n  {RED}Error: {e}{RST}")
        print(f"  Crash report: {DIM}{crash}{RST}")
        return None


# ────────────────────────────────────────────────────────────────────
# Input matching
# ────────────────────────────────────────────────────────────────────

def match_input(text, options_map):
    text = text.strip().lower()
    if not text:
        return None
    if text == "x":
        return "__exit__"
    if text in ("q", "quit"):
        return "__quit__"
    if text in ("b", "back"):
        return "__back__"
    if text in ("?", "h", "help"):
        return "__help__"
    if text.startswith("!"):
        # Shell escape disabled for safety (FDA clinical tool).
        # os.system(text[1:]) was command injection from user input.
        print(f"  {DIM}Shell escape disabled.{RST}")
        return "__shell__"
    if text in options_map:
        return text
    for key, label in options_map.items():
        if label.lower().startswith(text):
            return key
    return None
