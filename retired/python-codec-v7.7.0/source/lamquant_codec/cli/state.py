"""
LamQuant state file — resumability and crash recovery.

Contract: if LamQuant starts, it either completes the work, or it knows
exactly what it didn't complete and can pick up from there.

State file: {output_dir}/.lamquant_state.json
Writes are atomic (temp + rename). Survives crashes, OOM, SIGKILL.
"""
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


# File statuses — strict transitions only:
#   pending → in_progress → completed
#   pending → in_progress → failed → in_progress (retry)
#   failed → quarantined (after max_retries)
PENDING = "pending"
IN_PROGRESS = "in_progress"
COMPLETED = "completed"
FAILED = "failed"
QUARANTINED = "quarantined"


@dataclass
class FileState:
    status: str = PENDING
    output_path: str = ""
    bytes_in: int = 0
    bytes_out: int = 0
    sha256: str = ""
    completed_at: str = ""
    worker_pid: int = 0
    attempts: int = 0
    last_error: str = ""
    quarantined_to: str = ""


class StateFile:
    """Atomic, crash-safe state tracking for resumable runs."""

    def __init__(self, output_dir: Path, input_root: str, config_hash: str,
                 cli_args: list):
        self.path = output_dir / ".lamquant_state.json"
        self.prev_path = output_dir / ".lamquant_state.json.prev"
        self.output_dir = output_dir

        self.run_id = str(uuid.uuid4())[:8]
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.input_root = input_root
        self.config_hash = config_hash
        self.cli_args = cli_args
        self.files: Dict[str, FileState] = {}
        self.total_discovered = 0
        self._dirty = False

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> bool:
        """Load existing state. Returns True if recovery is possible."""
        data = None
        for p in (self.path, self.prev_path):
            if not p.exists():
                continue
            try:
                with open(p) as f:
                    data = json.load(f)
                break
            except (json.JSONDecodeError, OSError):
                continue

        if data is None:
            return False

        # Verify it matches current invocation
        if data.get("input_root") != self.input_root:
            return False

        self.run_id = data.get("run_id", self.run_id)
        self.started_at = data.get("started_at", self.started_at)
        self.total_discovered = data.get("total_files_discovered", 0)

        for filename, fdata in data.get("files", {}).items():
            fs = FileState()
            for k, v in fdata.items():
                if hasattr(fs, k):
                    setattr(fs, k, v)
            self.files[filename] = fs

        return True

    def recover_zombies(self) -> int:
        """Reset in_progress files (worker died) back to pending."""
        count = 0
        for fname, fs in self.files.items():
            if fs.status == IN_PROGRESS:
                # Check if PID is alive
                if fs.worker_pid > 0:
                    try:
                        os.kill(fs.worker_pid, 0)
                        continue  # still running
                    except (OSError, ProcessLookupError):
                        pass
                fs.status = PENDING
                fs.worker_pid = 0
                count += 1
                self._dirty = True
        return count

    def recovery_summary(self) -> dict:
        """Get counts for recovery display."""
        counts = {PENDING: 0, IN_PROGRESS: 0, COMPLETED: 0,
                  FAILED: 0, QUARANTINED: 0}
        for fs in self.files.values():
            counts[fs.status] = counts.get(fs.status, 0) + 1
        return counts

    def register_files(self, filenames: list):
        """Register discovered files. Only adds new ones."""
        for fn in filenames:
            if fn not in self.files:
                self.files[fn] = FileState()
        self.total_discovered = len(self.files)
        self._dirty = True

    def mark_in_progress(self, filename: str, pid: int = 0):
        fs = self.files.setdefault(filename, FileState())
        fs.status = IN_PROGRESS
        fs.worker_pid = pid or os.getpid()
        fs.attempts += 1
        self._dirty = True

    def mark_completed(self, filename: str, output_path: str,
                       bytes_in: int, bytes_out: int, sha256: str):
        fs = self.files.setdefault(filename, FileState())
        fs.status = COMPLETED
        fs.output_path = output_path
        fs.bytes_in = bytes_in
        fs.bytes_out = bytes_out
        fs.sha256 = sha256
        fs.completed_at = datetime.now(timezone.utc).isoformat()
        fs.worker_pid = 0
        self._dirty = True

    def mark_failed(self, filename: str, error: str):
        fs = self.files.setdefault(filename, FileState())
        fs.status = FAILED
        fs.last_error = error[:200]
        fs.worker_pid = 0
        self._dirty = True

    def quarantine(self, filename: str, quarantine_dir: str):
        fs = self.files.get(filename)
        if not fs:
            return
        fs.status = QUARANTINED
        fs.quarantined_to = quarantine_dir
        self._dirty = True

    def should_process(self, filename: str) -> bool:
        """True if file needs processing (pending or retryable failed)."""
        fs = self.files.get(filename)
        if fs is None:
            return True
        return fs.status in (PENDING, FAILED)

    def is_completed(self, filename: str) -> bool:
        fs = self.files.get(filename)
        return fs is not None and fs.status == COMPLETED

    def flush(self):
        """Atomic write to disk."""
        if not self._dirty:
            return

        doc = {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "started_at": self.started_at,
            "last_checkpoint": datetime.now(timezone.utc).isoformat(),
            "cli_invocation": self.cli_args,
            "config_hash": self.config_hash,
            "input_root": self.input_root,
            "output_root": str(self.output_dir),
            "total_files_discovered": self.total_discovered,
            "files": {},
            "statistics_so_far": {
                "files_completed": sum(
                    1 for f in self.files.values() if f.status == COMPLETED),
                "files_failed": sum(
                    1 for f in self.files.values() if f.status == FAILED),
                "files_remaining": sum(
                    1 for f in self.files.values()
                    if f.status in (PENDING, IN_PROGRESS)),
            },
        }
        for fname, fs in self.files.items():
            doc["files"][fname] = {
                k: v for k, v in fs.__dict__.items() if v is not None
            }

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Atomic: write temp, rename over current, keep .prev
        try:
            fd, tmp = tempfile.mkstemp(
                dir=self.output_dir, suffix=".tmp",
                prefix=".lamquant_state_")
            with os.fdopen(fd, "w") as f:
                json.dump(doc, f, separators=(",", ":"))

            # Rotate: current → .prev, tmp → current
            if self.path.exists():
                try:
                    shutil.copy2(self.path, self.prev_path)
                except OSError:
                    pass
            os.replace(tmp, self.path)
            self._dirty = False
        except OSError as e:
            # Disk full or permission error — try /tmp as last resort
            try:
                os.unlink(tmp)
            except OSError:
                pass
            backup = Path(tempfile.gettempdir()) / f"lamquant_state_{self.run_id}.json"
            try:
                with open(backup, "w") as f:
                    json.dump(doc, f, separators=(",", ":"))
                print(f"FATAL: Cannot write state to {self.path}: {e}",
                      file=sys.stderr)
                print(f"       Emergency backup: {backup}", file=sys.stderr)
            except OSError:
                pass


def print_recovery_summary(state: StateFile, zombies: int):
    """Print recovery information for the user."""
    from lamquant_codec.cli.terminal import S
    h = S["h"]
    counts = state.recovery_summary()
    print(file=sys.stdout)
    print(f" {h * 72}", file=sys.stdout)
    print(f"   LamQuant found an interrupted run. Recovery in progress.",
          file=sys.stdout)
    print(f" {h * 72}", file=sys.stdout)
    print(file=sys.stdout)
    print(f"   Previous run:     {state.started_at}", file=sys.stdout)
    print(f"   Run ID:           {state.run_id}", file=sys.stdout)
    print(file=sys.stdout)
    print(f"   Already complete: {counts[COMPLETED]:>6,} / "
          f"{state.total_discovered:,} files", file=sys.stdout)
    if zombies:
        print(f"   Zombie workers:   {zombies:>6,} files  "
              f"  → will retry", file=sys.stdout)
    if counts[FAILED]:
        print(f"   Failed previously:{counts[FAILED]:>6,} files  "
              f"  → will retry", file=sys.stdout)
    remaining = counts[PENDING] + counts[FAILED]
    print(f"   Remaining:        {remaining:>6,} files", file=sys.stdout)
    print(file=sys.stdout)
    print(f"   Proceeding with remaining work.", file=sys.stdout)
    print(f" {'─' * 72}", file=sys.stdout)
    print(file=sys.stdout)
    sys.stdout.flush()
