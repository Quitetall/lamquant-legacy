"""Coverage tests for ``lamquant_codec.cli.state``.

The state module tracks per-file progress to support crash recovery.
Pure helpers exercised:

  * StateFile construction, atomic save/load round-trip
  * State transitions: register, mark_in_progress, mark_completed,
    mark_failed, quarantine
  * recover_zombies (PID liveness check)
  * recovery_summary counters
  * print_recovery_summary writes to stdout
  * load() handles missing files + corrupted JSON

We pin status invariants (counters add to total, statuses are members
of the documented set) rather than exact bytes-on-disk.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from lamquant_codec.cli import state as state_mod
from lamquant_codec.cli.state import (
    COMPLETED,
    FAILED,
    IN_PROGRESS,
    PENDING,
    QUARANTINED,
    FileState,
    StateFile,
    print_recovery_summary,
)


# ----- module-level constants -------------------------------------------


def test_status_constants_are_strings_and_distinct():
    statuses = {PENDING, IN_PROGRESS, COMPLETED, FAILED, QUARANTINED}
    assert all(isinstance(s, str) for s in statuses)
    assert len(statuses) == 5


# ----- FileState --------------------------------------------------------


def test_file_state_default_values():
    fs = FileState()
    assert fs.status == PENDING
    assert fs.bytes_in == 0
    assert fs.bytes_out == 0
    assert fs.sha256 == ""
    assert fs.attempts == 0


def test_file_state_round_trip_attributes():
    fs = FileState(status=COMPLETED, bytes_in=100, sha256="abc")
    assert fs.status == COMPLETED
    assert fs.bytes_in == 100
    assert fs.sha256 == "abc"


# ----- StateFile construction -------------------------------------------


def test_statefile_construct(tmp_path):
    sf = StateFile(tmp_path, "/input/root", "abc", ["foo"])
    assert sf.input_root == "/input/root"
    assert sf.config_hash == "abc"
    assert sf.cli_args == ["foo"]
    assert isinstance(sf.run_id, str)
    assert len(sf.run_id) == 8
    assert sf.output_dir == tmp_path
    assert sf.files == {}
    assert sf.total_discovered == 0


def test_statefile_exists_false_on_empty_dir(tmp_path):
    sf = StateFile(tmp_path, "/in", "h", [])
    assert sf.exists() is False


# ----- register_files + transitions -------------------------------------


def test_register_files_creates_pending_entries(tmp_path):
    sf = StateFile(tmp_path, "/in", "h", [])
    sf.register_files(["a.edf", "b.edf"])
    assert set(sf.files.keys()) == {"a.edf", "b.edf"}
    assert all(fs.status == PENDING for fs in sf.files.values())
    assert sf.total_discovered == 2


def test_register_files_idempotent(tmp_path):
    sf = StateFile(tmp_path, "/in", "h", [])
    sf.register_files(["a.edf"])
    sf.register_files(["a.edf", "b.edf"])
    assert set(sf.files.keys()) == {"a.edf", "b.edf"}


def test_mark_in_progress_sets_status(tmp_path):
    sf = StateFile(tmp_path, "/in", "h", [])
    sf.register_files(["a.edf"])
    sf.mark_in_progress("a.edf", pid=12345)
    fs = sf.files["a.edf"]
    assert fs.status == IN_PROGRESS
    assert fs.worker_pid == 12345
    assert fs.attempts == 1


def test_mark_in_progress_uses_default_pid(tmp_path):
    sf = StateFile(tmp_path, "/in", "h", [])
    sf.mark_in_progress("x.edf")
    assert sf.files["x.edf"].worker_pid == os.getpid()


def test_mark_completed_sets_fields(tmp_path):
    sf = StateFile(tmp_path, "/in", "h", [])
    sf.mark_completed("a.edf", "/out/a.lml", 100, 30, "abc123")
    fs = sf.files["a.edf"]
    assert fs.status == COMPLETED
    assert fs.bytes_in == 100
    assert fs.bytes_out == 30
    assert fs.sha256 == "abc123"
    assert fs.output_path == "/out/a.lml"
    assert fs.worker_pid == 0
    assert fs.completed_at != ""


def test_mark_failed_truncates_error(tmp_path):
    sf = StateFile(tmp_path, "/in", "h", [])
    long_err = "X" * 500
    sf.mark_failed("a.edf", long_err)
    fs = sf.files["a.edf"]
    assert fs.status == FAILED
    # Pinned invariant: error is truncated to <=200 chars
    assert len(fs.last_error) <= 200


def test_quarantine_sets_status_and_dir(tmp_path):
    sf = StateFile(tmp_path, "/in", "h", [])
    sf.register_files(["a.edf"])
    sf.quarantine("a.edf", "/quarantine/")
    fs = sf.files["a.edf"]
    assert fs.status == QUARANTINED
    assert fs.quarantined_to == "/quarantine/"


def test_quarantine_unknown_file_is_noop(tmp_path):
    sf = StateFile(tmp_path, "/in", "h", [])
    # should not raise
    sf.quarantine("not-tracked.edf", "/q")
    assert "not-tracked.edf" not in sf.files


# ----- should_process / is_completed -----------------------------------


def test_should_process_unknown_file_returns_true(tmp_path):
    sf = StateFile(tmp_path, "/in", "h", [])
    assert sf.should_process("never-seen.edf") is True


def test_should_process_pending_returns_true(tmp_path):
    sf = StateFile(tmp_path, "/in", "h", [])
    sf.register_files(["a.edf"])
    assert sf.should_process("a.edf") is True


def test_should_process_completed_returns_false(tmp_path):
    sf = StateFile(tmp_path, "/in", "h", [])
    sf.mark_completed("a.edf", "/out", 10, 5, "h")
    assert sf.should_process("a.edf") is False


def test_should_process_failed_returns_true_retryable(tmp_path):
    sf = StateFile(tmp_path, "/in", "h", [])
    sf.mark_failed("a.edf", "boom")
    assert sf.should_process("a.edf") is True


def test_should_process_in_progress_returns_false(tmp_path):
    sf = StateFile(tmp_path, "/in", "h", [])
    sf.mark_in_progress("a.edf")
    assert sf.should_process("a.edf") is False


def test_is_completed(tmp_path):
    sf = StateFile(tmp_path, "/in", "h", [])
    assert sf.is_completed("nope.edf") is False
    sf.mark_completed("a.edf", "/out", 1, 1, "h")
    assert sf.is_completed("a.edf") is True


# ----- flush / load round-trip -----------------------------------------


def test_flush_writes_state_file(tmp_path):
    sf = StateFile(tmp_path, "/in", "abc", ["lamquant"])
    sf.register_files(["a.edf"])
    sf.mark_completed("a.edf", "/out/a.lml", 1024, 512, "deadbeef")
    sf.flush()
    assert sf.path.exists()
    # JSON is valid
    data = json.loads(sf.path.read_text())
    assert data["schema_version"] == "1.0"
    assert "files" in data
    assert "a.edf" in data["files"]


def test_flush_skips_when_not_dirty(tmp_path):
    sf = StateFile(tmp_path, "/in", "h", [])
    sf.flush()
    # Without any registers, file should NOT exist
    assert not sf.path.exists()


def test_load_after_flush_recovers_state(tmp_path):
    sf1 = StateFile(tmp_path, "/in", "h", ["x"])
    sf1.register_files(["a.edf", "b.edf"])
    sf1.mark_completed("a.edf", "/o", 100, 50, "h1")
    sf1.flush()

    sf2 = StateFile(tmp_path, "/in", "h", ["x"])
    ok = sf2.load()
    assert ok is True
    assert "a.edf" in sf2.files
    assert sf2.files["a.edf"].status == COMPLETED
    assert sf2.files["a.edf"].bytes_in == 100


def test_load_returns_false_when_no_file(tmp_path):
    sf = StateFile(tmp_path, "/in", "h", [])
    assert sf.load() is False


def test_load_returns_false_when_input_root_differs(tmp_path):
    sf1 = StateFile(tmp_path, "/in", "h", [])
    sf1.register_files(["a"])
    sf1.flush()

    sf2 = StateFile(tmp_path, "/DIFFERENT", "h", [])
    assert sf2.load() is False


def test_load_handles_corrupted_json(tmp_path):
    sf1 = StateFile(tmp_path, "/in", "h", [])
    sf1.path.write_text("{not valid json")
    sf = StateFile(tmp_path, "/in", "h", [])
    # No prev file either → returns False
    assert sf.load() is False


def test_load_falls_back_to_prev_when_main_corrupted(tmp_path):
    # Write a clean snapshot first to populate .prev.
    sf1 = StateFile(tmp_path, "/in", "h", [])
    sf1.register_files(["a.edf"])
    sf1.flush()
    # Second flush will copy current → prev, then overwrite current
    sf1.mark_completed("a.edf", "/o", 1, 1, "x")
    sf1.flush()
    # Corrupt the main file
    sf1.path.write_text("{not valid")
    sf2 = StateFile(tmp_path, "/in", "h", [])
    ok = sf2.load()
    # Should successfully fall back to prev
    assert ok is True
    assert "a.edf" in sf2.files


# ----- recover_zombies --------------------------------------------------


def test_recover_zombies_resets_dead_workers(tmp_path):
    sf = StateFile(tmp_path, "/in", "h", [])
    sf.register_files(["a.edf"])
    # Use a PID that almost certainly doesn't exist
    sf.mark_in_progress("a.edf", pid=999999999)
    with patch("lamquant_codec.cli.state.os.kill",
               side_effect=ProcessLookupError()):
        n = sf.recover_zombies()
    assert n == 1
    assert sf.files["a.edf"].status == PENDING
    assert sf.files["a.edf"].worker_pid == 0


def test_recover_zombies_keeps_live_workers(tmp_path):
    sf = StateFile(tmp_path, "/in", "h", [])
    sf.mark_in_progress("a.edf", pid=12345)
    with patch("lamquant_codec.cli.state.os.kill", return_value=None):
        n = sf.recover_zombies()
    # Live worker → no recovery
    assert n == 0
    assert sf.files["a.edf"].status == IN_PROGRESS


def test_recover_zombies_no_pid_treated_as_dead(tmp_path):
    sf = StateFile(tmp_path, "/in", "h", [])
    sf.files["a.edf"] = FileState(status=IN_PROGRESS, worker_pid=0)
    sf._dirty = True
    n = sf.recover_zombies()
    assert n == 1
    assert sf.files["a.edf"].status == PENDING


# ----- recovery_summary -------------------------------------------------


def test_recovery_summary_returns_dict_with_all_keys(tmp_path):
    sf = StateFile(tmp_path, "/in", "h", [])
    sf.register_files(["a.edf", "b.edf"])
    sf.mark_completed("a.edf", "/o", 1, 1, "h")
    summary = sf.recovery_summary()
    # The five canonical statuses should be present
    for key in (PENDING, IN_PROGRESS, COMPLETED, FAILED, QUARANTINED):
        assert key in summary
    assert summary[COMPLETED] == 1
    assert summary[PENDING] == 1


def test_recovery_summary_sums_to_file_count(tmp_path):
    sf = StateFile(tmp_path, "/in", "h", [])
    sf.register_files(["a", "b", "c"])
    sf.mark_completed("a", "/o", 1, 1, "h")
    sf.mark_failed("b", "err")
    summary = sf.recovery_summary()
    assert sum(summary.values()) == 3


# ----- print_recovery_summary ------------------------------------------


def test_print_recovery_summary_writes_to_stdout(tmp_path, capsys):
    sf = StateFile(tmp_path, "/in", "h", [])
    sf.register_files(["a.edf", "b.edf"])
    sf.mark_completed("a.edf", "/o", 1, 1, "h")
    sf.total_discovered = 2
    print_recovery_summary(sf, zombies=0)
    captured = capsys.readouterr()
    assert "Recovery" in captured.out
    assert sf.run_id in captured.out


def test_print_recovery_summary_with_zombies(tmp_path, capsys):
    sf = StateFile(tmp_path, "/in", "h", [])
    sf.register_files(["a.edf"])
    sf.mark_failed("a.edf", "boom")
    print_recovery_summary(sf, zombies=3)
    captured = capsys.readouterr()
    assert "Zombie" in captured.out
    assert "Failed" in captured.out
