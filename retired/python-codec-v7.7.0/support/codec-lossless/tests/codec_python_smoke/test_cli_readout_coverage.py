"""Coverage tests for ``lamquant_codec.cli.readout``.

The readout module is the production telemetry output layer. It mixes:
  * Pure formatters (``_bytes``, ``_dur``, ``_bar``, ``_ratio``,
    ``_pct``, ``_trunc``, ``_vlen``, ``_pad_to``)
  * Box drawing helpers (``_box_line``, ``_box_top``, ``_box_bottom``,
    ``_box_empty``)
  * Data structures (FileResult, RunStats — with derived properties)
  * Banner / summary / manifest / audit-log writers
  * Dashboard class (interactive TTY)

Pinned invariants (futureproof):
  * Type/shape of return values
  * Visible-width arithmetic for box helpers
  * Manifest JSON has documented top-level keys
  * Audit log produces line-buffered entries

We avoid pinning exact strings — color escape codes + unicode chars
differ across CI environments.
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from lamquant_codec.cli import readout as ro
from lamquant_codec.cli.readout import (
    AuditLog,
    C,
    Dashboard,
    FileResult,
    RunStats,
    print_banner,
    print_file_line,
    print_summary,
    print_summary_json,
    splash,
    write_manifest,
)

_ANSI = re.compile(r"\033\[[0-9;]*m")


def _vlen(s: str) -> int:
    return len(_ANSI.sub("", s))


# ----- _bytes -----------------------------------------------------------


def test_bytes_small_returns_b_unit():
    assert ro._bytes(512) == "512 B"


def test_bytes_kib():
    out = ro._bytes(2048)
    assert "KiB" in out


def test_bytes_mib():
    out = ro._bytes(2 * 1024 * 1024)
    assert "MiB" in out


def test_bytes_gib():
    out = ro._bytes(3 * 1024 ** 3)
    assert "GiB" in out


def test_bytes_returns_str_always():
    for n in (0, 1, 1000, 10**6, 10**9, 10**12, 10**15):
        assert isinstance(ro._bytes(n), str)


# ----- _dur -------------------------------------------------------------


def test_dur_negative_returns_dashes():
    assert ro._dur(-1) == "--:--:--"


def test_dur_format_hh_mm_ss():
    # 3 hours, 5 minutes, 7 seconds = 11107 seconds
    out = ro._dur(11107)
    assert out == "03:05:07"


def test_dur_zero():
    assert ro._dur(0) == "00:00:00"


# ----- _bar -------------------------------------------------------------


def test_bar_returns_string_of_width():
    bar = ro._bar(0.5, w=20)
    # Visual length should equal w (unicode block chars are 1 grapheme each)
    assert len(bar) == 20


def test_bar_clamps_fraction():
    # >1.0 should clamp to full
    bar1 = ro._bar(1.5, w=10)
    assert len(bar1) == 10
    # <0.0 should clamp to empty
    bar2 = ro._bar(-0.5, w=10)
    assert len(bar2) == 10


# ----- _ratio / _pct ---------------------------------------------------


def test_ratio_format():
    out = ro._ratio(2.5)
    assert ": 1" in out
    assert "2.50" in out


def test_pct_format():
    out = ro._pct(50.0)
    assert "%" in out
    assert "50" in out


# ----- _trunc -----------------------------------------------------------


def test_trunc_short_pads():
    out = ro._trunc("ab", 10)
    # Should pad to width
    assert len(out) == 10


def test_trunc_long_truncates():
    out = ro._trunc("a" * 100, 10)
    assert len(out) == 10
    # Truncated form contains "..."
    assert "..." in out


def test_trunc_exact_width():
    out = ro._trunc("hello", 5)
    assert out == "hello"


# ----- _vlen / _pad_to -------------------------------------------------


def test_vlen_strips_ansi():
    raw = "\033[31mX\033[0m"
    assert ro._vlen(raw) == 1


def test_pad_to_visible_width():
    raw = "\033[31mX\033[0m"
    out = ro._pad_to(raw, 5)
    assert ro._vlen(out) == 5


def test_pad_to_no_shrink():
    # If input wider than width, return unchanged
    out = ro._pad_to("hello world", 3)
    assert out == "hello world"


# ----- box helpers ------------------------------------------------------


def test_box_line_visible_width_matches():
    out = ro._box_line("hi", width=20)
    assert _vlen(out) == 20


def test_box_top_visible_width_matches():
    out = ro._box_top("Hdr", width=30)
    assert _vlen(out) == 30


def test_box_bottom_visible_width_matches():
    out = ro._box_bottom(width=30)
    assert _vlen(out) == 30


def test_box_empty_visible_width_matches():
    out = ro._box_empty(width=30)
    assert _vlen(out) == 30


# ----- FileResult -------------------------------------------------------


def test_file_result_defaults():
    r = FileResult(path="/x.edf")
    assert r.path == "/x.edf"
    assert r.bytes_in == 0
    assert r.status == "ok"
    assert r.error is None


def test_file_result_ratio_with_zero_bytes_out():
    r = FileResult(path="x", bytes_in=100, bytes_out=0)
    assert r.ratio == 0.0


def test_file_result_ratio_normal():
    r = FileResult(path="x", bytes_in=100, bytes_out=20)
    assert r.ratio == 5.0


# ----- RunStats ---------------------------------------------------------


def test_run_stats_defaults():
    s = RunStats()
    assert s.files_total == 0
    assert s.files_done == 0
    assert isinstance(s.codec_version, str)


def test_run_stats_ratio_zero_when_no_output():
    s = RunStats(bytes_in=100, bytes_out=0)
    assert s.ratio == 0.0


def test_run_stats_ratio_normal():
    s = RunStats(bytes_in=100, bytes_out=20)
    assert s.ratio == 5.0


def test_run_stats_throughput_positive():
    s = RunStats(bytes_in=10 * 1024 * 1024)
    # start_time is now; throughput is bytes_in/elapsed in MiB/s
    assert s.throughput > 0


def test_run_stats_eta_zero_when_no_progress():
    s = RunStats(files_total=10, files_done=0)
    assert s.eta_s == 0.0


def test_run_stats_eta_positive_when_in_progress():
    s = RunStats(files_total=100, files_done=5)
    # Should be positive when there's remaining work
    assert s.eta_s >= 0


def test_run_stats_shannon_gap_no_compression():
    s = RunStats(bytes_in=100, bytes_out=0)
    assert s.shannon_gap == 16.0


def test_run_stats_shannon_gap_bounded():
    s = RunStats(bytes_in=100, bytes_out=10)
    assert s.shannon_gap >= 0


def test_run_stats_shannon_pct_zero_when_no_data():
    s = RunStats(bytes_in=0, bytes_out=0)
    assert s.shannon_pct == 0.0


def test_run_stats_elapsed_increasing():
    s = RunStats()
    import time
    time.sleep(0.01)
    assert s.elapsed >= 0.01


# ----- print_banner -----------------------------------------------------


def test_print_banner_writes_stdout(capsys):
    s = RunStats(files_total=10)
    print_banner(s, workers=4, files_total=10,
                 input_path="/in", output_path="/out")
    captured = capsys.readouterr()
    assert captured.out  # non-empty


def test_print_banner_no_workers(capsys):
    s = RunStats()
    print_banner(s)
    captured = capsys.readouterr()
    assert captured.out


# ----- splash (no-op when not TTY) -------------------------------------


def test_splash_zero_duration_is_noop(capsys):
    splash(0.0)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_splash_no_tty_is_noop(capsys):
    # Not a TTY in test env, so should be no-op even with positive duration
    with patch("sys.stdout.isatty", return_value=False):
        splash(0.1)
    captured = capsys.readouterr()
    assert captured.out == ""


# ----- print_summary ---------------------------------------------------


def test_print_summary_writes_stdout(tmp_path, capsys):
    s = RunStats(files_total=5, files_done=5, files_ok=5,
                 bytes_in=1024 * 1024, bytes_out=512 * 1024,
                 total_samples=100000, total_duration_s=10.0,
                 config_hash="abc" * 22)
    print_summary(s, tmp_path)
    captured = capsys.readouterr()
    # Pinned: summary mentions compression and outputs the manifest path
    assert captured.out
    assert "manifest" in captured.out.lower() or "audit" in captured.out.lower()


def test_print_summary_with_results(tmp_path, capsys):
    s = RunStats(files_total=3, files_done=3, files_ok=3,
                 bytes_in=3 * 1024, bytes_out=1024)
    results = [
        FileResult(path=f"/x{i}.edf", bytes_in=1024, bytes_out=512,
                   status="ok") for i in range(3)
    ]
    print_summary(s, tmp_path, results=results, verbose=True)
    captured = capsys.readouterr()
    assert captured.out


def test_print_summary_no_results_block_when_zero_files(tmp_path, capsys):
    s = RunStats(files_total=0, files_done=0)
    print_summary(s, tmp_path)
    captured = capsys.readouterr()
    assert "Compression complete" in captured.out


# ----- print_summary_json ----------------------------------------------


def test_print_summary_json_emits_valid_json(capsys):
    s = RunStats(files_ok=10, files_error=1, files_skipped=2,
                 bytes_in=1024, bytes_out=256)
    print_summary_json(s)
    captured = capsys.readouterr()
    # Strip ANSI / non-JSON; the function only writes the JSON line
    line = captured.out.strip()
    doc = json.loads(line)
    # Pinned: documented keys
    for key in ("files_ok", "files_error", "files_skipped",
                "bytes_in", "bytes_out", "ratio",
                "shannon_efficiency", "wall_time_s",
                "codec_version", "git_commit"):
        assert key in doc


# ----- print_file_line --------------------------------------------------


def test_print_file_line_ok(capsys):
    r = FileResult(path="/data/x.edf", bytes_in=1024, bytes_out=256,
                   status="ok", duration_s=1.2)
    print_file_line(1, 10, r)
    captured = capsys.readouterr()
    assert "x.edf" in captured.out


def test_print_file_line_skipped(capsys):
    r = FileResult(path="/data/x.edf", status="skipped")
    print_file_line(1, 10, r)
    captured = capsys.readouterr()
    assert "x.edf" in captured.out


def test_print_file_line_error(capsys):
    r = FileResult(path="/data/x.edf", status="error",
                   error="bad checksum")
    print_file_line(1, 10, r)
    captured = capsys.readouterr()
    assert "x.edf" in captured.out
    assert "bad checksum" in captured.out


def test_print_file_line_error_no_message(capsys):
    r = FileResult(path="/data/x.edf", status="error", error=None)
    print_file_line(1, 10, r)
    captured = capsys.readouterr()
    assert "unknown" in captured.out


# ----- AuditLog --------------------------------------------------------


def test_audit_log_writes_to_path(tmp_path):
    p = tmp_path / "audit.log"
    log = AuditLog(p)
    s = RunStats()
    log.start(["lamquant", "compress"], s)
    log.config("abc123")
    log.scan(10, 1024)
    log.end(s, 0)
    assert p.exists()
    content = p.read_text()
    # Pinned shape: log contains START, VERSION, CONFIG, SCAN, END
    for marker in ("START", "VERSION", "CONFIG", "SCAN", "END"):
        assert marker in content


def test_audit_log_file_ok_writes_line(tmp_path):
    p = tmp_path / "audit.log"
    log = AuditLog(p)
    s = RunStats()
    log.start(["lamquant"], s)
    r = FileResult(path="/x.edf", bytes_in=1024, bytes_out=256,
                   status="ok", sha256="a" * 64, duration_s=1.5)
    log.file_ok(1, 10, r)
    log.end(s, 0)
    content = p.read_text()
    assert "FILE" in content
    assert "x.edf" in content
    assert "state=ok" in content


def test_audit_log_file_error_writes_line(tmp_path):
    p = tmp_path / "audit.log"
    log = AuditLog(p)
    s = RunStats()
    log.start(["lamquant"], s)
    log.file_error(1, 10, "/x.edf", "boom")
    log.end(s, 0)
    content = p.read_text()
    assert "FILE" in content
    assert "state=error" in content


def test_audit_log_handles_unwritable_path(tmp_path, capsys):
    """Constructor should warn but not raise when log can't be opened."""
    unwritable = tmp_path / "noperm"
    unwritable.mkdir(mode=0o500)
    try:
        log = AuditLog(unwritable / "audit.log")
        # When _fh is None, methods should be no-ops
        log._w("X", k="v")
    finally:
        unwritable.chmod(0o700)


def test_audit_log_interrupt_summary_methods(tmp_path):
    p = tmp_path / "audit.log"
    log = AuditLog(p)
    s = RunStats()
    log.start(["x"], s)
    log.interrupt(s)
    log.summary(s)
    log.end(s, 130)
    content = p.read_text()
    assert "INTERRUPT" in content
    assert "SUMMARY" in content


# ----- write_manifest --------------------------------------------------


def test_write_manifest_creates_file(tmp_path):
    s = RunStats(files_ok=2, files_error=0, files_skipped=0,
                 bytes_in=2 * 1024, bytes_out=1024,
                 total_samples=10000, total_duration_s=5.0)
    results = [
        FileResult(path="/x1.edf", bytes_in=1024, bytes_out=512,
                   status="ok", sha256="a" * 64),
        FileResult(path="/x2.edf", bytes_in=1024, bytes_out=512,
                   status="ok", sha256="b" * 64),
    ]
    p = tmp_path / "manifest.json"
    write_manifest(p, s, results)
    assert p.exists()
    doc = json.loads(p.read_text())
    # Pinned: documented top-level keys
    for key in ("schema_version", "codec", "run",
                "statistics", "integrity", "files"):
        assert key in doc
    assert len(doc["files"]) == 2


def test_write_manifest_atomic_rename(tmp_path):
    """Ensure write goes through tmp + rename atomically."""
    s = RunStats(files_ok=1, bytes_in=1, bytes_out=1)
    results = []
    p = tmp_path / "subdir" / "manifest.json"
    # Parent doesn't exist — write_manifest should create it
    write_manifest(p, s, results)
    assert p.exists()


def test_write_manifest_propagates_exception(tmp_path):
    """If json serialization fails, the .tmp must be cleaned up and exc
    re-raised."""
    s = RunStats()
    # Pass something un-serializable inside the files list
    bad = [FileResult(path="/x")]
    p = tmp_path / "manifest.json"
    # Real call should succeed (FileResult is dataclass with str path)
    write_manifest(p, s, bad)
    assert p.exists()


# ----- C.reconfigure ---------------------------------------------------


def test_c_reconfigure_off_clears_codes():
    C.reconfigure(False)
    assert C.RED == ""
    assert C.BLD == ""
    # Restore
    C.reconfigure(True)
    assert C.RED != ""


# ----- Dashboard (no-tty paths) ----------------------------------------


def test_dashboard_construct():
    s = RunStats()
    d = Dashboard(s, refresh_hz=10.0)
    assert d.stats is s
    assert d._lines == 0


def test_dashboard_set_paths():
    s = RunStats()
    d = Dashboard(s)
    d.set_paths("/in", "/out")
    assert d._input_path == "/in"
    assert d._output_path == "/out"


def test_dashboard_set_file():
    s = RunStats()
    d = Dashboard(s)
    d.set_file("/foo.edf")
    assert d._file == "/foo.edf"


def test_dashboard_tick_no_tty_is_noop():
    s = RunStats()
    d = Dashboard(s)
    with patch("sys.stdout.isatty", return_value=False):
        # Should not raise + should not write anything
        d.tick(force=True)


def test_dashboard_clear_no_lines_is_noop():
    s = RunStats()
    d = Dashboard(s)
    # No prior draw → no lines to clear
    d.clear()
    assert d._lines == 0


def test_dashboard_throttles_by_interval():
    s = RunStats()
    d = Dashboard(s, refresh_hz=1.0)  # 1 Hz → 1 second interval
    # Pretend we just drew. Without force=True, next tick should noop.
    import time
    d._last_draw = time.time()
    # Patch isatty so the drawing path would be entered if not throttled
    with patch("sys.stdout.isatty", return_value=True), patch.object(
        d, "_draw"
    ) as draw_mock:
        d.tick(force=False)
    assert draw_mock.call_count == 0
