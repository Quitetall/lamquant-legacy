"""Coverage tests for ``lamquant_codec.cli.menu``.

The menu module is split between an interactive REPL (untestable) and
pure helpers (testable). We focus on the pure paths:

  * Terminal capability detection (color/unicode)
  * Version probing (version/git_commit/gen_tag/cli_version)
  * History I/O: load_history, update_history, add_recent_path
  * Path resolution: _history_dir, _history_path
  * Schema migration: _empty_history, _load_or_migrate
  * Input matching: match_input
  * State detection: find_interrupted_runs, config_status
  * Error recovery: save_crash_report, run_safely

The interactive prompt loop (prompt/prompt_path/prompt_menu/instant_input)
and the run() subprocess wrapper are NOT exercised.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from lamquant_codec.cli import menu


# ----- terminal capability helpers -------------------------------------


def test_supports_color_returns_bool():
    out = menu._supports_color()
    assert isinstance(out, bool)


def test_supports_color_off_when_no_color_env(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert menu._supports_color() is False


def test_supports_unicode_returns_bool():
    out = menu._supports_unicode()
    assert isinstance(out, bool)


def test_clear_noop_when_not_tty(monkeypatch, capsys):
    # When stdout is not a TTY, clear() should do nothing
    with patch("sys.stdout.isatty", return_value=False):
        menu.clear()
        menu.clear(full=True)
    captured = capsys.readouterr()
    # No ANSI escapes written
    assert captured.out == ""


# ----- set_autocomplete / set_instant_nav -----------------------------


def test_set_autocomplete_toggles_module_state():
    menu.set_autocomplete(False)
    assert menu._autocomplete is False
    menu.set_autocomplete(True)
    assert menu._autocomplete is True


def test_set_instant_nav_toggles_module_state():
    menu.set_instant_nav(True)
    assert menu._instant_nav is True
    menu.set_instant_nav(False)
    assert menu._instant_nav is False


# ----- version helpers -------------------------------------------------


def test_version_returns_string():
    out = menu.version()
    assert isinstance(out, str)
    assert out  # non-empty


def test_git_commit_returns_string():
    out = menu.git_commit()
    assert isinstance(out, str)


def test_git_commit_handles_subprocess_failure():
    with patch("lamquant_codec.cli.menu.subprocess.run",
               side_effect=OSError("nope")):
        out = menu.git_commit()
    assert out == "unknown"


def test_gen_tag_format():
    out = menu.gen_tag()
    assert isinstance(out, str)
    assert out.startswith("Gen ") or out.startswith("v")


def test_cli_version_returns_string():
    out = menu.cli_version()
    assert isinstance(out, str)


# ----- history path resolution -----------------------------------------


def test_history_dir_uses_override(monkeypatch, tmp_path):
    override = tmp_path / "custom" / "history.json"
    monkeypatch.setenv("LAMQUANT_HISTORY", str(override))
    out = menu._history_dir()
    assert out == override.parent


def test_history_dir_uses_xdg(monkeypatch, tmp_path):
    monkeypatch.delenv("LAMQUANT_HISTORY", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    out = menu._history_dir()
    assert out == tmp_path / "xdg" / "lamquant"


def test_history_dir_linux_fallback(monkeypatch):
    monkeypatch.delenv("LAMQUANT_HISTORY", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    with patch("sys.platform", "linux"):
        out = menu._history_dir()
    assert isinstance(out, Path)
    # Should land under ~/.config/lamquant
    assert "lamquant" in str(out)


def test_history_path_uses_override(monkeypatch, tmp_path):
    override = tmp_path / "h.json"
    monkeypatch.setenv("LAMQUANT_HISTORY", str(override))
    assert menu._history_path() == override


def test_history_path_default_resolves_under_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("LAMQUANT_HISTORY", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    out = menu._history_path()
    assert out.name == "history.json"


# ----- _empty_history --------------------------------------------------


def test_empty_history_has_documented_keys():
    h = menu._empty_history()
    assert h["schema_version"] == "2.0"
    assert h["parity_version"] == 2
    assert "recent_operations" in h
    assert "recent_paths" in h
    assert "inputs" in h["recent_paths"]
    assert "outputs" in h["recent_paths"]


# ----- _load_or_migrate ------------------------------------------------


def test_load_or_migrate_missing_file_returns_empty(tmp_path):
    h = menu._load_or_migrate(tmp_path / "missing.json")
    assert h == menu._empty_history()


def test_load_or_migrate_existing_spec_format(tmp_path):
    p = tmp_path / "h.json"
    data = menu._empty_history()
    data["recent_operations"] = [{"action": "test"}]
    p.write_text(json.dumps(data))
    out = menu._load_or_migrate(p)
    assert out["recent_operations"][0]["action"] == "test"


def test_load_or_migrate_legacy_format_fails_closed(tmp_path):
    """Legacy history cannot authorize parity-v2 operations."""
    p = tmp_path / "h.json"
    legacy = {
        "schema_version": "0.9",
        "recent_inputs": ["/a"],
        "recent_outputs": ["/b"],
        "recent_operations": [],
    }
    p.write_text(json.dumps(legacy))
    with pytest.raises(menu.HistoryParityError):
        menu._load_or_migrate(p)


def test_load_or_migrate_corrupt_fails_closed(tmp_path):
    p = tmp_path / "h.json"
    p.write_text("{not valid")
    with pytest.raises(menu.HistoryFormatError):
        menu._load_or_migrate(p)


def test_explicit_history_migration_preserves_compatible_fields(tmp_path):
    p = tmp_path / "h.json"
    p.write_text(json.dumps({
        "schema_version": "1.0",
        "recent_inputs": ["/a"],
        "recent_outputs": ["/b"],
        "recent_operations": [{"action": "verify"}],
    }))
    migrated = menu.migrate_history_to_current(p)
    assert migrated["schema_version"] == "2.0"
    assert migrated["parity_version"] == 2
    assert migrated["recent_paths"] == {"inputs": ["/a"], "outputs": ["/b"]}
    assert json.loads(p.read_text())["parity_version"] == 2


# ----- load_history / update_history / add_recent_path ----------------


def test_load_history_returns_dict_shape(monkeypatch, tmp_path):
    monkeypatch.setenv("LAMQUANT_HISTORY", str(tmp_path / "h.json"))
    h = menu.load_history()
    assert isinstance(h, dict)
    assert "recent_paths" in h


def test_update_history_appends_op(monkeypatch, tmp_path):
    monkeypatch.setenv("LAMQUANT_HISTORY", str(tmp_path / "h.json"))
    menu.update_history("compress", "/in.edf", "ok")
    h = menu.load_history()
    assert len(h["recent_operations"]) == 1
    assert h["recent_operations"][0]["action"] == "compress"
    assert h["recent_operations"][0]["target"] == "/in.edf"
    assert h["recent_operations"][0]["result"] == "ok"


def test_update_history_caps_at_50(monkeypatch, tmp_path):
    monkeypatch.setenv("LAMQUANT_HISTORY", str(tmp_path / "h.json"))
    for i in range(60):
        menu.update_history("op", f"target{i}", "ok")
    h = menu.load_history()
    assert len(h["recent_operations"]) == 50


def test_add_recent_path_inputs(monkeypatch, tmp_path):
    monkeypatch.setenv("LAMQUANT_HISTORY", str(tmp_path / "h.json"))
    menu.add_recent_path("inputs", "/a")
    menu.add_recent_path("inputs", "/b")
    h = menu.load_history()
    # Most recent at front
    assert h["recent_paths"]["inputs"][0] == "/b"


def test_add_recent_path_dedup(monkeypatch, tmp_path):
    monkeypatch.setenv("LAMQUANT_HISTORY", str(tmp_path / "h.json"))
    menu.add_recent_path("inputs", "/a")
    menu.add_recent_path("inputs", "/b")
    menu.add_recent_path("inputs", "/a")
    h = menu.load_history()
    inputs = h["recent_paths"]["inputs"]
    # /a should appear exactly once
    assert inputs.count("/a") == 1
    assert inputs[0] == "/a"


def test_add_recent_path_singular_kind_normalized(monkeypatch, tmp_path):
    """The function accepts both 'input' and 'inputs'."""
    monkeypatch.setenv("LAMQUANT_HISTORY", str(tmp_path / "h.json"))
    menu.add_recent_path("input", "/x")
    h = menu.load_history()
    assert "/x" in h["recent_paths"]["inputs"]


def test_add_recent_path_unknown_kind_noop(monkeypatch, tmp_path):
    monkeypatch.setenv("LAMQUANT_HISTORY", str(tmp_path / "h.json"))
    menu.add_recent_path("garbage", "/x")
    h = menu.load_history()
    # Should not crash and history should be empty
    assert h["recent_paths"]["inputs"] == []


def test_add_recent_path_caps_at_20(monkeypatch, tmp_path):
    monkeypatch.setenv("LAMQUANT_HISTORY", str(tmp_path / "h.json"))
    for i in range(30):
        menu.add_recent_path("inputs", f"/p{i}")
    h = menu.load_history()
    assert len(h["recent_paths"]["inputs"]) == 20


# ----- find_interrupted_runs ------------------------------------------


def test_find_interrupted_runs_returns_list(monkeypatch, tmp_path):
    monkeypatch.setenv("LAMQUANT_HISTORY", str(tmp_path / "h.json"))
    # Switch CWD away from any project that might have a state file.
    monkeypatch.chdir(tmp_path)
    out = menu.find_interrupted_runs()
    assert isinstance(out, list)
    assert out == []  # nothing interrupted


def test_find_interrupted_runs_picks_up_output_state(
    monkeypatch, tmp_path
):
    """When a recent output has a .lamquant_state.json with remaining
    files, it should be reported as interrupted."""
    monkeypatch.setenv("LAMQUANT_HISTORY", str(tmp_path / "h.json"))
    monkeypatch.chdir(tmp_path)
    # Set up a fake interrupted run
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    state_file = out_dir / ".lamquant_state.json"
    state_file.write_text(
        json.dumps(
            {
                "statistics_so_far": {"files_remaining": 5},
                "run_id": "abc",
            }
        )
    )
    # Register the path in history
    menu.add_recent_path("outputs", str(out_dir))
    runs = menu.find_interrupted_runs()
    assert any(p == str(out_dir) for p, _ in runs)


# ----- config_status ---------------------------------------------------


def test_config_status_returns_optional_string(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    out = menu.config_status()
    # Either None (no config) or a string path
    assert out is None or isinstance(out, str)


def test_config_status_returns_path_when_present(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "lamquant.toml"
    cfg.write_text("# stub\n")
    out = menu.config_status()
    assert isinstance(out, str)
    assert "lamquant.toml" in out


def test_config_status_handles_exception():
    with patch(
        "lamquant_codec.cli.config._find_config_file",
        side_effect=RuntimeError("boom"),
    ):
        out = menu.config_status()
    assert out is None


# ----- save_crash_report ----------------------------------------------


def test_save_crash_report_writes_path(monkeypatch):
    try:
        raise ValueError("test exception")
    except ValueError as e:
        path = menu.save_crash_report(e)
    assert isinstance(path, str)
    assert Path(path).exists()
    # Cleanup
    try:
        Path(path).unlink()
    except OSError:
        pass


# ----- run_safely -----------------------------------------------------


def test_run_safely_returns_function_result():
    out = menu.run_safely(lambda args: 42)
    assert out == 42


def test_run_safely_catches_keyboardinterrupt(capsys):
    def raises():
        raise KeyboardInterrupt
    out = menu.run_safely(lambda args: raises())
    assert out is None
    captured = capsys.readouterr()
    assert "Cancelled" in captured.out


def test_run_safely_catches_exception_and_returns_none(capsys):
    def raises():
        raise RuntimeError("boom")
    out = menu.run_safely(lambda args: raises())
    assert out is None
    captured = capsys.readouterr()
    assert "Error" in captured.out


def test_run_safely_propagates_systemexit():
    def raises():
        raise SystemExit(7)
    with pytest.raises(SystemExit) as exc_info:
        menu.run_safely(lambda args: raises())
    assert exc_info.value.code == 7


# ----- match_input ----------------------------------------------------


def test_match_input_empty_returns_none():
    assert menu.match_input("", {"1": "one"}) is None


def test_match_input_exit():
    assert menu.match_input("x", {}) == "__exit__"


def test_match_input_quit():
    assert menu.match_input("q", {}) == "__quit__"
    assert menu.match_input("quit", {}) == "__quit__"


def test_match_input_back():
    assert menu.match_input("b", {}) == "__back__"
    assert menu.match_input("back", {}) == "__back__"


def test_match_input_help():
    assert menu.match_input("?", {}) == "__help__"
    assert menu.match_input("h", {}) == "__help__"
    assert menu.match_input("help", {}) == "__help__"


def test_match_input_shell_disabled(capsys):
    out = menu.match_input("!ls", {})
    assert out == "__shell__"
    captured = capsys.readouterr()
    assert "Shell escape disabled" in captured.out


def test_match_input_exact_key():
    options = {"1": "Compress", "2": "Decompress"}
    assert menu.match_input("1", options) == "1"


def test_match_input_prefix_match():
    options = {"1": "Compress", "2": "Decompress"}
    assert menu.match_input("comp", options) == "1"


def test_match_input_no_match_returns_none():
    assert menu.match_input("zzz", {"1": "Compress"}) is None
