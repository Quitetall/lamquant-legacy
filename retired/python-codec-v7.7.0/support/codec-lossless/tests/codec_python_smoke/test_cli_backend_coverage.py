"""Coverage tests for ``lamquant_codec.cli.backend``.

The backend module reports configured implementations but routes product
encode/decode through the Rust operation broker. Pure helpers exercised:

  * ``detect_backend(cfg)`` returns one of {rust, python, custom}
  * ``get_backend_version(cfg)`` returns a string for every branch
  * ``run_encode`` / ``run_decode`` invoke canonical broker operations
  * Missing-broker and unsupported-backend paths return nonzero exit codes
  * Timeout / FileNotFoundError / KeyboardInterrupt branches

All subprocess invocations are mocked — no real binary is executed.
"""
from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lamquant_codec.cli import backend as backend_mod
from lamquant_codec.cli.config import LamQuantConfig


def _cfg(mode="python", custom_binary=""):
    cfg = LamQuantConfig()
    cfg.backend.mode = mode
    cfg.backend.custom_binary = custom_binary
    return cfg


# ----- detect_backend ---------------------------------------------------


def test_detect_backend_returns_string():
    cfg = _cfg("python")
    out = backend_mod.detect_backend(cfg)
    assert isinstance(out, str)
    assert out in {"rust", "python", "custom"}


def test_detect_backend_python_mode():
    assert backend_mod.detect_backend(_cfg("python")) == "python"


def test_detect_backend_rust_mode_when_explicit():
    cfg = _cfg("rust")
    out = backend_mod.detect_backend(cfg)
    # rust mode is honored even when no binary is found, because
    # `resolve()` just returns "rust" when mode == "rust"
    assert out == "rust"


def test_detect_backend_custom_with_binary():
    cfg = _cfg("custom", custom_binary="/some/path/lml")
    assert backend_mod.detect_backend(cfg) == "custom"


# ----- get_backend_version ----------------------------------------------


def test_get_backend_version_python_returns_string():
    out = backend_mod.get_backend_version(_cfg("python"))
    assert isinstance(out, str)
    assert "lamquant-codec" in out.lower() or "python" in out.lower()


def test_get_backend_version_rust_unknown_when_no_binary():
    """When ``rust`` mode is set but no binary is on PATH, version is unknown."""
    cfg = _cfg("rust")
    with patch(
        "lamquant_codec.cli.backend._resolve_binary", return_value=None
    ):
        out = backend_mod.get_backend_version(cfg)
    # branch returns "unknown" when binary not found
    assert isinstance(out, str)


def test_get_backend_version_rust_with_subprocess_success():
    cfg = _cfg("rust")
    fake_proc = SimpleNamespace(returncode=0, stdout="lml 1.2.3\n")
    with patch(
        "lamquant_codec.cli.backend._resolve_binary",
        return_value="/usr/bin/lml",
    ), patch(
        "lamquant_codec.cli.backend.subprocess.run", return_value=fake_proc
    ):
        out = backend_mod.get_backend_version(cfg)
    assert out == "lml 1.2.3"


def test_get_backend_version_rust_with_subprocess_failure():
    cfg = _cfg("rust")
    fake_proc = SimpleNamespace(returncode=1, stdout="")
    with patch(
        "lamquant_codec.cli.backend._resolve_binary",
        return_value="/usr/bin/lml",
    ), patch(
        "lamquant_codec.cli.backend.subprocess.run", return_value=fake_proc
    ):
        out = backend_mod.get_backend_version(cfg)
    assert out == "unknown"


def test_get_backend_version_rust_subprocess_raises():
    cfg = _cfg("rust")
    with patch(
        "lamquant_codec.cli.backend._resolve_binary",
        return_value="/usr/bin/lml",
    ), patch(
        "lamquant_codec.cli.backend.subprocess.run", side_effect=OSError("boom")
    ):
        out = backend_mod.get_backend_version(cfg)
    assert out == "unknown"


def test_get_backend_version_custom_success():
    cfg = _cfg("custom", custom_binary="/path/to/x")
    fake_proc = SimpleNamespace(returncode=0, stdout="custom 0.1\n")
    with patch(
        "lamquant_codec.cli.backend.subprocess.run", return_value=fake_proc
    ):
        out = backend_mod.get_backend_version(cfg)
    assert out == "custom 0.1"


def test_get_backend_version_custom_oserror():
    cfg = _cfg("custom", custom_binary="/nope")
    with patch(
        "lamquant_codec.cli.backend.subprocess.run", side_effect=OSError()
    ):
        out = backend_mod.get_backend_version(cfg)
    assert out == "unknown"


# ----- run_encode -------------------------------------------------------


def test_run_encode_rust_missing_broker_returns_one(capsys):
    cfg = _cfg("rust")
    with patch(
        "lamquant_codec.cli.backend._resolve_broker", return_value=None
    ):
        rc = backend_mod.run_encode(cfg, "/in", "/out")
    assert rc == 1


def test_run_encode_rust_uses_canonical_operation_broker():
    cfg = _cfg("rust")
    fake_proc = MagicMock()
    fake_proc.wait.return_value = 0
    with patch(
        "lamquant_codec.cli.backend._resolve_broker",
        return_value="/usr/bin/lamquant",
    ), patch(
        "lamquant_codec.cli.backend.subprocess.Popen", return_value=fake_proc
    ) as popen_mock:
        rc = backend_mod.run_encode(cfg, "/in", "/out")
    assert rc == 0
    assert popen_mock.call_args[0][0] == [
        "/usr/bin/lamquant",
        "op",
        "run",
        "encode_lma",
        "--input",
        "/in",
        "--output",
        "/out",
    ]


def test_run_encode_rejects_noncanonical_overrides(capsys):
    cfg = _cfg("rust")
    with patch("lamquant_codec.cli.backend.subprocess.Popen") as popen_mock:
        rc = backend_mod.run_encode(cfg, "/in", "/out", workers=2)
    assert rc == 2
    popen_mock.assert_not_called()
    assert "canonical encode_lma contract" in capsys.readouterr().err


def test_run_encode_rust_timeout_returns_one():
    cfg = _cfg("rust")
    fake_proc = MagicMock()
    fake_proc.wait.side_effect = [
        subprocess.TimeoutExpired("lamquant", 7200),
        0,
    ]
    fake_proc.poll.return_value = None
    with patch(
        "lamquant_codec.cli.backend._resolve_broker",
        return_value="/usr/bin/lamquant",
    ), patch(
        "lamquant_codec.cli.backend.subprocess.Popen",
        return_value=fake_proc,
    ):
        rc = backend_mod.run_encode(cfg, "/in", "/out")
    assert rc == 1
    fake_proc.terminate.assert_called_once_with()
    assert fake_proc.wait.call_args_list[-1].kwargs == {"timeout": 10}


def test_run_encode_rust_filenotfound_returns_one():
    cfg = _cfg("rust")
    with patch(
        "lamquant_codec.cli.backend._resolve_broker",
        return_value="/usr/bin/lamquant",
    ), patch(
        "lamquant_codec.cli.backend.subprocess.Popen",
        side_effect=FileNotFoundError(),
    ):
        rc = backend_mod.run_encode(cfg, "/in", "/out")
    assert rc == 1


def test_run_encode_rust_keyboard_interrupt_returns_130():
    cfg = _cfg("rust")
    fake_proc = MagicMock()
    fake_proc.wait.side_effect = [KeyboardInterrupt(), 0]
    fake_proc.poll.return_value = None
    with patch(
        "lamquant_codec.cli.backend._resolve_broker",
        return_value="/usr/bin/lamquant",
    ), patch(
        "lamquant_codec.cli.backend.subprocess.Popen",
        return_value=fake_proc,
    ), patch(
        "lamquant_codec.cli.backend.os.name",
        "posix",
    ):
        rc = backend_mod.run_encode(cfg, "/in", "/out")
    assert rc == 130
    fake_proc.send_signal.assert_called_once_with(backend_mod.signal.SIGINT)


def test_stop_broker_escalates_after_grace_timeout():
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    fake_proc.wait.side_effect = [subprocess.TimeoutExpired("lamquant", 10), 0]
    backend_mod._stop_broker(fake_proc, interrupt=False)
    fake_proc.terminate.assert_called_once_with()
    fake_proc.kill.assert_called_once_with()
    assert fake_proc.wait.call_args_list[-1].args == ()
    assert fake_proc.wait.call_args_list[-1].kwargs == {}


def test_run_encode_python_backend_fails_closed(capsys):
    cfg = _cfg("python")
    with patch("lamquant_codec.cli.backend.subprocess.Popen") as popen_mock:
        rc = backend_mod.run_encode(cfg, "/in", "/out")
    assert rc == 2
    popen_mock.assert_not_called()
    assert "requires the Rust operation broker" in capsys.readouterr().err


def test_run_encode_custom_backend_fails_closed_without_manifest(capsys):
    cfg = _cfg("custom", custom_binary="/custom/lml")
    with patch("lamquant_codec.cli.backend.subprocess.Popen") as popen_mock:
        rc = backend_mod.run_encode(cfg, "/in", "/out")
    assert rc == 2
    popen_mock.assert_not_called()
    assert "signed capability manifest" in capsys.readouterr().err


# ----- run_decode -------------------------------------------------------


def test_run_decode_rust_missing_broker_returns_one():
    cfg = _cfg("rust")
    with patch(
        "lamquant_codec.cli.backend._resolve_broker", return_value=None
    ):
        rc = backend_mod.run_decode(cfg, "/in", "/out")
    assert rc == 1


def test_run_decode_rust_uses_canonical_operation_broker():
    cfg = _cfg("rust")
    fake_proc = MagicMock()
    fake_proc.wait.return_value = 0
    with patch(
        "lamquant_codec.cli.backend._resolve_broker",
        return_value="/usr/bin/lamquant",
    ), patch(
        "lamquant_codec.cli.backend.subprocess.Popen", return_value=fake_proc
    ) as popen_mock:
        rc = backend_mod.run_decode(cfg, "/in", "/out")
    assert rc == 0
    assert popen_mock.call_args[0][0] == [
        "/usr/bin/lamquant",
        "op",
        "run",
        "decode",
        "--input",
        "/in",
        "--output",
        "/out",
    ]


def test_run_decode_rejects_noncanonical_overrides(capsys):
    cfg = _cfg("rust")
    with patch("lamquant_codec.cli.backend.subprocess.Popen") as popen_mock:
        rc = backend_mod.run_decode(cfg, "/in", "/out", workers=2)
    assert rc == 2
    popen_mock.assert_not_called()
    assert "canonical decode contract" in capsys.readouterr().err


def test_run_decode_rust_timeout():
    cfg = _cfg("rust")
    fake_proc = MagicMock()
    fake_proc.wait.side_effect = [
        subprocess.TimeoutExpired("lamquant", 7200),
        0,
    ]
    fake_proc.poll.return_value = None
    with patch(
        "lamquant_codec.cli.backend._resolve_broker",
        return_value="/usr/bin/lamquant",
    ), patch(
        "lamquant_codec.cli.backend.subprocess.Popen",
        return_value=fake_proc,
    ):
        rc = backend_mod.run_decode(cfg, "/in", "/out")
    assert rc == 1
    fake_proc.terminate.assert_called_once_with()


def test_run_decode_rust_filenotfound():
    cfg = _cfg("rust")
    with patch(
        "lamquant_codec.cli.backend._resolve_broker",
        return_value="/usr/bin/lamquant",
    ), patch(
        "lamquant_codec.cli.backend.subprocess.Popen",
        side_effect=FileNotFoundError(),
    ):
        rc = backend_mod.run_decode(cfg, "/in", "/out")
    assert rc == 1


def test_run_decode_rust_keyboard_interrupt():
    cfg = _cfg("rust")
    fake_proc = MagicMock()
    fake_proc.wait.side_effect = [KeyboardInterrupt(), 0]
    fake_proc.poll.return_value = None
    with patch(
        "lamquant_codec.cli.backend._resolve_broker",
        return_value="/usr/bin/lamquant",
    ), patch(
        "lamquant_codec.cli.backend.subprocess.Popen",
        return_value=fake_proc,
    ), patch(
        "lamquant_codec.cli.backend.os.name",
        "posix",
    ):
        rc = backend_mod.run_decode(cfg, "/in", "/out")
    assert rc == 130
    fake_proc.send_signal.assert_called_once_with(backend_mod.signal.SIGINT)


def test_run_decode_python_backend_fails_closed(capsys):
    cfg = _cfg("python")
    with patch("lamquant_codec.cli.backend.subprocess.Popen") as popen_mock:
        rc = backend_mod.run_decode(cfg, "/in", "/out")
    assert rc == 2
    popen_mock.assert_not_called()
    assert "requires the Rust operation broker" in capsys.readouterr().err
