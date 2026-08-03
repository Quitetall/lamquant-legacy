"""Tests for the first-run wizard.

Verifies:
  - run() produces a complete WizardResult with all sections populated
  - Idempotent: a second run is a no-op (skipped=True)
  - --force re-runs even when the marker exists
  - --quiet produces no stdout
  - The smoke test actually exercises the lossless codec
  - Marker file is written under HOME/.lamquant
  - Wizard does NOT mark install complete when smoke test fails
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lamquant_codec import init_wizard


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Point HOME at a tmp dir so ~/.lamquant tests don't touch the real one."""
    monkeypatch.setenv('HOME', str(tmp_path))
    return tmp_path


def test_marker_path_under_home(isolated_home):
    p = init_wizard.marker_path()
    assert p == isolated_home / '.lamquant' / 'init_done'


def test_first_run_creates_marker_and_returns_initialised(isolated_home, capsys):
    result = init_wizard.run(quiet=True)
    assert result.initialised is True
    assert result.skipped is False
    assert init_wizard.marker_path().exists()


def test_second_run_is_skipped(isolated_home, capsys):
    init_wizard.run(quiet=True)
    result2 = init_wizard.run(quiet=True)
    assert result2.skipped is True
    assert result2.initialised is False


def test_force_reruns_even_if_marker_exists(isolated_home, capsys):
    init_wizard.run(quiet=True)
    result2 = init_wizard.run(force=True, quiet=True)
    assert result2.initialised is True
    assert result2.skipped is False


def test_quiet_produces_no_stdout(isolated_home, capsys):
    init_wizard.run(quiet=True)
    captured = capsys.readouterr()
    assert captured.out == ''


def test_normal_run_prints_human_output(isolated_home, capsys):
    init_wizard.run(quiet=False)
    captured = capsys.readouterr()
    assert 'LamQuant' in captured.out
    assert 'Hardware' in captured.out
    assert 'Dependencies' in captured.out
    assert 'Smoke test' in captured.out


def test_all_sections_populated(isolated_home):
    result = init_wizard.run(quiet=True)
    assert set(result.sections.keys()) == {
        'hardware', 'dependencies', 'weights', 'smoke',
    }


def test_hardware_section_has_expected_keys(isolated_home):
    result = init_wizard.run(quiet=True)
    hw = result.sections['hardware']
    for key in ('platform', 'python', 'cpu_count'):
        assert key in hw, f'missing hardware.{key}'
    assert hw['cpu_count'] >= 1


def test_dependencies_section_reports_status(isolated_home):
    result = init_wizard.run(quiet=True)
    deps = result.sections['dependencies']
    assert 'installed' in deps and 'missing_required' in deps
    # numpy is required and we know it's installed.
    assert 'numpy' in deps['installed']
    assert deps['ok'] is True


def test_smoke_test_actually_compresses(isolated_home):
    """The smoke section must report a real CR > 1 and bit-exact result."""
    result = init_wizard.run(quiet=True)
    smoke = result.sections['smoke']
    assert smoke['ok'] is True
    assert smoke['raw_bytes'] == 21 * 2500 * 8   # float64
    assert smoke['compressed_bytes'] > 0
    assert smoke['cr'] > 1.0


def test_smoke_failure_blocks_marker(isolated_home, monkeypatch):
    """If the smoke test errors, the marker is NOT written."""
    def boom(*args, **kw):
        return {'ok': False, 'error': 'simulated failure'}
    monkeypatch.setattr(init_wizard, 'smoke_test', boom)
    result = init_wizard.run(quiet=True)
    assert result.initialised is False
    assert not init_wizard.marker_path().exists()


def test_check_dependencies_detects_missing(monkeypatch):
    """If a 'required' package is missing, the dependency check reports it."""
    real_import = __builtins__['__import__'] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'numpy':
            raise ImportError('simulated missing numpy')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr('builtins.__import__', fake_import)
    deps = init_wizard.check_dependencies(quiet=True)
    assert 'numpy' in deps['missing_required']
    assert deps['ok'] is False


def test_check_model_weights_returns_dict():
    """Even when no checkpoint is present, the function returns a dict."""
    info = init_wizard.check_model_weights(quiet=True)
    assert 'checkpoint' in info
    assert 'searched' in info
    # checkpoint may be None (no weights) or a real path.
    assert info['checkpoint'] is None or Path(info['checkpoint']).exists()


def test_print_next_steps_emits_lines(isolated_home, capsys):
    deps = init_wizard.check_dependencies(quiet=True)
    weights = init_wizard.check_model_weights(quiet=True)
    steps = init_wizard.print_next_steps(deps, weights)
    captured = capsys.readouterr()
    # At least the lossless and validate hints should be there.
    assert 'compress' in captured.out
    assert 'validate' in captured.out
    # Returned list mirrors what was printed.
    assert any('compress' in s for s in steps)
