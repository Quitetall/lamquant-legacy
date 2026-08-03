"""First-run wizard.

The first 60 seconds with any tool determine whether users stick around.
This wizard runs once on a fresh install and walks the user through:

  1. Hardware detection (CPU / GPU / RAM)
  2. Dependency check (Python, key packages)
  3. Model-weights presence check (offers to download if missing)
  4. Sample compress/decompress on bundled synthetic data — proves it works
  5. Print pointers to docs, CLI commands, GUI

Marker
------
After a successful run the wizard writes:
    ~/.lamquant/init_done    (touch — modtime is the install timestamp)

So `lamquant init` is idempotent: re-runs print "already initialised"
unless `--force` is passed. New users hit the wizard exactly once.

CLI
---
    lamquant init           # run if not yet initialised; no-op otherwise
    lamquant init --force   # always run
    lamquant init --quiet   # suppress prompts (CI / scripted installs)
"""

from __future__ import annotations

import os
import sys
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Where the marker file lives.
def marker_path() -> Path:
    """~/.lamquant/init_done — created on first successful wizard run."""
    return Path(os.path.expanduser('~/.lamquant')) / 'init_done'


@dataclass
class WizardResult:
    """Structured result so callers / scripts can inspect what happened."""
    initialised: bool = False
    skipped: bool = False
    sections: dict = field(default_factory=dict)   # section_name -> dict
    next_steps: list = field(default_factory=list)


# ============================================================
# Section helpers — each returns a dict that gets stored in result.sections
# and either prints a one-line summary or a multi-line block.
# ============================================================

def _print_header(text: str):
    print()
    print(f"  ── {text} " + "─" * max(0, 70 - 6 - len(text)))


def _ok(msg: str):    print(f"      ✓ {msg}")
def _warn(msg: str):  print(f"      ⚠ {msg}")
def _info(msg: str):  print(f"        {msg}")


def detect_hardware(quiet: bool = False) -> dict:
    """Detect CPU, GPU, RAM. Best-effort, no hard dependencies."""
    if not quiet:
        _print_header("[1/5] Hardware")

    info = {
        'platform': platform.platform(),
        'python': sys.version.split()[0],
        'cpu_count': os.cpu_count(),
    }

    # CPU model — read /proc/cpuinfo on Linux, sysctl on macOS.
    cpu_model = None
    try:
        if sys.platform == 'linux':
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if line.startswith('model name'):
                        cpu_model = line.split(':', 1)[1].strip()
                        break
        elif sys.platform == 'darwin':
            cpu_model = subprocess.check_output(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        pass
    info['cpu_model'] = cpu_model

    # RAM in GB.
    ram_gb = None
    try:
        if sys.platform == 'linux':
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        kb = int(line.split()[1])
                        ram_gb = round(kb / (1024 * 1024), 1)
                        break
        elif sys.platform == 'darwin':
            mem_bytes = int(subprocess.check_output(
                ['sysctl', '-n', 'hw.memsize'], text=True))
            ram_gb = round(mem_bytes / (1024 ** 3), 1)
    except Exception:
        pass
    info['ram_gb'] = ram_gb

    # GPU — only honest answer is "ask torch if it's installed".
    gpu = None
    try:
        import torch    # may add ~600 ms; only happens when wizard runs
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            gpu = 'Apple Silicon (MPS)'
    except Exception:
        pass
    info['gpu'] = gpu

    if not quiet:
        if cpu_model:
            _ok(f"CPU: {info['cpu_count']} cores ({cpu_model})")
        else:
            _ok(f"CPU: {info['cpu_count']} cores")
        if ram_gb:
            _ok(f"RAM: {ram_gb} GB")
        else:
            _info("RAM: unknown")
        if gpu:
            _ok(f"GPU: {gpu}")
        else:
            _info("GPU: none detected (CPU mode is fine)")

    return info


def check_dependencies(quiet: bool = False) -> dict:
    """Check that key Python packages are importable."""
    if not quiet:
        _print_header("[2/5] Dependencies")

    required = ['numpy', 'scipy', 'numba']
    optional = ['torch', 'mne', 'matplotlib', 'jinja2', 'tqdm']

    found = {}
    for name in required + optional:
        try:
            mod = __import__(name)
            ver = getattr(mod, '__version__', '?')
            found[name] = ver
        except ImportError:
            found[name] = None

    missing_required = [n for n in required if found[n] is None]
    missing_optional = [n for n in optional if found[n] is None]

    if not quiet:
        for name in required:
            if found[name]:
                _ok(f"{name} {found[name]}")
            else:
                _warn(f"{name} NOT INSTALLED (required)")
        for name in optional:
            if found[name]:
                _ok(f"{name} {found[name]}")
            else:
                _info(f"{name} not installed (optional)")

    return {
        'installed': {k: v for k, v in found.items() if v},
        'missing_required': missing_required,
        'missing_optional': missing_optional,
        'ok': not missing_required,
    }


def check_model_weights(quiet: bool = False) -> dict:
    """Look for an encoder checkpoint in the standard locations."""
    if not quiet:
        _print_header("[3/5] Model weights")

    from lamquant_codec._paths import REPO_ROOT as repo_root
    candidates = [
        repo_root / 'weights' / 'student_subband_gold.ckpt',
        repo_root / 'weights' / 'student_subband.ckpt',
    ]
    found = next((p for p in candidates if p.exists()), None)

    info = {
        'checkpoint': str(found) if found else None,
        'searched': [str(p) for p in candidates],
    }

    if not quiet:
        if found:
            sz = found.stat().st_size / (1024 * 1024)
            _ok(f"encoder checkpoint: {found.name} ({sz:.1f} MB)")
        else:
            _warn("no encoder checkpoint found in weights/")
            _info("Lossless mode (.lml) works without one — neural mode (.lmq) does not.")
            _info("Train one with:  lamquant train")
            _info("Or download the latest release from GitHub.")

    return info


def smoke_test(quiet: bool = False) -> dict:
    """End-to-end compress→decompress on a synthetic in-memory window.

    Proves the install actually works without needing any external data
    files or model weights — uses the lossless codec which has no neural
    dependency. All errors (including ImportError for missing numba/etc.)
    are caught so the wizard always finishes and reports the failure
    cleanly instead of crashing.
    """
    if not quiet:
        _print_header("[4/5] Smoke test (lossless codec)")

    try:
        import numpy as np
        # Defer the codec import — it pulls numba via ops.golomb / ops.rans.
        # If a required package is missing this import is where it'll fail,
        # and we want to catch it cleanly rather than crash the whole wizard.
        from lamquant_codec.lossless import LosslessCodec

        rng = np.random.default_rng(0)
        sig = rng.integers(-2000, 2000, (21, 2500)).astype(np.float64)
        raw_bytes = sig.nbytes

        codec = LosslessCodec()
        compressed = codec.compress(sig)
        recon = codec.decompress(compressed)
        bit_exact = np.array_equal(recon.astype(np.int64),
                                   sig.astype(np.int64))
        cr = raw_bytes / max(len(compressed), 1)
        result = {
            'ok': bit_exact,
            'raw_bytes': raw_bytes,
            'compressed_bytes': len(compressed),
            'cr': cr,
        }
        if not quiet:
            if bit_exact:
                _ok(f"21ch × 2500 → {len(compressed):,}B  (CR {cr:.1f}:1, bit-exact)")
            else:
                _warn("compress/decompress round-trip is NOT bit-exact (please file a bug)")
        return result
    except ImportError as e:
        if not quiet:
            _warn(f"cannot import codec: {e}")
            _info("Re-install missing packages, then run `lamquant init --force`.")
        return {'ok': False, 'error': f'ImportError: {e}'}
    except Exception as e:
        if not quiet:
            _warn(f"smoke test FAILED: {type(e).__name__}: {e}")
        return {'ok': False, 'error': f'{type(e).__name__}: {e}'}


def print_next_steps(deps: dict, weights: dict) -> list[str]:
    """Print a punch-list of useful first commands and return them."""
    print()
    print("  ── [5/5] You're ready. Try one of these:".ljust(70, ' ') + " ─" * 0)

    steps = []

    if not deps.get('ok'):
        steps.append("Install missing required packages first:")
        for pkg in deps.get('missing_required', []):
            steps.append(f"   pip install {pkg}")

    steps.append("Compress a directory of EEG files (no model needed):")
    steps.append("   lamquant compress  data/   -o compressed/  -r --mode lossless")
    steps.append("")
    steps.append("Decompress one back to .npy:")
    steps.append("   lamquant decompress compressed/ -o decoded/ -r")
    steps.append("")
    steps.append("Validate quality across an archive (HTML report):")
    steps.append("   lamquant validate  compressed/ -r --reference data/ \\")
    steps.append("                       --level C --report-html quality.html")
    steps.append("")
    steps.append("Inspect a single file's header without decoding:")
    steps.append("   lamquant info compressed/recording.lml")
    steps.append("")

    if weights.get('checkpoint'):
        steps.append(f"Neural compression ({Path(weights['checkpoint']).name}):")
        steps.append("   lamquant compress data/ -o compressed/ --mode neural \\")
        steps.append(f"                     -c {weights['checkpoint']}")
    else:
        steps.append("To use neural compression (.lmq, ~270:1 ratio), train a model:")
        steps.append("   lamquant train")

    steps.append("")
    steps.append("Documentation: https://github.com/Quitetall/LamQuant-Lossless")

    for s in steps:
        if s.startswith('   '):
            print(f"     {s.lstrip()}")
        elif s == '':
            print()
        else:
            print(f"  • {s}")

    return steps


# ============================================================
# Top-level entry
# ============================================================

def run(force: bool = False, quiet: bool = False) -> WizardResult:
    """Run the first-run wizard. Returns a WizardResult."""
    marker = marker_path()
    result = WizardResult()

    if marker.exists() and not force:
        result.skipped = True
        if not quiet:
            ts = marker.stat().st_mtime
            from datetime import datetime
            print(f"\n  LamQuant already initialised on "
                  f"{datetime.fromtimestamp(ts).strftime('%Y-%m-%d')}.")
            print(f"  Re-run with `lamquant init --force` to repeat.\n")
        return result

    if not quiet:
        print()
        print("  ╔══════════════════════════════════════════╗")
        print("  ║  LamQuant — First-run setup              ║")
        print("  ║  Verifies your install + shows the basics║")
        print("  ╚══════════════════════════════════════════╝")

    result.sections['hardware'] = detect_hardware(quiet)
    result.sections['dependencies'] = check_dependencies(quiet)
    result.sections['weights'] = check_model_weights(quiet)
    result.sections['smoke'] = smoke_test(quiet)

    smoke_ok = result.sections['smoke'].get('ok', False)

    if not quiet:
        result.next_steps = print_next_steps(
            result.sections['dependencies'],
            result.sections['weights'])

    # Mark as initialised only if the smoke test passed.
    if smoke_ok:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        result.initialised = True
    else:
        if not quiet:
            print()
            _warn("Smoke test failed — NOT marking install as complete.")
            _info("Fix the error above, then re-run `lamquant init --force`.")

    return result


__all__ = [
    'run', 'marker_path', 'WizardResult',
    'detect_hardware', 'check_dependencies', 'check_model_weights',
    'smoke_test', 'print_next_steps',
]
