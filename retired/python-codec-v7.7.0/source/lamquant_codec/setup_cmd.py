"""Implementation of `lamquant setup` — wraps OpenHuman Portal install scripts."""

from __future__ import annotations

import subprocess
import sys
from lamquant_codec._paths import REPO_ROOT as PROJECT_ROOT


def run(yes: bool = False) -> int:
    """Invoke the appropriate OpenHuman Portal installer for the current OS."""
    root = PROJECT_ROOT

    if sys.platform.startswith("win"):
        script = root / "install.ps1"
        if not script.exists():
            print(f"error: {script} not found", file=sys.stderr)
            return 1
        cmd = [
            "powershell",
            "-ExecutionPolicy", "Bypass",
            "-File", str(script),
        ]
        if yes:
            cmd.append("-Quiet")
    else:
        script = root / "install.sh"
        if not script.exists():
            print(f"error: {script} not found", file=sys.stderr)
            return 1
        cmd = ["bash", str(script)]
        if yes:
            cmd.append("--quiet")

    print(f"Running OpenHuman Portal: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(root))
    return proc.returncode
