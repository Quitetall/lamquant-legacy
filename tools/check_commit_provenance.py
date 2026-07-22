#!/usr/bin/env python3
"""Small fail-closed provenance check for the independent legacy repository."""

from __future__ import annotations

import json
import subprocess
import sys


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, text=True, stdout=subprocess.PIPE
    ).stdout


def check(commit: str) -> None:
    message = git("show", "-s", "--format=%B", commit)
    changed = {
        line.split("\t")[-1]
        for line in git("diff-tree", "--root", "--no-commit-id", "--name-status", "-r", commit)
        .splitlines()
        if line
    }
    assisted = []
    contributions: dict[str, dict[str, object]] = {}
    for line in message.splitlines():
        if line.startswith("AI-Assisted-By: "):
            assisted.append(json.loads(line.removeprefix("AI-Assisted-By: ")))
        elif line.startswith("File-Contribution: "):
            value = json.loads(line.removeprefix("File-Contribution: "))
            path = value.get("path")
            if not isinstance(path, str) or path in contributions:
                raise SystemExit("duplicate or invalid File-Contribution path")
            contributions[path] = value
    if not assisted:
        raise SystemExit("missing AI-Assisted-By trailer")
    if set(contributions) != changed:
        missing = sorted(changed - set(contributions))
        extra = sorted(set(contributions) - changed)
        raise SystemExit(f"File-Contribution mismatch: missing={missing} extra={extra}")
    for path, value in contributions.items():
        if not value.get("by") or not value.get("summary") or not value.get("operation"):
            raise SystemExit(f"incomplete File-Contribution for {path}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_commit_provenance.py COMMIT")
    check(sys.argv[1])


if __name__ == "__main__":
    main()
