#!/usr/bin/env python3
"""Small fail-closed provenance check for the independent legacy repository."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def git(*args: str, repo: str | None = None) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout


def check_message(message: str, changed: set[str]) -> None:
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


def check(commit: str, repo: str | None = None) -> None:
    message = git("show", "-s", "--format=%B", commit, repo=repo)
    changed = {
        line.split("\t")[-1]
        for line in git(
            "diff-tree",
            "--root",
            "--no-commit-id",
            "--name-status",
            "-r",
            commit,
            repo=repo,
        ).splitlines()
        if line
    }
    check_message(message, changed)


def _blob(ref: str, path: str, repo: str) -> str | None:
    """Blob id at `ref`, or None when the path does not exist there."""
    try:
        return git("rev-parse", f"{ref}:{path}", repo=repo).strip()
    except Exception:  # noqa: BLE001 - absent on that side
        return None


def _drop_inherited_merge_paths(changed: set[str], repo: str) -> set[str]:
    """During a merge, keep only what the merger actually authored.

    `git diff --cached` compares the index against HEAD, the FIRST parent, so a
    merge reports every file the second parent contributed and the policy then
    demands a per-file authorship trailer for each. That claim is false: a clean
    merge authors nothing. Those files were authored on the branch being merged
    and already carry that attribution in their own commits.

    What a merger genuinely authors is conflict resolution -- content matching
    NEITHER side. A path whose staged blob equals MERGE_HEAD's came across
    verbatim and is dropped. Absence counts as a value: a file deleted on the
    merged branch is absent from both, and that agreement means the deletion was
    inherited too. Outside a merge this is a no-op.
    """
    try:
        git_dir = Path(git("rev-parse", "--absolute-git-dir", repo=repo).strip())
    except Exception:  # noqa: BLE001 - not a git dir
        return changed
    if not (git_dir / "MERGE_HEAD").exists():
        return changed
    return {
        path
        for path in changed
        if _blob(":0", path, repo) != _blob("MERGE_HEAD", path, repo)
    }


def check_staged(message_path: str, repo: str) -> None:
    message = Path(message_path).read_text(encoding="utf-8")
    changed = {
        line.split("\t")[-1]
        for line in git("diff", "--cached", "--name-status", "--no-renames", repo=repo).splitlines()
        if line
    }
    check_message(message, _drop_inherited_merge_paths(changed, repo))


def main() -> None:
    if len(sys.argv) == 2:
        check(sys.argv[1])
        return
    if len(sys.argv) == 5 and sys.argv[1] == "--repo" and sys.argv[3] == "staged":
        check_staged(sys.argv[4], sys.argv[2])
        return
    raise SystemExit(
        "usage: check_commit_provenance.py COMMIT | "
        "--repo REPO staged COMMIT_MESSAGE"
    )


if __name__ == "__main__":
    main()
