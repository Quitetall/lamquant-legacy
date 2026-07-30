#!/usr/bin/env python3
"""Verify the exact, non-publishable retired Python codec source snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RETIREMENT_ROOT = ROOT / "retired/python-codec-v7.7.0"
SOURCE_ROOT = RETIREMENT_ROOT / "source/lamquant_codec"
MANIFEST_PATH = RETIREMENT_ROOT / "source-manifest.json"
SCHEMA = "lamquant.legacy.python-codec-source/v1"
SOURCE_REPOSITORY = "https://github.com/Quitetall/LamQuant-Lossless.git"
SOURCE_REVISION = "f9b915466e67a87ad8d290a9793d349df250c9fb"
SOURCE_PATH = "reference_implementations/python_codec/lamquant_codec"
FORBIDDEN_PACKAGING = ("pyproject.toml", "setup.py", "setup.cfg")
MAX_FILES = 1_000
MAX_BYTES = 16 * 1024 * 1024


class VerificationError(RuntimeError):
    """One fail-closed retirement-snapshot violation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def checked_relative(path: str) -> PurePosixPath:
    relative = PurePosixPath(path)
    require(
        bool(relative.parts)
        and not relative.is_absolute()
        and "." not in relative.parts
        and ".." not in relative.parts,
        f"unsafe manifest path: {path!r}",
    )
    return relative


def source_files() -> dict[str, dict[str, Any]]:
    require(SOURCE_ROOT.is_dir(), f"missing retired source: {SOURCE_ROOT}")
    require(not SOURCE_ROOT.is_symlink(), "retired source root must not be a symlink")
    files: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for path in sorted(SOURCE_ROOT.rglob("*")):
        require(not path.is_symlink(), f"symlink forbidden in retired source: {path}")
        if path.is_dir():
            continue
        require(path.is_file(), f"special file forbidden in retired source: {path}")
        relative = path.relative_to(SOURCE_ROOT).as_posix()
        checked_relative(relative)
        payload = path.read_bytes()
        total_bytes += len(payload)
        require(total_bytes <= MAX_BYTES, "retired source exceeds byte limit")
        files[relative] = {
            "bytes": len(payload),
            "sha256": sha256(payload),
        }
        require(len(files) <= MAX_FILES, "retired source exceeds file limit")
    require(bool(files), "retired source is empty")
    return files


def manifest_for(files: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tree_payload = canonical_json(files)
    line_count = 0
    for relative in files:
        payload = (SOURCE_ROOT / relative).read_bytes()
        # Match `wc -l`, which is how roadmap LOC estimates are recorded.
        line_count += payload.count(b"\n")
    return {
        "schema": SCHEMA,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "path": SOURCE_PATH,
        },
        "inventory": {
            "file_count": len(files),
            "byte_count": sum(entry["bytes"] for entry in files.values()),
            "line_count": line_count,
            "tree_sha256": sha256(tree_payload),
            "files": files,
        },
    }


def load_manifest() -> dict[str, Any]:
    require(MANIFEST_PATH.is_file(), f"missing manifest: {MANIFEST_PATH}")
    try:
        value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"invalid manifest: {error}") from error
    require(isinstance(value, dict), "manifest root must be an object")
    return value


def git_bytes(repo: Path, revision: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=repo,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise VerificationError(
            f"git show failed for {revision}:{path}: "
            f"{completed.stderr.decode(errors='replace').strip()}"
        )
    return completed.stdout


def verify_source_repo(
    repo: Path,
    files: dict[str, dict[str, Any]],
) -> None:
    require(repo.is_dir(), f"source repository does not exist: {repo}")
    completed = subprocess.run(
        [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            SOURCE_REVISION,
            "--",
            SOURCE_PATH,
        ],
        cwd=repo,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise VerificationError(
            f"cannot inspect source revision: {completed.stderr.strip()}"
        )
    prefix = f"{SOURCE_PATH}/"
    source_paths = sorted(
        line[len(prefix) :]
        for line in completed.stdout.splitlines()
        if line.startswith(prefix)
    )
    require(
        source_paths == sorted(files),
        "retired file inventory differs from originating revision",
    )
    for relative, entry in files.items():
        payload = git_bytes(repo, SOURCE_REVISION, f"{SOURCE_PATH}/{relative}")
        require(len(payload) == entry["bytes"], f"source byte count differs: {relative}")
        require(sha256(payload) == entry["sha256"], f"source hash differs: {relative}")


def verify(source_repo: Path | None) -> dict[str, Any]:
    for name in FORBIDDEN_PACKAGING:
        require(
            not (RETIREMENT_ROOT / name).exists(),
            f"retired snapshot must remain non-publishable: {name}",
        )
    actual_files = source_files()
    expected = manifest_for(actual_files)
    manifest = load_manifest()
    require(manifest == expected, "retired source manifest or source bytes drifted")
    require(expected["inventory"]["file_count"] == 65, "unexpected source file count")
    require(expected["inventory"]["line_count"] == 20_101, "unexpected source line count")
    if source_repo is not None:
        verify_source_repo(source_repo.resolve(), actual_files)
    return expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-repo",
        type=Path,
        help="optional LamQuant-Lossless checkout for cross-repository byte proof",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="write canonical manifest from the local snapshot before verification",
    )
    args = parser.parse_args()
    try:
        if args.write_manifest:
            MANIFEST_PATH.write_bytes(canonical_json(manifest_for(source_files())))
        result = verify(args.source_repo)
    except VerificationError as error:
        print(f"retired-python-codec: FAIL: {error}", file=sys.stderr)
        return 1
    print(
        "retired-python-codec: PASS "
        f"files={result['inventory']['file_count']} "
        f"bytes={result['inventory']['byte_count']} "
        f"tree_sha256={result['inventory']['tree_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
