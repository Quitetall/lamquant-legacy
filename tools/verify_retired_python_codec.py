#!/usr/bin/env python3
"""Verify retired Python codec source plus its final test/support closure."""

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
SUPPORT_ROOT = RETIREMENT_ROOT / "support/codec-lossless"
SUPPORT_MANIFEST_PATH = RETIREMENT_ROOT / "support-manifest.json"
SUPPORT_SCHEMA = "lamquant.legacy.python-codec-support/v1"
SOURCE_REPOSITORY = "https://github.com/Quitetall/LamQuant-Lossless.git"
SOURCE_REVISION = "f9b915466e67a87ad8d290a9793d349df250c9fb"
SOURCE_PATH = "reference_implementations/python_codec/lamquant_codec"
SUPPORT_REVISION = "b87216ffb66ad1c38d21da2857cf07afe3a4c518"
SUPPORT_PATHS = (
    "pyproject.toml",
    "tests/codec",
    "tests/codec_python_smoke",
    "tests/conftest.py",
    "tests/container/test_lml_container.py",
    "tests/edf_reader/test_edf_reader.py",
    "tests/edf_reader/test_mne_io.py",
    "tests/helpers/asserts.py",
    "tests/helpers/roundtrip.py",
    "tests/integration/test_batch.py",
    "tests/integration/test_init_wizard.py",
    "tests/integration/test_l3_batch.py",
    "tests/integration/test_l3_e2e_codec.py",
    "tests/integration/test_lma_dataset.py",
    "tests/integration/test_perf_sentinels.py",
    "tests/integration/test_snn_codec_integration.py",
    "tools/validate_rust_normalize.py",
)
FORBIDDEN_PACKAGING = ("pyproject.toml", "setup.py", "setup.cfg")
MAX_FILES = 1_000
MAX_BYTES = 32 * 1024 * 1024


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


def inventory_files(
    root: Path,
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    require(root.is_dir(), f"missing {label}: {root}")
    require(not root.is_symlink(), f"{label} root must not be a symlink")
    files: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"symlink forbidden in {label}: {path}")
        if path.is_dir():
            continue
        require(path.is_file(), f"special file forbidden in {label}: {path}")
        relative = path.relative_to(root).as_posix()
        checked_relative(relative)
        payload = path.read_bytes()
        total_bytes += len(payload)
        require(total_bytes <= MAX_BYTES, f"{label} exceeds byte limit")
        files[relative] = {
            "bytes": len(payload),
            "sha256": sha256(payload),
        }
        require(len(files) <= MAX_FILES, f"{label} exceeds file limit")
    require(bool(files), f"{label} is empty")
    return files


def source_files() -> dict[str, dict[str, Any]]:
    return inventory_files(SOURCE_ROOT, label="retired source")


def support_files() -> dict[str, dict[str, Any]]:
    return inventory_files(SUPPORT_ROOT, label="retired support closure")


def line_count(root: Path, files: dict[str, dict[str, Any]]) -> int:
    return sum((root / relative).read_bytes().count(b"\n") for relative in files)


def manifest_for(files: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tree_payload = canonical_json(files)
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
            # Match `wc -l`, which is how roadmap LOC estimates are recorded.
            "line_count": line_count(SOURCE_ROOT, files),
            "tree_sha256": sha256(tree_payload),
            "files": files,
        },
    }


def support_manifest_for(files: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": SUPPORT_SCHEMA,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SUPPORT_REVISION,
            "paths": list(SUPPORT_PATHS),
        },
        "inventory": {
            "file_count": len(files),
            "byte_count": sum(entry["bytes"] for entry in files.values()),
            "line_count": line_count(SUPPORT_ROOT, files),
            "tree_sha256": sha256(canonical_json(files)),
            "files": files,
        },
    }


def load_manifest(path: Path, *, label: str) -> dict[str, Any]:
    require(path.is_file(), f"missing {label} manifest: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"invalid {label} manifest: {error}") from error
    require(isinstance(value, dict), f"{label} manifest root must be an object")
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


def verify_support_repo(
    repo: Path,
    files: dict[str, dict[str, Any]],
) -> None:
    completed = subprocess.run(
        [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            SUPPORT_REVISION,
            "--",
            *SUPPORT_PATHS,
        ],
        cwd=repo,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise VerificationError(
            f"cannot inspect support revision: {completed.stderr.strip()}"
        )
    source_paths = sorted(completed.stdout.splitlines())
    require(
        source_paths == sorted(files),
        "retired support inventory differs from originating revision",
    )
    for relative, entry in files.items():
        payload = git_bytes(repo, SUPPORT_REVISION, relative)
        require(
            len(payload) == entry["bytes"],
            f"support byte count differs: {relative}",
        )
        require(
            sha256(payload) == entry["sha256"],
            f"support hash differs: {relative}",
        )


def verify(source_repo: Path | None) -> dict[str, dict[str, Any]]:
    for name in FORBIDDEN_PACKAGING:
        require(
            not (RETIREMENT_ROOT / name).exists(),
            f"retired snapshot must remain non-publishable: {name}",
        )
    actual_files = source_files()
    expected = manifest_for(actual_files)
    manifest = load_manifest(MANIFEST_PATH, label="source")
    require(manifest == expected, "retired source manifest or source bytes drifted")
    require(expected["inventory"]["file_count"] == 65, "unexpected source file count")
    require(expected["inventory"]["line_count"] == 20_101, "unexpected source line count")
    actual_support = support_files()
    expected_support = support_manifest_for(actual_support)
    support_manifest = load_manifest(SUPPORT_MANIFEST_PATH, label="support")
    require(
        support_manifest == expected_support,
        "retired support manifest or support bytes drifted",
    )
    require(
        expected_support["inventory"]["file_count"] == 90,
        "unexpected support file count",
    )
    if source_repo is not None:
        source_repo = source_repo.resolve()
        verify_source_repo(source_repo, actual_files)
        verify_support_repo(source_repo, actual_support)
    return {"source": expected, "support": expected_support}


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
        help="write canonical source and support manifests before verification",
    )
    args = parser.parse_args()
    try:
        if args.write_manifest:
            MANIFEST_PATH.write_bytes(canonical_json(manifest_for(source_files())))
            SUPPORT_MANIFEST_PATH.write_bytes(
                canonical_json(support_manifest_for(support_files()))
            )
        result = verify(args.source_repo)
    except VerificationError as error:
        print(f"retired-python-codec: FAIL: {error}", file=sys.stderr)
        return 1
    print(
        "retired-python-codec: PASS "
        f"source_files={result['source']['inventory']['file_count']} "
        f"support_files={result['support']['inventory']['file_count']} "
        f"source_tree_sha256={result['source']['inventory']['tree_sha256']} "
        f"support_tree_sha256={result['support']['inventory']['tree_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
