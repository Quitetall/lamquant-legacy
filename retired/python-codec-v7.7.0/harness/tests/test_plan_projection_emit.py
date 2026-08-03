"""Python parity tests for graph-bound plan projections."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

from lamquant_codec.cli.plan_projection_emit import (
    PROJECTION_ARTIFACT,
    PROJECTION_DIAGNOSTIC,
    PROJECTION_PLANNED,
    PROJECTION_PROGRESS,
    PlanIdentity,
    PlanProjection,
    artifact,
    diagnostic,
    parse_line,
    parse_lines,
    planned,
    progress,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURE = (
    REPO_ROOT
    / "crates"
    / "lamquant-ops"
    / "tests"
    / "fixtures"
    / "plan-projections-sample.jsonl"
)
INVALID_FIXTURE = FIXTURE.with_name("plan-projections-invalid.jsonl")
IDENTITY = PlanIdentity("11" * 32, "22" * 32, "33" * 32)


def test_parse_planned_projection():
    line = json.dumps(
        {
            "schema": "org.quitetall.lamquant.plan-projection/v1",
            "observed_at_ms": 1,
            "plan": IDENTITY.as_dict(),
            "projection": PROJECTION_PLANNED,
            "operation": "encode_lma",
            "total_nodes": 1,
            "total_work": 42,
        }
    )
    projection = parse_line(line)
    assert isinstance(projection, PlanProjection)
    assert projection.plan == IDENTITY
    assert projection.projection == PROJECTION_PLANNED


@pytest.mark.parametrize(
        ("producer", "expected"),
        [
            (
                lambda: planned("encode_lma", total_work=42, identity=IDENTITY),
                PROJECTION_PLANNED,
            ),
        (
            lambda: progress(5, 42, "file 5/42", identity=IDENTITY),
            PROJECTION_PROGRESS,
        ),
        (
            lambda: artifact(
                "a.lml",
                True,
                12,
                identity=IDENTITY,
                bytes_in=100,
                bytes_out=40,
            ),
            PROJECTION_ARTIFACT,
        ),
        (
            lambda: diagnostic("warning", level="warning", identity=IDENTITY),
            PROJECTION_DIAGNOSTIC,
        ),
    ],
)
def test_producers_emit_parseable_identity_bound_lines(producer, expected, capsys):
    producer()
    projection = parse_line(capsys.readouterr().out.strip())
    assert projection.projection == expected
    assert projection.plan == IDENTITY


def test_parse_rejects_unknown_projection():
    line = json.dumps(
        {
            "schema": "org.quitetall.lamquant.plan-projection/v1",
            "observed_at_ms": 1,
            "plan": IDENTITY.as_dict(),
            "projection": "mystery",
        }
    )
    with pytest.raises(ValueError, match="unknown PlanProjection"):
        parse_line(line)


def test_parse_rejects_uppercase_identity():
    value = {
        "schema": "org.quitetall.lamquant.plan-projection/v1",
        "observed_at_ms": 1,
        "plan": {**IDENTITY.as_dict(), "graph_id": "AA" * 32},
        "projection": PROJECTION_DIAGNOSTIC,
        "level": "info",
        "message": "x",
    }
    with pytest.raises(ValueError, match="lower-case"):
        parse_line(json.dumps(value))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"surprise": True}, "unknown fields"),
        ({"total_nodes": True}, "invalid planned"),
        ({"total_nodes": "1"}, "invalid planned"),
    ],
)
def test_parse_rejects_noncanonical_planned_shape(changes, message):
    value = {
        "schema": "org.quitetall.lamquant.plan-projection/v1",
        "observed_at_ms": 1,
        "plan": IDENTITY.as_dict(),
        "projection": PROJECTION_PLANNED,
        "operation": "encode_lma",
        "total_nodes": 1,
        **changes,
    }
    with pytest.raises(ValueError, match=message):
        parse_line(json.dumps(value))


def test_parse_rejects_incomplete_artifact_shape():
    value = {
        "schema": "org.quitetall.lamquant.plan-projection/v1",
        "observed_at_ms": 1,
        "plan": IDENTITY.as_dict(),
        "projection": PROJECTION_ARTIFACT,
        "node_id": 0,
        "artifact": {
            "path": "a.lml",
            "success": True,
            "elapsed_ms": 1,
            "bytes_in": 42,
        },
    }
    with pytest.raises(ValueError, match="complete or absent"):
        parse_line(json.dumps(value))


def test_parse_lines_skips_malformed_and_continues(capsys):
    valid = {
        "schema": "org.quitetall.lamquant.plan-projection/v1",
        "observed_at_ms": 1,
        "plan": IDENTITY.as_dict(),
        "projection": PROJECTION_DIAGNOSTIC,
        "level": "info",
        "message": "ok",
    }
    stream = [json.dumps(valid) + "\n", "{not json\n", json.dumps(valid) + "\n"]
    projections = list(parse_lines(stream))
    assert len(projections) == 2
    assert "dropping malformed projection line" in capsys.readouterr().err


def test_check_command_passes_against_fixture():
    assert FIXTURE.exists()
    environment = os.environ.copy()
    python_root = (
        REPO_ROOT
        / "codec-lossless"
        / "reference_implementations"
        / "python_codec"
    )
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(python_root), existing_pythonpath]
        if existing_pythonpath
        else [str(python_root)]
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lamquant_codec.cli.plan_projection_emit",
            "--check",
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "6 lines OK; 8 invalid lines rejected" in result.stdout


def test_shared_invalid_corpus_is_rejected():
    assert INVALID_FIXTURE.exists()
    invalid = [
        line
        for line in INVALID_FIXTURE.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert len(invalid) == 8
    for line in invalid:
        with pytest.raises(ValueError):
            parse_line(line)
