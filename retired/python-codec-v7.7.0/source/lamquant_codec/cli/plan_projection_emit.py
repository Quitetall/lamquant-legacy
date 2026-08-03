"""Dependency-free Python plan-projection producer and parser.

Live observations are bound to compiled ``graph_id``, ``plan_id``, and
``invocation_id`` values. Python producers never invent terminal receipts;
the supervising graph executor owns receipt and failure projection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional, TypeAlias, Union, cast

JsonScalar: TypeAlias = Union[None, bool, int, float, str]
JsonValue: TypeAlias = Union[
    JsonScalar,
    list["JsonValue"],
    dict[str, "JsonValue"],
]
JsonObject: TypeAlias = dict[str, JsonValue]

SCHEMA = "org.quitetall.lamquant.plan-projection/v1"
PROJECTION_PLANNED = "planned"
PROJECTION_PROGRESS = "progress"
PROJECTION_ARTIFACT = "artifact"
PROJECTION_RECEIPT = "receipt"
PROJECTION_FAILURE = "failure"
PROJECTION_DIAGNOSTIC = "diagnostic"
CANONICAL_OPERATION_IDS = [
    "encode_lma",
    "encode_lml_siblings",
    "decode",
    "verify",
    "info",
    "stats",
    "bench",
    "archive",
    "extract",
    "list_archive",
    "verify_archive",
    "verify_manifest",
    "export_csv",
    "export_npy",
    "export_raw",
    "recover",
    "diff",
    "train_encoder",
    "train_snn",
    "train_tnn",
    "train_resume",
    "eagle_quick",
    "eagle_full",
    "eagle_bench",
    "eagle_lqs_l",
    "eagle_lqs_c",
    "eagle_lqs_m",
    "eagle_lqs_a",
    "eagle_perf",
    "eagle_rd",
    "eagle_h2h",
    "test_conformance",
    "test_full",
    "test_paranoid",
    "test_codec",
    "setup_pip",
    "setup_extras",
    "setup_cargo",
    "setup_musl",
    "setup_windows",
    "gui",
    "viz_lamquant-gui",
    "viz_eeglab",
    "viz_mne",
    "viz_legacy_OpenBCIGUI",
    "viz_legacy_BVAnalyzer",
    "viz_legacy_besa",
    "viz_OpenBCIGUI",
    "viz_BVAnalyzer",
    "viz_besa",
    "viz_install_lamquant_gui",
    "viz_install_mne",
    "viz_install_scope_tui",
    "viz_install_bottom",
    "viz_install_television",
    "viz_install_csvlens",
    "viz_install_gitui",
    "viz_uninstall_lamquant_gui",
    "viz_uninstall_mne",
    "viz_uninstall_scope_tui",
    "viz_uninstall_bottom",
    "viz_uninstall_television",
    "viz_uninstall_csvlens",
    "viz_uninstall_gitui",
    "cockpit_reset",
    "cockpit_checkpoints",
    "cockpit_metrics",
    "fw_list_devices",
    "fw_build_rp2350",
    "fw_build_nrf54l15",
    "fw_build_esp32p4",
    "fw_build_stm32n6",
    "fw_flash_rp2350",
    "fw_flash_nrf54l15",
    "fw_flash_esp32p4",
    "fw_flash_stm32n6",
    "fw_size_rp2350",
    "fw_size_stm32n6",
    "fw_size_esp32p4",
    "fw_size_nrf54l15",
    "fw_check_rp2350",
    "fw_check_stm32n6",
    "fw_check_esp32p4",
    "fw_check_nrf54l15",
    "fw_export",
    "fw_legacy_esp32s3",
    "cockpit_jobs",
    "cockpit_export",
    "syscheck_py",
    "cockpit_data_prep",
    "cockpit_train_encoder",
    "cockpit_train_snn",
    "cockpit_train_oracle",
    "setup_install_lml",
    "setup_install_eagle",
    "setup_install_lqt",
]
CANONICAL_OPERATION_IDS_SET = set(CANONICAL_OPERATION_IDS)
VALID_PROJECTIONS = frozenset(
    {
        PROJECTION_PLANNED,
        PROJECTION_PROGRESS,
        PROJECTION_ARTIFACT,
        PROJECTION_RECEIPT,
        PROJECTION_FAILURE,
        PROJECTION_DIAGNOSTIC,
    }
)
_HEX_256 = re.compile(r"^[0-9a-f]{64}$")
MAX_SAFE_INTEGER = 9_007_199_254_740_991
_COMMON_FIELDS = {"schema", "observed_at_ms", "plan", "projection"}
_VARIANT_FIELDS = {
    PROJECTION_PLANNED: {"operation", "total_nodes", "total_work"},
    PROJECTION_PROGRESS: {"node_id", "current", "total", "message"},
    PROJECTION_ARTIFACT: {"node_id", "artifact"},
    PROJECTION_RECEIPT: {"receipt", "message"},
    PROJECTION_FAILURE: {"receipt", "failure", "cancelled"},
    PROJECTION_DIAGNOSTIC: {"node_id", "level", "message"},
}
_ARTIFACT_FIELDS = {
    "path",
    "success",
    "elapsed_ms",
    "compression_ratio",
    "bytes_in",
    "bytes_out",
    "samples",
    "duration_seconds",
    "channel_count",
    "sample_rate_hz",
    "sha256",
    "window_count",
}
_RECEIPT_FIELDS = {
    "invocation_id",
    "graph_id",
    "plan_id",
    "realm",
    "completed_node_ids",
    "attempts",
    "committed_transactions",
    "gaps",
}
_ATTEMPT_FIELDS = {
    "step_id",
    "node_ids",
    "kernel_id",
    "implementation_id",
    "attempts",
    "kernel_succeeded",
    "completed",
}
_GAP_FIELDS = {
    "step_id",
    "node_ids",
    "output_index",
    "offset",
    "length",
    "domain",
    "code",
}
_FAILURE_FIELDS = {"domain", "code", "message", "retryable"}


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass(frozen=True)
class PlanIdentity:
    graph_id: str
    plan_id: str
    invocation_id: str

    def validate(self) -> None:
        for name, value in (
            ("graph_id", self.graph_id),
            ("plan_id", self.plan_id),
            ("invocation_id", self.invocation_id),
        ):
            if not isinstance(value, str) or not _HEX_256.fullmatch(value):
                raise ValueError(f"PlanIdentity.{name} must be lower-case 256-bit hex")

    def as_dict(self) -> dict[str, str]:
        return {
            "graph_id": self.graph_id,
            "plan_id": self.plan_id,
            "invocation_id": self.invocation_id,
        }

    @classmethod
    def from_mapping(
        cls, value: object, *, allow_additional: bool = False
    ) -> "PlanIdentity":
        if not isinstance(value, dict):
            raise ValueError("PlanProjection.plan must be an object")
        if not allow_additional:
            _require_exact_fields(
                value,
                {"graph_id", "plan_id", "invocation_id"},
                "PlanProjection.plan",
            )
        try:
            identity = cls(
                graph_id=value["graph_id"],
                plan_id=value["plan_id"],
                invocation_id=value["invocation_id"],
            )
        except (KeyError, TypeError) as error:
            raise ValueError("PlanProjection.plan is incomplete") from error
        identity.validate()
        return identity

    @classmethod
    def from_environment(cls) -> "PlanIdentity":
        try:
            identity = cls(
                graph_id=os.environ["LAMQUANT_GRAPH_ID"],
                plan_id=os.environ["LAMQUANT_PLAN_ID"],
                invocation_id=os.environ["LAMQUANT_INVOCATION_ID"],
            )
        except KeyError as error:
            raise ValueError(
                f"plan projection emission requires environment variable {error.args[0]}"
            ) from error
        identity.validate()
        return identity


@dataclass(frozen=True)
class PlannedPayload:
    operation: str
    total_nodes: int
    total_work: Optional[int]


@dataclass(frozen=True)
class ProgressPayload:
    node_id: int
    current: int
    total: int
    message: str


@dataclass(frozen=True)
class ArtifactPayload:
    node_id: int
    artifact: JsonObject


@dataclass(frozen=True)
class ReceiptPayload:
    receipt: JsonObject
    message: str


@dataclass(frozen=True)
class FailurePayload:
    receipt: JsonObject
    failure: JsonObject
    cancelled: bool


@dataclass(frozen=True)
class DiagnosticPayload:
    node_id: Optional[int]
    level: str
    message: str


ProjectionPayload: TypeAlias = Union[
    PlannedPayload,
    ProgressPayload,
    ArtifactPayload,
    ReceiptPayload,
    FailurePayload,
    DiagnosticPayload,
]


@dataclass
class PlanProjection:
    schema: str
    observed_at_ms: int
    plan: PlanIdentity
    projection: str
    payload: ProjectionPayload
    raw: JsonObject = field(default_factory=dict)


def _emit(
    projection: str,
    fields: JsonObject,
    identity: Optional[PlanIdentity],
) -> None:
    plan = identity or PlanIdentity.from_environment()
    plan.validate()
    payload = {
        "schema": SCHEMA,
        "observed_at_ms": _now_ms(),
        "plan": plan.as_dict(),
        "projection": projection,
        **fields,
    }
    rendered = json.dumps(payload, separators=(",", ":"))
    parse_line(rendered)
    sys.stdout.write(rendered + "\n")
    sys.stdout.flush()


def _is_integer(value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and -MAX_SAFE_INTEGER <= value <= MAX_SAFE_INTEGER
    )


def _is_nonnegative_integer(value: object) -> bool:
    return _is_integer(value) and value >= 0


def _is_nonnegative_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _require_exact_fields(
    value: JsonObject, allowed: set[str], label: str
) -> None:
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise ValueError(f"{label} has unknown fields {unexpected}")


def _require_string(value: object, label: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        qualifier = "non-empty " if nonempty else ""
        raise ValueError(f"{label} must be {qualifier}string")
    return value


def _validate_artifact(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("artifact must be an object")
    _require_exact_fields(value, _ARTIFACT_FIELDS, "artifact")
    for field in ("path", "success", "elapsed_ms"):
        if field not in value:
            raise ValueError(f"artifact missing {field}")
    _require_string(value["path"], "artifact.path")
    if not isinstance(value["success"], bool):
        raise ValueError("artifact.success must be boolean")
    if not _is_nonnegative_integer(value["elapsed_ms"]):
        raise ValueError("artifact.elapsed_ms must be nonnegative integer")
    if ("bytes_in" in value) != ("bytes_out" in value):
        raise ValueError("artifact byte telemetry must be complete or absent")
    for field in (
        "bytes_in",
        "bytes_out",
        "samples",
        "channel_count",
        "window_count",
    ):
        if field in value and not _is_nonnegative_integer(value[field]):
            raise ValueError(f"artifact.{field} must be nonnegative integer")
    for field in ("compression_ratio", "duration_seconds", "sample_rate_hz"):
        if field in value and not _is_nonnegative_number(value[field]):
            raise ValueError(f"artifact.{field} must be finite nonnegative number")
    if "sha256" in value:
        _require_string(value["sha256"], "artifact.sha256")


def _validate_integer_array(value: object, label: str) -> None:
    if not isinstance(value, list) or any(
        not _is_nonnegative_integer(item) for item in value
    ):
        raise ValueError(f"{label} must be array of nonnegative integers")


def _validate_receipt(value: object, plan: PlanIdentity) -> None:
    if not isinstance(value, dict):
        raise ValueError("receipt must be an object")
    _require_exact_fields(value, _RECEIPT_FIELDS, "receipt")
    missing = sorted(_RECEIPT_FIELDS - set(value))
    if missing:
        raise ValueError(f"receipt missing {missing}")
    receipt_identity = PlanIdentity.from_mapping(value, allow_additional=True)
    if receipt_identity != plan:
        raise ValueError("terminal receipt identity does not match projection plan")
    if value["realm"] not in {"mcu-aot", "host-stream", "blut-durable"}:
        raise ValueError("receipt.realm is unknown")
    _validate_integer_array(value["completed_node_ids"], "receipt.completed_node_ids")
    if not isinstance(value["committed_transactions"], list) or any(
        not isinstance(item, str) for item in value["committed_transactions"]
    ):
        raise ValueError("receipt.committed_transactions must be string array")
    if not isinstance(value["attempts"], list):
        raise ValueError("receipt.attempts must be array")
    for index, attempt in enumerate(value["attempts"]):
        label = f"receipt.attempts[{index}]"
        if not isinstance(attempt, dict):
            raise ValueError(f"{label} must be object")
        _require_exact_fields(attempt, _ATTEMPT_FIELDS, label)
        missing = sorted(_ATTEMPT_FIELDS - set(attempt))
        if missing:
            raise ValueError(f"{label} missing {missing}")
        for field in ("step_id", "kernel_id", "attempts"):
            if not _is_nonnegative_integer(attempt[field]):
                raise ValueError(f"{label}.{field} must be nonnegative integer")
        _validate_integer_array(attempt["node_ids"], f"{label}.node_ids")
        if not isinstance(attempt["kernel_succeeded"], bool) or not isinstance(
            attempt["completed"], bool
        ):
            raise ValueError(f"{label} status fields must be boolean")
        if not isinstance(attempt["implementation_id"], str) or not _HEX_256.fullmatch(
            attempt["implementation_id"]
        ):
            raise ValueError(f"{label}.implementation_id must be lower-case 256-bit hex")
    if not isinstance(value["gaps"], list):
        raise ValueError("receipt.gaps must be array")
    for index, gap in enumerate(value["gaps"]):
        label = f"receipt.gaps[{index}]"
        if not isinstance(gap, dict):
            raise ValueError(f"{label} must be object")
        _require_exact_fields(gap, _GAP_FIELDS, label)
        required = _GAP_FIELDS - {"length"}
        missing = sorted(required - set(gap))
        if missing:
            raise ValueError(f"{label} missing {missing}")
        for field in ("step_id", "output_index", "offset"):
            if not _is_nonnegative_integer(gap[field]):
                raise ValueError(f"{label}.{field} must be nonnegative integer")
        if "length" in gap and (
            not _is_integer(gap["length"]) or gap["length"] < 1
        ):
            raise ValueError(f"{label}.length must be positive integer")
        _validate_integer_array(gap["node_ids"], f"{label}.node_ids")
        _require_string(gap["domain"], f"{label}.domain", nonempty=True)
        _require_string(gap["code"], f"{label}.code", nonempty=True)


def _validate_failure(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("failure must be an object")
    _require_exact_fields(value, _FAILURE_FIELDS, "failure")
    missing = sorted(_FAILURE_FIELDS - set(value))
    if missing:
        raise ValueError(f"failure missing {missing}")
    _require_string(value["domain"], "failure.domain", nonempty=True)
    _require_string(value["code"], "failure.code", nonempty=True)
    _require_string(value["message"], "failure.message")
    if not isinstance(value["retryable"], bool):
        raise ValueError("failure.retryable must be boolean")


def planned(
    operation: str,
    total_nodes: int = 1,
    total_work: Optional[int] = None,
    *,
    identity: Optional[PlanIdentity] = None,
) -> None:
    fields: JsonObject = {
        "operation": str(operation),
        "total_nodes": int(total_nodes),
    }
    if total_work is not None:
        fields["total_work"] = int(total_work)
    _emit(PROJECTION_PLANNED, fields, identity)


def progress(
    current: int,
    total: int,
    message: str,
    *,
    node_id: int = 0,
    identity: Optional[PlanIdentity] = None,
) -> None:
    _emit(
        PROJECTION_PROGRESS,
        {
            "node_id": int(node_id),
            "current": int(current),
            "total": int(total),
            "message": str(message),
        },
        identity,
    )


def artifact(
    path: str,
    success: bool,
    elapsed_ms: int,
    *,
    node_id: int = 0,
    identity: Optional[PlanIdentity] = None,
    **telemetry: JsonValue,
) -> None:
    value: JsonObject = {
        "path": str(path),
        "success": bool(success),
        "elapsed_ms": int(elapsed_ms),
    }
    value.update({name: item for name, item in telemetry.items() if item is not None})
    _emit(
        PROJECTION_ARTIFACT,
        {"node_id": int(node_id), "artifact": value},
        identity,
    )


def diagnostic(
    message: str,
    *,
    level: str = "info",
    node_id: Optional[int] = 0,
    identity: Optional[PlanIdentity] = None,
) -> None:
    fields: JsonObject = {"level": level, "message": str(message)}
    if node_id is not None:
        fields["node_id"] = int(node_id)
    _emit(PROJECTION_DIAGNOSTIC, fields, identity)


def parse_line(line: str) -> PlanProjection:
    try:
        data = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"PlanProjection JSON parse: {error}") from error
    if not isinstance(data, dict):
        raise ValueError("PlanProjection must be a JSON object")
    data = cast(JsonObject, data)
    if data.get("schema") != SCHEMA:
        raise ValueError(f"unsupported PlanProjection schema: {data.get('schema')!r}")
    observed_at_ms = data.get("observed_at_ms")
    if not _is_nonnegative_integer(observed_at_ms):
        raise ValueError(
            "PlanProjection.observed_at_ms must be nonnegative safe integer"
        )
    plan = PlanIdentity.from_mapping(data.get("plan"))
    projection = data.get("projection")
    if projection not in VALID_PROJECTIONS:
        raise ValueError(f"unknown PlanProjection projection: {projection!r}")
    _require_exact_fields(
        data,
        _COMMON_FIELDS | _VARIANT_FIELDS[projection],
        f"PlanProjection {projection}",
    )

    required = {
        PROJECTION_PLANNED: ("operation", "total_nodes"),
        PROJECTION_PROGRESS: ("node_id", "current", "total", "message"),
        PROJECTION_ARTIFACT: ("node_id", "artifact"),
        PROJECTION_RECEIPT: ("receipt", "message"),
        PROJECTION_FAILURE: ("receipt", "failure", "cancelled"),
        PROJECTION_DIAGNOSTIC: ("level", "message"),
    }[projection]
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"PlanProjection {projection} missing {missing}")

    if projection == PROJECTION_PLANNED:
        if not isinstance(data["operation"], str):
            raise ValueError("invalid planned projection: operation must be string")
        if data["operation"] not in CANONICAL_OPERATION_IDS_SET:
            raise ValueError(
                f"invalid planned projection: unknown operation {data['operation']!r}"
            )
        if (
            not _is_integer(data["total_nodes"])
            or data["total_nodes"] < 1
            or (
                "total_work" in data
                and not _is_nonnegative_integer(data["total_work"])
            )
        ):
            raise ValueError("invalid planned projection")
    elif projection == PROJECTION_PROGRESS:
        node_id, current, total = data["node_id"], data["current"], data["total"]
        if (
            not _is_nonnegative_integer(node_id)
            or not _is_integer(current)
            or not _is_integer(total)
            or total < 1
            or current < 0
            or current > total
            or not isinstance(data["message"], str)
        ):
            raise ValueError("invalid progress projection")
    elif projection == PROJECTION_ARTIFACT:
        if not _is_nonnegative_integer(data["node_id"]):
            raise ValueError("artifact.node_id must be nonnegative integer")
        _validate_artifact(data["artifact"])
    elif projection in (PROJECTION_RECEIPT, PROJECTION_FAILURE):
        _validate_receipt(data["receipt"], plan)
        if projection == PROJECTION_RECEIPT:
            _require_string(data["message"], "receipt message")
        else:
            _validate_failure(data["failure"])
            if not isinstance(data["cancelled"], bool):
                raise ValueError("failure.cancelled must be boolean")
    else:
        if (
            data["level"] not in {"info", "warning", "error"}
            or not isinstance(data["message"], str)
            or (
                "node_id" in data
                and not _is_nonnegative_integer(data["node_id"])
            )
        ):
            raise ValueError("invalid diagnostic projection")

    return PlanProjection(
        schema=SCHEMA,
        observed_at_ms=cast(int, observed_at_ms),
        plan=plan,
        projection=cast(str, projection),
        payload=_typed_payload(data, cast(str, projection)),
        raw=data,
    )


def _typed_payload(data: JsonObject, projection: str) -> ProjectionPayload:
    if projection == PROJECTION_PLANNED:
        return PlannedPayload(
            operation=cast(str, data["operation"]),
            total_nodes=cast(int, data["total_nodes"]),
            total_work=cast(Optional[int], data.get("total_work")),
        )
    if projection == PROJECTION_PROGRESS:
        return ProgressPayload(
            node_id=cast(int, data["node_id"]),
            current=cast(int, data["current"]),
            total=cast(int, data["total"]),
            message=cast(str, data["message"]),
        )
    if projection == PROJECTION_ARTIFACT:
        return ArtifactPayload(
            node_id=cast(int, data["node_id"]),
            artifact=cast(JsonObject, data["artifact"]),
        )
    if projection == PROJECTION_RECEIPT:
        return ReceiptPayload(
            receipt=cast(JsonObject, data["receipt"]),
            message=cast(str, data["message"]),
        )
    if projection == PROJECTION_FAILURE:
        return FailurePayload(
            receipt=cast(JsonObject, data["receipt"]),
            failure=cast(JsonObject, data["failure"]),
            cancelled=cast(bool, data["cancelled"]),
        )
    return DiagnosticPayload(
        node_id=cast(Optional[int], data.get("node_id")),
        level=cast(str, data["level"]),
        message=cast(str, data["message"]),
    )


def parse_lines(stream: Iterable[str]) -> Iterable[PlanProjection]:
    for line in stream:
        line = line.rstrip("\n")
        if not line:
            continue
        try:
            yield parse_line(line)
        except ValueError as error:
            sys.stderr.write(
                f"plan_projection_emit: dropping malformed projection line: {error}\n"
            )


def _check_round_trip() -> int:
    from lamquant_codec._paths import REPO_ROOT

    candidates = [
        REPO_ROOT.parent
        / "crates"
        / "lamquant-ops"
        / "tests"
        / "fixtures"
        / "plan-projections-sample.jsonl",
        REPO_ROOT
        / "crates"
        / "lamquant-ops"
        / "tests"
        / "fixtures"
        / "plan-projections-sample.jsonl",
    ]
    fixture = next((path for path in candidates if path.exists()), None)
    if fixture is None:
        print(f"fixture not found; checked {candidates}", file=sys.stderr)
        return 1
    invalid_fixture = fixture.with_name("plan-projections-invalid.jsonl")
    if not invalid_fixture.exists():
        print(f"invalid fixture not found: {invalid_fixture}", file=sys.stderr)
        return 1
    failures = 0
    count = 0
    for index, raw in enumerate(fixture.read_text().splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        try:
            parse_line(raw)
            count += 1
        except ValueError as error:
            print(f"line {index}: {error}", file=sys.stderr)
            failures += 1
    if failures:
        print(
            f"plan_projection_emit --check: {failures} failure(s)",
            file=sys.stderr,
        )
        return 1
    rejected = 0
    for index, raw in enumerate(invalid_fixture.read_text().splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        try:
            parse_line(raw)
        except ValueError:
            rejected += 1
        else:
            print(
                f"invalid line {index}: parser accepted invalid projection",
                file=sys.stderr,
            )
            failures += 1
    if failures:
        return 1
    print(
        f"plan_projection_emit --check: {count} lines OK; "
        f"{rejected} invalid lines rejected"
    )
    return 0


def _main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lamquant_codec.cli.plan_projection_emit"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate canonical cross-language fixtures and exit.",
    )
    args = parser.parse_args()
    if args.check:
        return _check_round_trip()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
