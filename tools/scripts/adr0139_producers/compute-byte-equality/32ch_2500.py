#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ADR 0139 P3 cross-backend byte-equality producer.

Emits ONE conformance-vector receipt as deterministic JSON on stdout. The vector
is this file's stem; all vector producers are byte-identical (one reviewed
source, several names). Each drives the real codec, encoding the vector with
`ComputeBackend::Firmware` and again with `ComputeBackend::Desktop`, and records
both SHA-256 digests. The `.lml` wire format is backend-independent by contract,
so any divergence is a wire-format change rather than an optimisation.
"""

import json
import subprocess
import sys
from pathlib import Path

PRODUCER_CONTRACT = "compute-byte-equality"

_VECTORS = ("1ch_100", "4ch_2500", "32ch_2500")
_SCHEMA = "lamquant.adr0139.compute-byte-equality-receipt/v1"


def _encode_both_backends(vector):
    """Run the real codec once per backend and return its reported digests."""
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "lamquant-lml-legacy",
            "--features",
            "legacy-encode",
            "--example",
            "backend_byte_equality",
            "--",
            vector,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"byte-equality encode failed for {vector}")
    return json.loads(completed.stdout)


def produce_evidence():
    """Encode this vector on both backends and return its equality receipt."""
    vector = Path(__file__).stem
    if vector not in _VECTORS:
        raise SystemExit(f"unknown conformance vector: {vector}")
    measured = _encode_both_backends(vector)
    firmware = measured["firmware_sha256"]
    desktop = measured["desktop_sha256"]
    receipt = {}
    receipt["schema"] = _SCHEMA
    receipt["case_id"] = vector
    receipt["status"] = "pass" if firmware == desktop else "fail"
    receipt["firmware_sha256"] = firmware
    receipt["desktop_sha256"] = desktop
    receipt["channels"] = measured["channels"]
    receipt["samples"] = measured["samples"]
    receipt["encoded_bytes"] = measured["firmware_bytes"]
    return receipt


def main():
    rendered = json.dumps(produce_evidence(), indent=2, sort_keys=True) + "\n"
    sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
