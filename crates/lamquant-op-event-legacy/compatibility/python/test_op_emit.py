"""Compatibility checks for the sequestered Python OpEvent binding."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("op_emit.py")
SPEC = importlib.util.spec_from_file_location("legacy_op_emit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
op_emit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = op_emit
SPEC.loader.exec_module(op_emit)


class OpEmitCompatibilityTests(unittest.TestCase):
    def test_check_uses_sequestered_fixture(self) -> None:
        self.assertEqual(op_emit._check_round_trip(), 0)

    def test_typed_view_preserves_filedone_telemetry(self) -> None:
        payload = {
            "type": "FileDone",
            "ts_ms": 1,
            "path": "recording.lml",
            "success": True,
            "ms": 2,
            "cr": 2.5,
            "bytes_in": 100,
            "bytes_out": 40,
            "samples": 25,
            "duration_s": 0.1,
            "n_channels": 1,
            "sample_rate": 250.0,
            "sha256": "00",
            "n_windows": 1,
        }
        event = op_emit.parse_line(json.dumps(payload))
        self.assertEqual(event.bytes_in, 100)
        self.assertEqual(event.bytes_out, 40)
        self.assertEqual(event.samples, 25)
        self.assertEqual(event.duration_s, 0.1)
        self.assertEqual(event.n_channels, 1)
        self.assertEqual(event.sample_rate, 250.0)
        self.assertEqual(event.sha256, "00")
        self.assertEqual(event.n_windows, 1)


if __name__ == "__main__":
    unittest.main()
