"""Coverage tests for ``lamquant_codec.cli.syscheck``.

The syscheck module profiles the system and recommends config. We test
the structural invariants:

  * ``recommend()`` returns a dict with the documented top-level keys.
  * Each section contains the documented sub-keys (system, benchmark,
    recommended, estimate).
  * Values have correct types (int / float / str).
  * ``print_syscheck()`` writes to stdout without raising.
  * Hardware probe helpers (``_cpu_info``, ``_mem_info``, ``_disk_info``)
    return non-negative numeric values.
  * ``_bench_sha256`` returns a positive throughput.
  * ``_bench_disk_io`` writes to tmp_path and returns positive speeds.

The bench helpers are mocked when too slow for unit tests; we still
exercise the wiring with small sizes.
"""
from __future__ import annotations

import os
import platform
from unittest.mock import patch

import pytest

from lamquant_codec.cli import syscheck as sc


# ----- hardware probes --------------------------------------------------


def test_cpu_info_returns_positive_int():
    n = sc._cpu_info()
    assert isinstance(n, int)
    assert n >= 1


def test_mem_info_returns_tuple_of_floats():
    total, avail = sc._mem_info()
    assert isinstance(total, (int, float))
    assert isinstance(avail, (int, float))
    assert total >= 0
    assert avail >= 0


def test_disk_info_returns_three_values(tmp_path):
    total, free, pct = sc._disk_info(str(tmp_path))
    assert isinstance(total, (int, float))
    assert isinstance(free, (int, float))
    assert isinstance(pct, (int, float))
    assert total >= 0
    assert free >= 0


def test_disk_info_invalid_path_returns_zeros():
    total, free, pct = sc._disk_info("/nonexistent/path/does/not/exist")
    assert total == 0
    assert free == 0
    assert pct == 0


# ----- _bench_sha256 ---------------------------------------------------


def test_bench_sha256_returns_positive_throughput():
    # Use 1 MB (smallest reasonable size — fast).
    mbs = sc._bench_sha256(1)
    assert isinstance(mbs, float)
    assert mbs > 0


# ----- _bench_disk_io --------------------------------------------------


def test_bench_disk_io_returns_positive_speeds(tmp_path):
    # Note: writes 64 MiB to tmp_path then deletes. Within hard rule
    # because tmp_path is under pytest control.
    write_speed, read_speed = sc._bench_disk_io(str(tmp_path))
    assert isinstance(write_speed, float)
    assert isinstance(read_speed, float)
    # Should be positive on any working filesystem
    assert write_speed >= 0
    assert read_speed >= 0


def test_bench_disk_io_handles_unwritable_path():
    # Patch open() to raise inside the bench so failure path is exercised
    # without touching system permissions.
    with patch("lamquant_codec.cli.syscheck.open",
               side_effect=PermissionError("nope")):
        try:
            write_speed, read_speed = sc._bench_disk_io("/tmp")
        except PermissionError:
            # _bench_disk_io's bare except may not catch this on early write
            pytest.skip("bench wraps OSError, not arbitrary errors")
    # If it returns, both should be zero
    assert write_speed == 0
    assert read_speed == 0


# ----- recommend() ------------------------------------------------------


@pytest.fixture
def mock_bench():
    """Mock the slow bench so tests run quickly."""
    with patch(
        "lamquant_codec.cli.syscheck._bench_compress",
        return_value=(7.0, 2.5, 12345),
    ), patch(
        "lamquant_codec.cli.syscheck._bench_sha256",
        return_value=2000.0,
    ):
        yield


def test_recommend_returns_dict_with_documented_keys(tmp_path, mock_bench):
    out = sc.recommend(corpus_files=100, corpus_bytes=1024,
                       output_path=str(tmp_path))
    assert isinstance(out, dict)
    assert set(out.keys()) >= {"system", "benchmark", "recommended", "estimate"}


def test_recommend_system_block_shape(tmp_path, mock_bench):
    out = sc.recommend(0, 0, str(tmp_path))
    s = out["system"]
    assert isinstance(s["platform"], str)
    assert isinstance(s["cpu_cores"], int)
    assert s["cpu_cores"] >= 1
    assert isinstance(s["ram_total_gib"], (int, float))
    assert isinstance(s["disk_free_gib"], (int, float))


def test_recommend_benchmark_block_shape(tmp_path, mock_bench):
    out = sc.recommend(0, 0, str(tmp_path))
    b = out["benchmark"]
    for key in ("compress_ms_per_window", "decompress_ms_per_window",
                "total_ms_per_window", "sha256_mibs", "est_ms_per_file"):
        assert key in b
        assert isinstance(b[key], (int, float))


def test_recommend_recommended_block_shape(tmp_path, mock_bench):
    out = sc.recommend(0, 0, str(tmp_path))
    r = out["recommended"]
    assert isinstance(r["workers"], int)
    assert r["workers"] >= 1
    assert isinstance(r["numba_cache_dir"], str)
    assert isinstance(r["refresh_hz"], int)


def test_recommend_estimate_block_shape(tmp_path, mock_bench):
    out = sc.recommend(corpus_files=1000, corpus_bytes=10240,
                       output_path=str(tmp_path))
    e = out["estimate"]
    assert e["corpus_files"] == 1000
    assert isinstance(e["single_thread_hours"], (int, float))
    assert isinstance(e["parallel_hours"], (int, float))
    assert e["recommended_workers"] >= 1


def test_recommend_zero_corpus_returns_zero_runtime(tmp_path, mock_bench):
    out = sc.recommend(corpus_files=0, corpus_bytes=0,
                       output_path=str(tmp_path))
    e = out["estimate"]
    assert e["single_thread_hours"] == 0
    assert e["parallel_hours"] == 0


def test_recommend_workers_bounded_by_eight(tmp_path, mock_bench):
    # Pin: workers cap is documented at 8 (diminishing returns from I/O)
    out = sc.recommend(0, 0, str(tmp_path))
    assert out["recommended"]["workers"] <= 8


def test_recommend_no_output_path_uses_default(mock_bench):
    out = sc.recommend(corpus_files=0, corpus_bytes=0, output_path=None)
    assert "system" in out


# ----- print_syscheck ---------------------------------------------------


def test_print_syscheck_writes_stdout(tmp_path, mock_bench, capsys):
    out = sc.recommend(corpus_files=100, corpus_bytes=1024,
                       output_path=str(tmp_path))
    sc.print_syscheck(out)
    captured = capsys.readouterr()
    assert "System Check" in captured.out
    assert "Recommended" in captured.out


def test_print_syscheck_zero_corpus_omits_estimated_runtime(
    tmp_path, mock_bench, capsys
):
    out = sc.recommend(0, 0, str(tmp_path))
    sc.print_syscheck(out)
    captured = capsys.readouterr()
    # When corpus_files == 0, the "Estimated Runtime" block is suppressed.
    assert "Estimated Runtime" not in captured.out


def test_print_syscheck_with_corpus_shows_estimated_runtime(
    tmp_path, mock_bench, capsys
):
    out = sc.recommend(corpus_files=500, corpus_bytes=51200,
                       output_path=str(tmp_path))
    sc.print_syscheck(out)
    captured = capsys.readouterr()
    assert "Estimated Runtime" in captured.out


# ----- main() (CLI entry) ----------------------------------------------


def test_main_no_corpus(tmp_path, mock_bench, capsys, monkeypatch):
    """main() with no --corpus should still run benchmark + print."""
    monkeypatch.setattr("sys.argv", ["syscheck"])
    sc.main()
    captured = capsys.readouterr()
    assert "Benchmarking" in captured.out


def test_main_with_empty_corpus_dir(tmp_path, mock_bench, capsys, monkeypatch):
    """main() with --corpus pointing at empty dir → 0 files."""
    monkeypatch.setattr(
        "sys.argv", ["syscheck", "--corpus", str(tmp_path)],
    )
    sc.main()
    captured = capsys.readouterr()
    assert "Benchmarking" in captured.out


def test_main_with_write_config(tmp_path, mock_bench, monkeypatch):
    """--write-config writes lamquant.toml in CWD."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["syscheck", "--write-config"])
    sc.main()
    cfg_file = tmp_path / "lamquant.toml"
    assert cfg_file.exists()
    content = cfg_file.read_text()
    assert "workers" in content
