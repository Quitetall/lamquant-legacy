#!/usr/bin/env python3
"""
lamquant syscheck — automatic system profiling and configuration.

Benchmarks the current system and recommends optimal settings:
  - Worker count based on CPU cores and memory
  - Numba JIT cache location
  - I/O throughput estimate
  - Compression speed per core
  - Total estimated runtime for a given corpus

Usage:
    python -m lamquant_codec.cli.syscheck
    python -m lamquant_codec.cli.syscheck --corpus /data/tueg/
    python -m lamquant_codec.cli.syscheck --write-config
"""
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _cpu_info():
    cores_phys = os.cpu_count() or 1
    try:
        # Try to get physical cores (not hyperthreads)
        import multiprocessing
        cores_phys = multiprocessing.cpu_count()
    except Exception:
        pass
    return cores_phys


def _mem_info():
    """Available memory in GiB."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.total / (1024**3), vm.available / (1024**3)
    except ImportError:
        try:
            with open("/proc/meminfo") as f:
                lines = f.readlines()
            total = avail = 0
            for line in lines:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) / (1024**2)
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) / (1024**2)
            return total, avail
        except Exception:
            return 0, 0


def _disk_info(path: str):
    """Disk usage for a path."""
    try:
        u = shutil.disk_usage(path)
        return u.total / (1024**3), u.free / (1024**3), u.used / u.total * 100
    except Exception:
        return 0, 0, 0


def _bench_compress(n_iter: int = 20):
    """Benchmark compression speed. Returns ms/window."""
    import numpy as np
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba_cache"))
    from lamquant_codec.lossless import _compress_bytes, _decompress_bytes

    sig = np.random.RandomState(42).randn(21, 2500) * 5000
    # Warmup JIT
    _compress_bytes(sig)
    _decompress_bytes(_compress_bytes(sig))

    t0 = time.perf_counter()
    for _ in range(n_iter):
        c = _compress_bytes(sig)
    t_enc = (time.perf_counter() - t0) / n_iter * 1000

    t0 = time.perf_counter()
    for _ in range(n_iter):
        _decompress_bytes(c)
    t_dec = (time.perf_counter() - t0) / n_iter * 1000

    return t_enc, t_dec, len(c)


def _bench_sha256(size_mb: int = 50):
    """SHA-256 throughput in MiB/s."""
    import hashlib
    data = os.urandom(size_mb * 1024 * 1024)
    t0 = time.perf_counter()
    hashlib.sha256(data).hexdigest()
    elapsed = time.perf_counter() - t0
    return size_mb / elapsed


def _bench_disk_io(path: str):
    """Sequential write throughput in MiB/s."""
    test_file = os.path.join(path, ".lamquant_io_test")
    size = 64 * 1024 * 1024  # 64 MiB
    data = os.urandom(size)
    try:
        t0 = time.perf_counter()
        with open(test_file, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        write_speed = (size / (1024**2)) / (time.perf_counter() - t0)

        t0 = time.perf_counter()
        with open(test_file, "rb") as f:
            f.read()
        read_speed = (size / (1024**2)) / (time.perf_counter() - t0)
        return write_speed, read_speed
    except Exception:
        return 0, 0
    finally:
        try:
            os.unlink(test_file)
        except OSError:
            pass


def recommend(corpus_files: int = 0, corpus_bytes: int = 0,
              output_path: str = None):
    """Run benchmarks and return recommended config dict."""
    if output_path is None:
        output_path = tempfile.gettempdir()
    cores = _cpu_info()
    mem_total, mem_avail = _mem_info()
    disk_total, disk_free, disk_pct = _disk_info(output_path)

    # Benchmark
    t_enc, t_dec, comp_size = _bench_compress(30)
    sha_speed = _bench_sha256(50)

    # Worker recommendation:
    # Each worker needs ~500 MiB RAM for buffering a large EDF
    # Leave 2 GiB for OS + main process
    max_by_mem = max(1, int((mem_avail - 2.0) / 0.5))
    # Leave 1 core for OS
    max_by_cpu = max(1, cores - 1)
    # Don't exceed 8 — diminishing returns from I/O contention
    recommended_workers = min(max_by_mem, max_by_cpu, 8)

    # Time estimate
    # Per file: read(120ms) + compress(7ms * ~100 windows) + verify(2.5ms * ~100 windows) + SHA(~300ms)
    # ≈ 1.3s per average file single-threaded
    ms_per_window = t_enc + t_dec  # compress + verify
    avg_windows_per_file = 100  # typical TUEG
    ms_per_file = 120 + ms_per_window * avg_windows_per_file + 300
    if corpus_files > 0:
        total_s_single = corpus_files * ms_per_file / 1000
        total_s_parallel = total_s_single / recommended_workers
    else:
        total_s_single = total_s_parallel = 0

    return {
        "system": {
            "platform": platform.platform(),
            "cpu_cores": cores,
            "ram_total_gib": round(mem_total, 1),
            "ram_available_gib": round(mem_avail, 1),
            "disk_free_gib": round(disk_free, 1),
        },
        "benchmark": {
            "compress_ms_per_window": round(t_enc, 2),
            "decompress_ms_per_window": round(t_dec, 2),
            "total_ms_per_window": round(t_enc + t_dec, 2),
            "sha256_mibs": round(sha_speed, 0),
            "est_ms_per_file": round(ms_per_file, 0),
        },
        "recommended": {
            "workers": recommended_workers,
            "numba_cache_dir": os.path.join(tempfile.gettempdir(), "numba_cache"),
            "refresh_hz": 10,
            "verification": "standard",
        },
        "estimate": {
            "corpus_files": corpus_files,
            "single_thread_hours": round(total_s_single / 3600, 1),
            "parallel_hours": round(total_s_parallel / 3600, 1),
            "recommended_workers": recommended_workers,
        },
    }


def print_syscheck(rec: dict):
    """Print system check results."""
    s = rec["system"]
    b = rec["benchmark"]
    r = rec["recommended"]
    e = rec["estimate"]

    try:
        from lamquant_codec.cli.readout import C, B, _bytes
    except ImportError:
        class C:
            RST = DIM = CYN = GRN = RED = YEL = BLD = ""
        B = {"h": "-", "ok": "OK", "dot": "-"}
        _bytes = lambda n: f"{n} B"

    w = 72
    print()
    print(f"  {C.BLD}LamQuant System Check{C.RST}")
    print(f"  {C.DIM}{B['h'] * w}{C.RST}")
    print()

    print(f"  {C.DIM}Platform{C.RST}     {s['platform']}")
    print(f"  {C.DIM}CPU{C.RST}          {s['cpu_cores']} cores")
    print(f"  {C.DIM}RAM{C.RST}          {s['ram_total_gib']:.1f} GiB total, "
          f"{s['ram_available_gib']:.1f} GiB available")
    print(f"  {C.DIM}Disk{C.RST}         {s['disk_free_gib']:.1f} GiB free")
    print()

    print(f"  {C.DIM}Compress{C.RST}     {b['compress_ms_per_window']:.1f} ms/window  "
          f"({21*2500*2/1024/b['compress_ms_per_window']*1000:.0f} MiB/s)")
    print(f"  {C.DIM}Decompress{C.RST}   {b['decompress_ms_per_window']:.1f} ms/window  "
          f"({21*2500*2/1024/b['decompress_ms_per_window']*1000:.0f} MiB/s)")
    print(f"  {C.DIM}SHA-256{C.RST}      {b['sha256_mibs']:.0f} MiB/s")
    print(f"  {C.DIM}Per file{C.RST}     ~{b['est_ms_per_file']:.0f} ms  "
          f"(incl. read + compress + verify + SHA-256)")
    print()

    print(f"  {C.BLD}Recommended Configuration{C.RST}")
    print(f"  {C.DIM}{B['h'] * w}{C.RST}")
    print(f"  {C.DIM}Workers{C.RST}      {C.GRN}{r['workers']}{C.RST}  "
          f"{C.DIM}(limited by {'RAM' if r['workers'] < s['cpu_cores'] - 1 else 'CPU cores'}){C.RST}")
    print(f"  {C.DIM}JIT cache{C.RST}    {r['numba_cache_dir']}")
    print(f"  {C.DIM}Refresh{C.RST}      {r['refresh_hz']} Hz")

    if e["corpus_files"] > 0:
        print()
        print(f"  {C.BLD}Estimated Runtime{C.RST}")
        print(f"  {C.DIM}{B['h'] * w}{C.RST}")
        print(f"  {C.DIM}Files{C.RST}        {e['corpus_files']:,}")
        print(f"  {C.DIM}1 thread{C.RST}     {e['single_thread_hours']:.1f} hours")
        print(f"  {C.DIM}{r['workers']} workers{C.RST}    "
              f"{C.GRN}{e['parallel_hours']:.1f} hours{C.RST}")

    print()
    print(f"  {C.DIM}Apply: lamquant config --auto{C.RST}")
    print()


def main():
    import argparse
    p = argparse.ArgumentParser(prog="lamquant syscheck",
                                description="Profile system and recommend config")
    p.add_argument("--corpus", type=str, default=None,
                   help="Path to EDF corpus for runtime estimate")
    p.add_argument("--output", type=str, default=tempfile.gettempdir(),
                   help="Output path for disk check")
    p.add_argument("--write-config", action="store_true",
                   help="Write recommended config to lamquant.toml")
    args = p.parse_args()

    corpus_files = 0
    corpus_bytes = 0
    if args.corpus:
        import glob
        edfs = glob.glob(os.path.join(args.corpus, "**/*.edf"), recursive=True)
        corpus_files = len(edfs)
        if edfs:
            sample = edfs[:100]
            corpus_bytes = sum(os.path.getsize(f) for f in sample) * (len(edfs) / len(sample))
        else:
            corpus_bytes = 0

    print(f"\n  Benchmarking... ", end="", flush=True)
    rec = recommend(corpus_files, int(corpus_bytes), args.output)
    print("done.\n")

    print_syscheck(rec)

    if args.write_config:
        from lamquant_codec.cli.config import generate_default_config
        config_path = Path("lamquant.toml")
        content = generate_default_config()
        # Patch in recommended values
        content = content.replace("workers = 0", f"workers = {rec['recommended']['workers']}")
        config_path.write_text(content)
        print(f"  Config written: {config_path.resolve()}")
        print()


if __name__ == "__main__":
    main()
