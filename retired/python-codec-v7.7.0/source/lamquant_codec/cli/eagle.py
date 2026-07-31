"""
OpenHuman Eagle — Validation Suite for EEG Processing

Eagle is the open evaluation platform for EEG signal processing.
Any codec can be tested against standardized datasets, metrics, and
quality levels. Eagle is to EEG what MLPerf is to ML benchmarks.

The codec is the subject, not the user. Eagle examines a codec.
Every number carries provenance. Every result is reproducible.

Entry: oh eagle | lamquant.py → [4] Validate with Eagle
"""
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from lamquant_codec._paths import REPO_ROOT as _REPO

# Terminal (shared detection from terminal.py)
from lamquant_codec.cli.terminal import C as _TC, S as _S

DIM = _TC.DIM
GRN = _TC.GRN
RED = _TC.RED
YEL = _TC.YEL
BLD = _TC.BLD
RST = _TC.RST

H   = _S["h"]
OK  = _S["ok"]
NO  = _S["no"]
DOT = _S["dot"]

W = 72

def _clear():
    if sys.stdout.isatty():
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

def _input(msg="  > "):
    try:
        return input(msg).strip().lower()
    except (KeyboardInterrupt, EOFError):
        raise KeyboardInterrupt

def _wait():
    try:
        input(f"\n  {DIM}Press Enter...{RST}")
    except (KeyboardInterrupt, EOFError):
        pass

def _version():
    try:
        from lamquant_codec import __version__
        return __version__
    except Exception:
        return "0.0.0"

def _stub(name, phase=2):
    print(f"\n  {DIM}{name} — coming in Eagle Phase {phase}.{RST}")
    print(f"  {DIM}See decisions/0015-eagle-platform-spec.md for the roadmap.{RST}")
    _wait()


# ────────────────────────────────────────────────────────────────────
# Banner
# ────────────────────────────────────────────────────────────────────

_EAGLE_LOGO = r"""
  ███████╗ █████╗  ██████╗ ██╗     ███████╗
  ██╔════╝██╔══██╗██╔════╝ ██║     ██╔════╝
  █████╗  ███████║██║  ███╗██║     █████╗
  ██╔══╝  ██╔══██║██║   ██║██║     ██╔══╝
  ███████╗██║  ██║╚██████╔╝███████╗███████╗
  ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝
"""

_show_eagle_logo = True


def _print_banner():
    global _show_eagle_logo
    v = _version()

    if _show_eagle_logo:
        print(f"\n{DIM}{_EAGLE_LOGO}{RST}")
        print(f"       {DIM}OpenHuman Eagle {DOT} Validation Suite for EEG Processing{RST}")
        _show_eagle_logo = False
    else:
        print(f"\n  {BLD}OpenHuman Eagle{RST}  {DIM}Validation Suite{RST}")

    print()
    print(f"  {DIM}Eagle{RST}  v1.0.0   {DIM}LQS{RST}  v1.0   {DIM}Codec{RST}  LamQuant v{v}")

    # Subject codec panel
    from lamquant_codec.cli.box import Box
    box = Box(title=f"LamQuant {v}", width=58)
    box.line(f"Mode     neural (LMQ) + lossless (LML)")
    box.line(f"Target   LQS-L/C/M/A compliance")
    box.line(f"Status   {DIM}ready to test{RST}")

    # Indent the box
    for line in box.render().split("\n"):
        print(f"    {line}")

    print(f"\n  {DIM}{H * W}{RST}")


# ────────────────────────────────────────────────────────────────────
# Main menu
# ────────────────────────────────────────────────────────────────────

def _print_main():
    _print_banner()

    print(f"\n    {DIM}COMPLIANCE{RST}                          "
          f"{DIM}What is being verified?{RST}\n")
    for k, n, d in [
        ("1", "LQS compliance test",    "all four levels against holdout"),
        ("2", "Quick quality check",     "30-second sanity check"),
        ("3", "Targeted level",          "test one specific LQS level"),
    ]:
        print(f"  [{k}]  {n:<26s}  {DIM}{d}{RST}")

    print(f"\n    {DIM}BENCHMARKING{RST}                        "
          f"{DIM}How does it perform?{RST}\n")
    for k, n, d in [
        ("4", "Performance suite",       "latency p50/p95/p99, throughput"),
        ("5", "Rate-distortion sweep",   "quality vs CR curve"),
        ("6", "Head-to-head comparison", "against gzip, zstd, baselines"),
    ]:
        print(f"  [{k}]  {n:<26s}  {DIM}{d}{RST}")

    print(f"\n    {DIM}CLINICAL VALIDATION{RST}                 "
          f"{DIM}Is it safe for patients?{RST}\n")
    for k, n, d in [
        ("7", "Downstream tasks",        "seizure, sleep, pathology"),
        ("8", "Hallucination tests",     "detect generative fabrication"),
    ]:
        print(f"  [{k}]  {n:<26s}  {DIM}{d}{RST}")

    print(f"\n    {DIM}EXPLORATION{RST}\n")
    print(f"  [9]  {'Metrics explorer':<26s}  {DIM}drill into last run's metrics{RST}")

    print(f"\n    {DIM}REGISTRY{RST}\n")
    for k, n, d in [
        ("p", "Publish badge",           "signed compliance certificate"),
        ("r", "Leaderboard",             "current state of the field"),
    ]:
        print(f"  [{k}]  {n:<26s}  {DIM}{d}{RST}")

    print(f"\n  [{DIM}x{RST}] Export report    [{DIM}b{RST}] Back    [{DIM}q{RST}] Exit")
    print(f"\n  {DIM}{H * W}{RST}")
    print()


# ────────────────────────────────────────────────────────────────────
# LQS compliance
# ────────────────────────────────────────────────────────────────────

def _ensure_jit():
    if "NUMBA_CACHE_DIR" not in os.environ:
        nc = _REPO / ".numba_cache"
        nc.mkdir(exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = str(nc)


def _get_codec():
    """Return (encode, decode) callables for the current codec."""
    _ensure_jit()
    from lamquant_codec.lossless import _compress_bytes, _decompress_bytes
    import numpy as np
    def encode(sig):
        return _compress_bytes(sig.astype(np.float64))
    def decode(pkt):
        return np.round(_decompress_bytes(pkt)).astype(np.int16)
    return encode, decode


def _run_lqs(level: str, n_windows: int = 50):
    _clear()
    encode, decode = _get_codec()
    from lamquant_codec.lqs import run_compliance
    from lamquant_codec import __version__

    print(f"\n  {DIM}Running LQS-{level} compliance ({n_windows} windows)...{RST}")

    result = run_compliance(
        codec_encode=encode, codec_decode=decode, level=level,
        codec_name=f"OpenHuman LamQuant v{__version__}",
        dataset_name=f"synthetic ({n_windows} windows)",
    )

    print()
    print(result.badge())
    print()

    if result.passed:
        print(f"  {GRN}{OK}{RST} {BLD}LQS-{level} COMPLIANT{RST}")
    else:
        print(f"  {RED}{NO}{RST} {BLD}LQS-{level} FAILED{RST}  ({result.n_violations} violations)")
        for v in result.violations[:10]:
            print(f"    {DIM}{v}{RST}")

    print(f"\n  {DIM}CR: {result.mean_cr:.2f}:1  "
          f"PRD: {result.mean_prd:.2f}%  "
          f"R: {result.mean_r:.4f}  "
          f"SNR: {result.mean_snr_db:.1f} dB{RST}")
    print(f"  {DIM}Encode: {result.total_encode_ms / max(result.n_windows, 1):.1f}ms/window  "
          f"Wall: {result.wall_time_s:.1f}s{RST}")

    # Actions
    print(f"\n  [s] Paper snippet    [x] Export JSON    [Enter/b] Back")
    choice = _input()
    if choice == "s":
        _paper_snippet(result)
    elif choice == "x":
        path = Path(f"eagle_lqs_{level}_{int(time.time())}.json")
        path.write_text(json.dumps(result.to_dict(), indent=2, default=str))
        print(f"  {GRN}{OK}{RST} Exported: {path}")
        _wait()
    # b, Enter, or anything else → return to Eagle menu


def _run_all_lqs():
    for level in ("L", "C", "M", "A"):
        _run_lqs(level)
        print(f"\n  {DIM}[Enter] Next level    [b] Back to Eagle{RST}")
        if _input() in ("b", "q", "back", "quit"):
            return


def _run_quick():
    """Quick quality check — 10 windows, <30 seconds."""
    _run_lqs("L", n_windows=10)


def _run_targeted():
    _clear()
    print(f"\n  Select LQS level:\n")
    for k, n, d in [
        ("1", "LQS-L  Lossless",   "bit-exact, PRD=0%, R=1.0"),
        ("2", "LQS-C  Clinical",    "PRD<9%, R>0.95"),
        ("3", "LQS-M  Monitoring",  "PRD<20%, R>0.85"),
        ("4", "LQS-A  Alerting",    "PRD<40%, R>0.70"),
    ]:
        print(f"    [{k}]  {n:<22s}  {DIM}{d}{RST}")
    print(f"\n  [b] Back\n")
    choice = _input()
    level_map = {"1": "L", "2": "C", "3": "M", "4": "A"}
    if choice in level_map:
        _run_lqs(level_map[choice])


def _paper_snippet(result):
    """Generate a markdown table for a paper."""
    print(f"\n  {BLD}Paper snippet (copy this):{RST}\n")
    print(f"  ```markdown")
    print(f"  | Metric | Value |")
    print(f"  |--------|-------|")
    print(f"  | LQS Level | {result.level} ({('PASS' if result.passed else 'FAIL')}) |")
    print(f"  | Compression Ratio | {result.mean_cr:.1f}:1 |")
    print(f"  | PRD | {result.mean_prd:.2f}% |")
    print(f"  | Pearson R | {result.mean_r:.4f} |")
    print(f"  | SNR | {result.mean_snr_db:.1f} dB |")
    print(f"  | Encode latency | {result.total_encode_ms / max(result.n_windows, 1):.0f} ms/window |")
    print(f"  | Test windows | {result.n_windows} |")
    print(f"  | Codec | {result.codec_name} |")
    print(f"  | Standard | LQS v{result.version} |")
    print(f"  ```")
    _wait()


# ────────────────────────────────────────────────────────────────────
# Benchmarking
# ────────────────────────────────────────────────────────────────────

def _run_throughput():
    _clear()
    _ensure_jit()
    from lamquant_codec.lossless import _compress_bytes, _decompress_bytes
    import numpy as np

    sig = np.random.RandomState(42).randn(21, 2500) * 5000
    _compress_bytes(sig); _decompress_bytes(_compress_bytes(sig))  # warmup

    n = 200
    enc_times = []
    dec_times = []
    for _ in range(n):
        t0 = time.perf_counter()
        c = _compress_bytes(sig)
        enc_times.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        _decompress_bytes(c)
        dec_times.append((time.perf_counter() - t0) * 1000)

    enc_times.sort(); dec_times.sort()
    raw = 21 * 2500 * 2
    t_enc = sum(enc_times) / n / 1000
    t_dec = sum(dec_times) / n / 1000

    print(f"\n  {BLD}Codec Performance Suite{RST}  (21ch x 2500, {n} iterations)")
    print(f"  {DIM}{H * 60}{RST}")
    print(f"  Encode    {t_enc*1000:>6.1f} ms    {raw/t_enc/1e6:>6.0f} MiB/s")
    print(f"  Decode    {t_dec*1000:>6.1f} ms    {raw/t_dec/1e6:>6.0f} MiB/s")
    print(f"  CR        {raw/len(c):>6.2f}:1")

    print(f"\n  {BLD}Latency Distribution{RST}")
    print(f"  {DIM}{H * 60}{RST}")
    for label, times in [("Encode", enc_times), ("Decode", dec_times)]:
        p50 = times[len(times)//2]
        p95 = times[int(len(times)*0.95)]
        p99 = times[int(len(times)*0.99)]
        print(f"  {label}  p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms  max={times[-1]:.1f}ms")

    _wait()


def _run_rate_distortion():
    _clear()
    _ensure_jit()
    from lamquant_codec.lossless import _compress_bytes, _decompress_bytes
    from lamquant_codec.lqs import prd, pearson_r
    import numpy as np

    sig = np.random.RandomState(42).randn(21, 2500) * 5000
    raw = sig.size * 2

    print(f"\n  {BLD}Rate-Distortion Sweep{RST}  (noise_bits 0-16)")
    print(f"  {DIM}{H * 60}{RST}")
    print(f"  {'nb':>4s}  {'CR':>8s}  {'PRD':>8s}  {'R':>8s}  {'Mode':>10s}")
    print(f"  {DIM}{H * 42}{RST}")

    for nb in [0, 1, 2, 3, 4, 6, 8, 12, 16]:
        c = _compress_bytes(sig, noise_bits=nb)
        d = _decompress_bytes(c)
        si = np.round(sig).astype(np.int16)
        di = np.round(d).astype(np.int16)
        cr = raw / len(c)
        p = prd(si, di)
        r = pearson_r(si, di)
        mode = "lossless" if nb == 0 else f"nb={nb}"
        print(f"  {nb:>4d}  {cr:>8.2f}  {p:>7.2f}%  {r:>8.4f}  {mode:>10s}")

    _wait()


def _run_head_to_head():
    _clear()
    _ensure_jit()
    from lamquant_codec.lossless import _compress_bytes, _decompress_bytes
    from lamquant_codec.lqs import prd, pearson_r, _synthetic_signals
    import numpy as np

    signals = _synthetic_signals(n=20)
    encode, decode = _get_codec()

    print(f"\n  {BLD}Head-to-Head Comparison{RST}  (20 windows)")
    print(f"  {DIM}{H * W}{RST}\n")

    # Codecs to compare
    codecs = {
        "LamQuant LML": (encode, decode),
        "gzip -6": (
            lambda s: gzip.compress(s.astype(np.int16).tobytes(), compresslevel=6),
            lambda b: np.frombuffer(gzip.decompress(b), dtype=np.int16).reshape(21, -1),
        ),
    }

    try:
        import pyzstd
        codecs["zstd -3"] = (
            lambda s: pyzstd.compress(s.astype(np.int16).tobytes(), 3),
            lambda b: np.frombuffer(pyzstd.decompress(b), dtype=np.int16).reshape(21, -1),
        )
    except ImportError:
        pass

    results = {}
    for name, (enc, dec) in codecs.items():
        crs, prds, rs, lats = [], [], [], []
        for sig in signals:
            raw = sig.size * 2
            t0 = time.perf_counter()
            pkt = enc(sig)
            lat = (time.perf_counter() - t0) * 1000
            rec = dec(pkt)
            si = sig.astype(np.int16)
            ri = np.asarray(rec).astype(np.int16)
            crs.append(raw / len(pkt))
            prds.append(prd(si, ri))
            rs.append(pearson_r(si, ri))
            lats.append(lat)
        results[name] = {
            "cr": np.mean(crs), "prd": np.mean(prds),
            "r": np.mean(rs), "lat_p50": np.median(lats),
        }

    # Table
    col = 18
    header = f"  {'Metric':<20s}"
    for name in results:
        header += f"  {BLD}{name[:col]:<{col}s}{RST}"
    print(header)
    print(f"  {H * (20 + (col + 2) * len(results))}")

    for metric, fmt, best_fn in [
        ("CR", ".1f", max), ("PRD", ".2f", min), ("R", ".4f", max), ("Latency ms", ".1f", min),
    ]:
        key = {"CR": "cr", "PRD": "prd", "R": "r", "Latency ms": "lat_p50"}[metric]
        vals = [results[n][key] for n in results]
        best = best_fn(vals)
        row = f"  {metric:<20s}"
        for name in results:
            v = results[name][key]
            vs = f"{v:{fmt}}"
            if v == best:
                row += f"  {GRN}{vs:<{col}s}{RST}"
            else:
                row += f"  {vs:<{col}s}"
        print(row)

    print(f"\n  {DIM}Green = best in category{RST}")

    # Actions
    print(f"\n  [s] Paper snippet    [b] Back")
    if _input() == "s":
        print(f"\n  {BLD}Markdown table:{RST}\n")
        print(f"  | Codec | CR | PRD | R | Latency |")
        print(f"  |-------|-----|-----|---|---------|")
        for name, m in results.items():
            print(f"  | {name} | {m['cr']:.1f}:1 | {m['prd']:.2f}% | {m['r']:.4f} | {m['lat_p50']:.1f}ms |")
        _wait()


# ────────────────────────────────────────────────────────────────────
# Main loop
# ────────────────────────────────────────────────────────────────────

def eagle_main(args=None):
    while True:
        try:
            _clear()
            _print_main()

            choice = _input()

            if choice in ("b", "back", "q", "quit"):
                return 0

            dispatch = {
                "1": _run_all_lqs,
                "2": _run_quick,
                "3": _run_targeted,
                "4": _run_throughput,
                "5": _run_rate_distortion,
                "6": _run_head_to_head,
                "7": lambda: _stub("Downstream task preservation", 3),
                "8": lambda: _stub("Hallucination detection", 3),
                "9": lambda: _stub("Metrics explorer", 2),
                "p": lambda: _stub("Badge publishing", 4),
                "r": lambda: _stub("Public leaderboard", 4),
                "x": lambda: _stub("Report export", 2),
            }

            if choice in dispatch:
                dispatch[choice]()

        except KeyboardInterrupt:
            return 0
        except EOFError:
            return 0


if __name__ == "__main__":
    sys.exit(eagle_main())
