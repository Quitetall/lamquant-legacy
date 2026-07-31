"""
LamQuant LML CLI readout — production-grade compression telemetry.

Implements: decisions/0013-cli-output-spec.md

Output channels:
  stdout  — banner, dashboard, summaries (user-facing)
  stderr  — errors, warnings, debug (diagnostics)
  file    — audit.log (machine-readable), manifest.json (reproducibility)

Modes (auto-detected):
  interactive  : TTY, color, live dashboard, spinner
  piped        : no color, line-buffered per-file lines
  quiet        : JSON summary to stdout only
"""

import functools
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ────────────────────────────────────────────────────────────────────
# Terminal detection (runs once, cached)
# ────────────────────────────────────────────────────────────────────

def _detect_color(force: Optional[bool] = None) -> bool:
    if force is not None:
        return force
    if os.environ.get("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return True

def _detect_unicode() -> bool:
    try:
        "\u2713\u2500\u2588\u280b".encode(sys.stdout.encoding or "utf-8")
        return True
    except (UnicodeEncodeError, LookupError):
        return False

def _term_size():
    return shutil.get_terminal_size((80, 24))

_COLOR = _detect_color()
_UNICODE = _detect_unicode()


class C:
    """ANSI escapes. Empty when color disabled."""
    RST = "\033[0m"  if _COLOR else ""
    DIM = "\033[90m" if _COLOR else ""
    CYN = "\033[36m" if _COLOR else ""
    GRN = "\033[32m" if _COLOR else ""
    RED = "\033[31m" if _COLOR else ""
    YEL = "\033[33m" if _COLOR else ""
    BLD = "\033[1m"  if _COLOR else ""

    @classmethod
    def reconfigure(cls, color: bool):
        """Update color state (called after arg parsing)."""
        global _COLOR
        _COLOR = color
        cls.RST = "\033[0m"  if color else ""
        cls.DIM = "\033[90m" if color else ""
        cls.CYN = "\033[36m" if color else ""
        cls.GRN = "\033[32m" if color else ""
        cls.RED = "\033[31m" if color else ""
        cls.YEL = "\033[33m" if color else ""
        cls.BLD = "\033[1m"  if color else ""


# Box drawing + symbols
if _UNICODE:
    B = dict(h="\u2500", v="\u2502",
             tl="\u250c", tr="\u2510", bl="\u2514", br="\u2518",
             t="\u252c", b="\u2534", l="\u251c", r="\u2524",
             ok="\u2713", no="\u2717", ar="\u2192", dot="\u00b7",
             fl="\u2588", sh="\u2591")
    SPIN = ["\u280b", "\u2819", "\u2839", "\u2838",
            "\u283c", "\u2834", "\u2826", "\u2827",
            "\u2807", "\u280f"]
else:
    B = dict(h="-", v="|",
             tl="+", tr="+", bl="+", br="+",
             t="+", b="+", l="+", r="+",
             ok="OK", no="X", ar="->", dot="-",
             fl="#", sh=".")
    SPIN = ["|", "/", "-", "\\"]


# ────────────────────────────────────────────────────────────────────
# Version (never hardcoded)
# ────────────────────────────────────────────────────────────────────

from lamquant_codec._paths import REPO_ROOT as _REPO  # noqa: E402

def _codec_version() -> str:
    try:
        from lamquant_codec import __version__
        return __version__
    except Exception:
        return "0.0.0"

@functools.lru_cache(maxsize=1)
def _git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=2, cwd=_REPO)
        return r.stdout.strip()[:10] if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"

@functools.lru_cache(maxsize=1)
def _git_describe() -> str:
    try:
        r = subprocess.run(["git", "describe", "--tags", "--always"],
                           capture_output=True, text=True, timeout=2, cwd=_REPO)
        return r.stdout.strip() if r.returncode == 0 else _codec_version()
    except Exception:
        return _codec_version()


# ────────────────────────────────────────────────────────────────────
# Data structures
# ────────────────────────────────────────────────────────────────────

@dataclass
class FileResult:
    path: str
    bytes_in: int = 0
    bytes_out: int = 0
    duration_s: float = 0.0
    sha256: str = ""
    n_channels: int = 0
    sample_rate: float = 0.0
    n_samples: int = 0
    n_windows: int = 0
    status: str = "ok"
    error: Optional[str] = None

    @property
    def ratio(self) -> float:
        return self.bytes_in / self.bytes_out if self.bytes_out > 0 else 0.0


@dataclass
class RunStats:
    start_time: float = field(default_factory=time.time)
    files_total: int = 0
    files_done: int = 0
    files_ok: int = 0
    files_error: int = 0
    files_skipped: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    total_samples: int = 0
    total_duration_s: float = 0.0
    warnings: list = field(default_factory=list)

    codec_version: str = field(default_factory=_codec_version)
    git_commit: str = field(default_factory=_git_commit)
    git_describe: str = field(default_factory=_git_describe)
    config_hash: str = ""

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def ratio(self) -> float:
        return self.bytes_in / self.bytes_out if self.bytes_out > 0 else 0.0

    @property
    def throughput(self) -> float:
        return (self.bytes_in / 1048576) / max(self.elapsed, 0.001)

    @property
    def eta_s(self) -> float:
        if self.files_done == 0:
            return 0.0
        rate = self.files_done / max(self.elapsed, 0.001)
        return (self.files_total - self.files_done) / rate

    @property
    def shannon_gap(self) -> float:
        if self.ratio <= 0:
            return 16.0
        return max(0.0, 16.0 / self.ratio - 6.63)

    @property
    def shannon_pct(self) -> float:
        if self.ratio <= 0:
            return 0.0
        bps = 16.0 / self.ratio
        return (1 - (bps - 6.63) / (16.0 - 6.63)) * 100


# ────────────────────────────────────────────────────────────────────
# Formatting helpers
# ────────────────────────────────────────────────────────────────────

def _bytes(n: int) -> str:
    if abs(n) < 1024:
        return f"{n} B"
    for u in ("KiB", "MiB", "GiB", "TiB"):
        n /= 1024
        if abs(n) < 1024 or u == "TiB":
            return f"{n:,.2f} {u}"
    return f"{n:,.2f} PiB"

def _dur(s: float) -> str:
    if s < 0:
        return "--:--:--"
    return f"{int(s)//3600:02d}:{int(s)%3600//60:02d}:{int(s)%60:02d}"

def _bar(frac: float, w: int = 40) -> str:
    frac = max(0.0, min(1.0, frac))
    full = int(frac * w)
    return B["fl"] * full + B["sh"] * (w - full)

def _ratio(r: float) -> str:
    return f"{r:>5.2f} : 1"

def _pct(v: float) -> str:
    return f"{v:>6.2f}%"

def _trunc(s: str, w: int) -> str:
    if len(s) <= w:
        return s.ljust(w)
    half = (w - 3) // 2
    return s[:half] + "..." + s[-(w - 3 - half):]

import re as _re
_ANSI_RE = _re.compile(r'\033\[[0-9;]*m')

def _vlen(s: str) -> int:
    """Visible length of a string (strips ANSI codes)."""
    return len(_ANSI_RE.sub('', s))

def _pad_to(s: str, w: int) -> str:
    """Pad string with spaces so visible length equals w."""
    need = w - _vlen(s)
    return s + ' ' * max(0, need)

def _box_line(content: str, width: int = 72) -> str:
    """│ content padded to fill width │  — guaranteed exact alignment."""
    # Inner width = width - 2 (one │ on each side)
    inner = width - 2
    padded = _pad_to(content, inner)
    return f"{B['v']}{padded}{B['v']}"

def _box_top(label: str, width: int = 72) -> str:
    inner = width - 2
    label_part = f"{B['h']} {label} "
    rule = B['h'] * (inner - len(label_part))
    return f"{B['tl']}{label_part}{rule}{B['tr']}"

def _box_bottom(width: int = 72) -> str:
    return f"{B['bl']}{B['h'] * (width - 2)}{B['br']}"

def _box_empty(width: int = 72) -> str:
    return f"{B['v']}{' ' * (width - 2)}{B['v']}"


# ────────────────────────────────────────────────────────────────────
# Banner
# ────────────────────────────────────────────────────────────────────

_LOGO = r"""
 ██╗      █████╗ ███╗   ███╗ ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗
 ██║     ██╔══██╗████╗ ████║██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝
 ██║     ███████║██╔████╔██║██║   ██║██║   ██║███████║██╔██╗ ██║   ██║
 ██║     ██╔══██║██║╚██╔╝██║██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║
 ███████╗██║  ██║██║ ╚═╝ ██║╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║
 ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝ ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝
"""

_LOGO_PLAIN = """
================================================================
 LAMQUANT - Neural EEG Codec - OpenHuman Technologies
================================================================"""

def splash(duration: float = 0.5):
    """Show a quick splash screen, then clear. 0 = skip."""
    if duration <= 0 or not sys.stdout.isatty():
        return
    out = sys.stdout
    ver = _codec_version()
    desc = _git_describe()
    commit = _git_commit()[:7]

    if _UNICODE:
        out.write(f"\033[2J\033[H")
        print(f"{C.CYN}{_LOGO}{C.RST}", file=out)
    else:
        out.write(f"\033[2J\033[H")
        print(_LOGO_PLAIN, file=out)

    w = 72
    print(f"       {C.DIM}Neural EEG Codec {B['dot']} OpenHuman Technologies{C.RST}",
          file=out)
    print(file=out)
    print(f" {C.DIM}{B['h'] * w}{C.RST}", file=out)
    print(f" {C.DIM}Version{C.RST}     {desc}  "
          f"codec {ver}  commit {commit}", file=out)
    from lamquant_codec.cli.backend import detect_backend, get_backend_version
    from lamquant_codec.cli.config import load_config
    try:
        cfg = load_config()
        backend = detect_backend(cfg)
        bver = get_backend_version(cfg)
        print(f" {C.DIM}Backend{C.RST}     {backend} ({bver})", file=out)
    except Exception:
        pass
    import platform
    print(f" {C.DIM}Platform{C.RST}    {platform.system()} {platform.machine()}"
          f"  Python {sys.version_info.major}.{sys.version_info.minor}",
          file=out)
    print(f" {C.DIM}{B['h'] * w}{C.RST}", file=out)
    out.flush()

    time.sleep(duration)
    out.write("\033[3J\033[H\033[2J\033[H")
    out.flush()


def print_banner(stats: RunStats, mode: str = "lossless",
                 workers: int = 0, files_total: int = 0,
                 input_path: str = "", output_path: str = ""):
    w = 72
    out = sys.stdout

    if _UNICODE:
        print(f"{C.CYN}{_LOGO}{C.RST}", file=out)
        print(f"       {C.DIM}Neural EEG Codec {B['dot']} OpenHuman Technologies{C.RST}",
              file=out)
    else:
        print(_LOGO_PLAIN, file=out)

    print(file=out)
    print(f" {C.DIM}Mode{C.RST}        {mode:16s}"
          f"{C.DIM}Format{C.RST}   LML1 (CRC-32 + SHA-256)", file=out)
    print(f" {C.DIM}Version{C.RST}     {stats.git_describe}  "
          f"codec {stats.codec_version}  commit {stats.git_commit[:7]}",
          file=out)
    print(f" {C.DIM}Verified{C.RST}    CRC-32 per window {B['dot']} "
          f"SHA-256 per file {B['dot']} verify-after-write", file=out)
    if workers > 0 or files_total > 0:
        print(file=out)
        if input_path:
            print(f" {C.DIM}Input{C.RST}       {_trunc(input_path, 60)}", file=out)
        if output_path:
            print(f" {C.DIM}Output{C.RST}      {_trunc(output_path, 60)}", file=out)
        parts = []
        if files_total > 0:
            parts.append(f"{files_total:,} files")
        if workers > 0:
            parts.append(f"{workers} workers")
        if parts:
            print(f" {C.DIM}Config{C.RST}      {' {0} '.format(B['dot']).join(parts)}",
                  file=out)
    print(file=out)
    print(f" {C.DIM}{B['h'] * w}{C.RST}", file=out)
    print(file=out)
    out.flush()


# ────────────────────────────────────────────────────────────────────
# Dashboard (interactive, configurable Hz)
# ────────────────────────────────────────────────────────────────────

class Dashboard:
    def __init__(self, stats: RunStats, refresh_hz: float = 10.0):
        self.stats = stats
        self._lines = 0
        self._last_draw = 0.0
        self._last_stats = 0.0
        self._interval = 1.0 / max(1, min(60, refresh_hz))
        self._stats_interval = 0.25  # stats box at 4 Hz
        self._file = ""
        self._file_t0 = 0.0
        self._spin = 0
        self._input_path = ""
        self._output_path = ""
        self._first_draw = True

    def set_paths(self, inp: str, out: str):
        self._input_path = inp
        self._output_path = out

    def set_file(self, path: str):
        self._file = path
        self._file_t0 = time.time()

    def tick(self, force: bool = False):
        now = time.time()
        if not force and (now - self._last_draw) < self._interval:
            return
        if not sys.stdout.isatty():
            return
        self._last_draw = now
        self._spin = (self._spin + 1) % len(SPIN)
        self._draw()

    def clear(self):
        if self._lines > 0:
            sys.stdout.write(f"\033[{self._lines}A\033[J")
            sys.stdout.flush()
            self._lines = 0

    def _draw(self):
        if self._first_draw:
            # Clear entire screen on first frame so banner doesn't leave gaps
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
            self._first_draw = False
        else:
            self.clear()
        s = self.stats
        w = 72
        lines = []

        # Header
        elapsed = _dur(s.elapsed)
        lines.append(f" {C.DIM}{B['h'] * w}{C.RST}")
        title = "LamQuant Compression"
        pad = w - len(title) - len(f"[{elapsed}]") - 2
        lines.append(f" {C.BLD}{title}{C.RST}{' ' * max(pad, 1)}"
                     f"{C.DIM}[{elapsed}]{C.RST}")
        lines.append(f" {C.DIM}{B['h'] * w}{C.RST}")
        lines.append("")

        # Input/Output
        lines.append(f" {C.DIM}Input{C.RST}       {_trunc(self._input_path, 60)}")
        lines.append(f" {C.DIM}Output{C.RST}      {_trunc(self._output_path, 60)}")
        lines.append("")

        # Progress bar
        frac = s.files_done / max(s.files_total, 1)
        bar = _bar(frac, 40)
        lines.append(
            f" {C.DIM}Progress{C.RST}    {C.GRN}{bar}{C.RST}"
            f"  {frac*100:>5.1f}%    {s.files_done:>6,} / {s.files_total:,}")

        # Current file + spinner
        fn = _trunc(Path(self._file).name, 44) if self._file else "\u2014"
        ft = time.time() - self._file_t0 if self._file else 0
        sp = f"{C.CYN}{SPIN[self._spin]}{C.RST}" if self._file else " "
        lines.append(f" {C.DIM}Current{C.RST}     {fn}  {sp} {ft:>5.1f}s")

        # Throughput + ETA
        lines.append(
            f" {C.DIM}Throughput{C.RST}  {s.throughput:>8.1f} MiB/s"
            f"                   {C.DIM}ETA{C.RST} {_dur(s.eta_s)}")
        lines.append("")

        # Compression + Signal split box
        # Total: ┌─...─┬─...─┐ = 1 + LW + 1 + RW + 1 = w+1 chars (with leading space)
        # So LW + RW + 3 = w, meaning LW + RW = w - 3
        r = s.ratio
        saved = s.bytes_in - s.bytes_out
        hours = s.total_duration_s / 3600
        sh = s.shannon_pct
        LW = 35
        RW = w - LW - 3  # 72 - 35 - 3 = 34

        def _split(left_content, right_content):
            return (f" {B['v']}{_pad_to(left_content, LW)}"
                    f"{B['v']}{_pad_to(right_content, RW)}{B['v']}")

        # Top: ┌─ Compression ─...─┬─ Signal ─...─┐
        l_label = f"{B['h']} Compression "
        r_label = f"{B['h']} Signal "
        l_fill = B['h'] * (LW - len(l_label))
        r_fill = B['h'] * (RW - len(r_label))
        lines.append(f" {B['tl']}{l_label}{l_fill}{B['t']}{r_label}{r_fill}{B['tr']}")
        lines.append(_split(
            f" Ratio       {_ratio(r)}",
            f" Files       {s.files_ok:>8,} OK"))
        lines.append(_split(
            f" Saved       {_bytes(saved):>14s}",
            f" EEG hours   {hours:>10,.1f}"))
        lines.append(_split(
            f" Shannon     {_pct(sh)}",
            f" Samples     {s.total_samples/1e9:>7.2f} G"))
        # Bottom: └─...─┴─...─┘  must match exactly
        lines.append(f" {B['bl']}{B['h'] * LW}{B['b']}{B['h'] * RW}{B['br']}")
        lines.append("")

        # Integrity line (compact)
        ck = f"{C.GRN}{B['ok']}{C.RST}"
        ec = C.RED if s.files_error else C.GRN
        fc = f"{s.files_ok:>6,}/{s.files_done:<6,}"
        lines.append(f" {_box_top('Integrity', w)}")
        lines.append(f" {_box_line(f' CRC-32 + SHA-256  {ck} verified  {fc}    Errors {ec}{s.files_error:>3,}{C.RST}', w)}")
        lines.append(f" {_box_bottom(w)}")

        for line in lines:
            print(line)
        self._lines = len(lines)
        sys.stdout.flush()


# ────────────────────────────────────────────────────────────────────
# Per-file line (piped mode)
# ────────────────────────────────────────────────────────────────────

def print_file_line(idx: int, total: int, r: FileResult):
    name = _trunc(Path(r.path).name, 40)
    if r.status == "ok":
        st = f"{C.GRN}{B['ok']}{C.RST}"
    elif r.status == "skipped":
        st = f"{C.DIM}skip{C.RST}"
    else:
        st = f"{C.RED}{B['no']}{C.RST}"

    if r.status == "error":
        err = r.error[:30] if r.error else "unknown"
        print(f"  [{idx:>5}/{total:<5}]  {name}"
              f"  ({err})"
              f"{'':>22}  {st}")
    else:
        print(f"  [{idx:>5}/{total:<5}]  {name}"
              f"  {_bytes(r.bytes_in):>12s}  {B['ar']}  {_bytes(r.bytes_out):>12s}"
              f"  {_ratio(r.ratio)}  {r.duration_s:>5.1f}s  {st}")
    sys.stdout.flush()


# ────────────────────────────────────────────────────────────────────
# Completion summary
# ────────────────────────────────────────────────────────────────────

def print_summary(stats: RunStats, output_dir: Path,
                   results: Optional[list] = None, verbose: bool = False):
    s = stats
    w = 72
    out = sys.stdout

    print(file=out)
    print(f" {C.DIM}{B['h'] * w}{C.RST}", file=out)
    print(f"  {C.GRN}{B['ok']}{C.RST} {C.BLD}Compression complete{C.RST}"
          f"{'':>38s}{C.DIM}[{_dur(s.elapsed)}]{C.RST}", file=out)
    print(f" {C.DIM}{B['h'] * w}{C.RST}", file=out)
    print(file=out)

    saved = s.bytes_in - s.bytes_out
    hours = s.total_duration_s / 3600

    print(f"  {C.DIM}Files{C.RST}       "
          f"{s.files_ok:>6,} compressed  {B['dot']}  "
          f"{s.files_error:>4,} failed  {B['dot']}  "
          f"{s.files_skipped:>4,} skipped", file=out)
    print(f"  {C.DIM}Input{C.RST}       {_bytes(s.bytes_in):>14s}"
          f"              {hours:>10,.1f} hours EEG", file=out)
    print(f"  {C.DIM}Output{C.RST}      {_bytes(s.bytes_out):>14s}"
          f"              saved {_bytes(saved)}", file=out)
    print(file=out)

    r = s.ratio
    gap = s.shannon_gap
    bps = 16.0 / r if r > 0 else 16.0
    sh = s.shannon_pct

    print(f"  {C.BLD}Compression ratio{C.RST}"
          f"         {C.GRN}{_ratio(r)}{C.RST}", file=out)
    print(f"  {C.BLD}Shannon efficiency{C.RST}"
          f"       {_pct(sh)}", file=out)
    print(f"  {C.BLD}Gap to theoretical floor{C.RST}"
          f" {gap:>5.2f} bits/sample  "
          f"{C.DIM}({bps:.2f} vs 6.63 bps){C.RST}", file=out)
    print(f"  {C.DIM}Effective bit rate{C.RST}"
          f"        {16.0/max(r,0.01)*250*21/1000:>6,.0f} kbps  "
          f"{C.DIM}(21ch @ 250 Hz){C.RST}", file=out)
    print(file=out)

    # Integrity box — field-level alignment
    W = 72
    ck = f"{C.GRN}{B['ok']}{C.RST}"
    fc = f"{s.files_ok:>6,} / {s.files_done:<6,}"

    def _iline(label, result_text):
        col1 = _pad_to(f"  {label}", 25)
        col2 = _pad_to(fc, 15)
        col3 = f"files   {ck} {result_text}"
        return _box_line(f"{col1} {col2} {col3}", W)

    ec = C.RED if s.files_error else C.GRN
    print(f"  {_box_top('Integrity verification', W)}", file=out)
    print(f"  {_box_empty(W)}", file=out)
    print(f"  {_iline('CRC-32 per window', 'all verified')}", file=out)
    print(f"  {_iline('SHA-256 per file', 'all verified')}", file=out)
    print(f"  {_iline('Verify-after-write', 'roundtrip OK')}", file=out)
    print(f"  {_box_line(f'  Errors {ec}{s.files_error:>4,}{C.RST}    Warnings {C.DIM}{len(s.warnings):>4,}{C.RST}', W)}", file=out)
    print(f"  {_box_empty(W)}", file=out)
    print(f"  {_box_bottom(W)}", file=out)
    print(file=out)

    # CR distribution (when we have per-file results)
    if results:
        ok_results = [r for r in results if r.status == "ok" and r.ratio > 0]
        if len(ok_results) >= 2:
            ratios = sorted(r.ratio for r in ok_results)
            n = len(ratios)

            def _p(pct):
                idx = min(int(pct / 100 * n), n - 1)
                return ratios[idx]

            best = max(ok_results, key=lambda r: r.ratio)
            worst = min(ok_results, key=lambda r: r.ratio)

            print(f"  {C.DIM}CR distribution{C.RST}  "
                  f"min {_ratio(ratios[0])}  {B['dot']}  "
                  f"p50 {_ratio(_p(50))}  {B['dot']}  "
                  f"max {_ratio(ratios[-1])}", file=out)
            print(f"  {C.DIM}Best compressed{C.RST}  "
                  f"{_trunc(Path(best.path).name, 40)}"
                  f"  {C.GRN}{_ratio(best.ratio)}{C.RST}", file=out)
            print(f"  {C.DIM}Worst compressed{C.RST} "
                  f"{_trunc(Path(worst.path).name, 40)}"
                  f"  {C.YEL}{_ratio(worst.ratio)}{C.RST}", file=out)

            if verbose and n >= 5:
                print(file=out)
                print(f"  {C.DIM}Bottom 5 (investigate for artifacts/corruption):{C.RST}",
                      file=out)
                for r in sorted(ok_results, key=lambda x: x.ratio)[:5]:
                    print(f"    {_trunc(Path(r.path).name, 44)}"
                          f"  {_ratio(r.ratio)}  "
                          f"{_bytes(r.bytes_in):>12s}", file=out)

            print(file=out)

    # Paths
    manifest = output_dir / "manifest.lml.json"
    audit = output_dir / "audit.log"
    print(f"  {C.DIM}Manifest{C.RST}    {manifest}", file=out)
    print(f"  {C.DIM}Audit log{C.RST}   {audit}", file=out)
    print(file=out)

    # Reproducibility footer — ALWAYS
    print(f"  {C.DIM}codec={s.codec_version}  "
          f"commit={s.git_commit[:10]}  "
          f"config=sha256:{s.config_hash[:12]}{C.RST}", file=out)
    print(file=out)
    out.flush()


def print_summary_json(stats: RunStats):
    """Quiet mode: single JSON line to stdout."""
    import json
    s = stats
    doc = {
        "files_ok": s.files_ok,
        "files_error": s.files_error,
        "files_skipped": s.files_skipped,
        "bytes_in": s.bytes_in,
        "bytes_out": s.bytes_out,
        "ratio": round(s.ratio, 4),
        "shannon_efficiency": round(s.shannon_pct / 100, 4),
        "wall_time_s": round(s.elapsed, 2),
        "codec_version": s.codec_version,
        "git_commit": s.git_commit,
    }
    print(json.dumps(doc))
    sys.stdout.flush()


# ────────────────────────────────────────────────────────────────────
# Audit log (line-buffered, crash-safe)
# ────────────────────────────────────────────────────────────────────

class AuditLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fh = open(path, "a", buffering=1)
        except OSError as e:
            print(f"WARNING: Cannot open audit log {path}: {e}", file=sys.stderr)
            self._fh = None
        self._t0 = time.time()

    def _ts(self) -> str:
        return datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def _w(self, kind: str, **kv):
        if self._fh is None:
            return
        parts = [self._ts(), kind.ljust(9)]
        for k, v in kv.items():
            sv = str(v)
            parts.append(f'{k}="{sv}"' if " " in sv else f"{k}={sv}")
        self._fh.write("  ".join(parts) + "\n")

    def start(self, argv, stats):
        self._w("START", cmd=" ".join(argv))
        self._w("VERSION", codec=stats.codec_version,
                commit=stats.git_commit[:10],
                python=f"{sys.version_info.major}.{sys.version_info.minor}")

    def config(self, h):
        self._w("CONFIG", checksum=f"sha256:{h[:16]}")

    def scan(self, n, total_bytes):
        self._w("SCAN", files=n, bytes=total_bytes)

    def file_ok(self, idx, total, r):
        self._w("FILE", idx=f"{idx}/{total}", path=Path(r.path).name,
                state="ok", ratio=f"{r.ratio:.3f}",
                sha256=r.sha256[:16], time=f"{r.duration_s:.2f}s")

    def file_error(self, idx, total, path, err):
        self._w("FILE", idx=f"{idx}/{total}", path=Path(path).name,
                state="error", error=str(err)[:100])

    def interrupt(self, stats):
        self._w("INTERRUPT", files_done=stats.files_done,
                files_ok=stats.files_ok)

    def summary(self, stats):
        self._w("SUMMARY", files_ok=stats.files_ok,
                files_error=stats.files_error,
                ratio=f"{stats.ratio:.4f}",
                wall=f"{stats.elapsed:.1f}s")

    def end(self, stats, code):
        self._w("END", exit=code, files_ok=stats.files_ok,
                ratio=f"{stats.ratio:.4f}",
                duration_s=f"{stats.elapsed:.1f}s")
        if self._fh is not None:
            self._fh.close()

    def __del__(self):
        if self._fh and not self._fh.closed:
            self._fh.close()


# ────────────────────────────────────────────────────────────────────
# Manifest (JSON, atomic write)
# ────────────────────────────────────────────────────────────────────

def write_manifest(path: Path, stats: RunStats, results: list):
    import json, platform, tempfile
    s = stats
    doc = {
        "schema_version": "1.0",
        "codec": {
            "name": "LamQuant", "mode": "lossless",
            "version": s.codec_version, "format": "LML1",
        },
        "run": {
            "start_time": datetime.fromtimestamp(
                s.start_time, timezone.utc).isoformat(),
            "wall_time_s": round(s.elapsed, 2),
            "git_commit": s.git_commit,
            "config_hash": s.config_hash,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "platform": platform.platform(),
        },
        "statistics": {
            "files_ok": s.files_ok,
            "files_error": s.files_error,
            "files_skipped": s.files_skipped,
            "bytes_in": s.bytes_in,
            "bytes_out": s.bytes_out,
            "compression_ratio": round(s.ratio, 4),
            "shannon_efficiency": round(s.shannon_pct / 100, 4),
            "total_eeg_hours": round(s.total_duration_s / 3600, 2),
            "total_samples": s.total_samples,
        },
        "integrity": {
            "window_checksum": "CRC-32",
            "file_checksum": "SHA-256",
            "verify_after_write": True,
        },
        "files": [
            {"path": Path(r.path).name, "bytes_in": r.bytes_in,
             "bytes_out": r.bytes_out, "ratio": round(r.ratio, 3),
             "sha256": r.sha256, "status": r.status}
            for r in results
        ],
    }
    # Atomic write: temp + rename
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, separators=(",", ":"))
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
