"""
Shared terminal capability detection for the LamQuant CLI.

Single source of truth for color, unicode, and width detection.
All CLI modules import from here instead of doing their own detection.

    from lamquant_codec.cli.terminal import C, S, term_width, error, warning
"""
import os
import re
import shutil
import sys
from typing import Optional

# ── ANSI regex (used by box.py, readout.py, cockpit.py) ──
ANSI_RE = re.compile(r'\033\[[0-9;]*m')


def vlen(s: str) -> int:
    """Visible length of a string (strips ANSI escape codes)."""
    return len(ANSI_RE.sub('', s))


# ── Detection ──

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


def term_width() -> int:
    """Terminal width, clamped to a usable range."""
    try:
        return shutil.get_terminal_size((80, 24)).columns
    except Exception:
        return 80


def term_height() -> int:
    """Terminal height."""
    try:
        return shutil.get_terminal_size((80, 24)).lines
    except Exception:
        return 24


# ── Cached state ──

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
        for name, code in [("RST", "0"), ("DIM", "90"), ("CYN", "36"),
                           ("GRN", "32"), ("RED", "31"), ("YEL", "33"),
                           ("BLD", "1")]:
            setattr(cls, name, f"\033[{code}m" if color else "")


# ── Symbols ──

def _build_symbols(unicode: bool) -> dict:
    if unicode:
        return dict(
            h="\u2500", v="\u2502",
            tl="\u250c", tr="\u2510", bl="\u2514", br="\u2518",
            t="\u252c", b="\u2534", l="\u251c", r="\u2524",
            ok="\u2713", no="\u2717", ar="\u2192", dot="\u00b7",
            fl="\u2588", sh="\u2591",
        )
    return dict(
        h="-", v="|",
        tl="+", tr="+", bl="+", br="+",
        t="+", b="+", l="+", r="+",
        ok="OK", no="X", ar="->", dot="-",
        fl="#", sh=".",
    )


S = _build_symbols(_UNICODE)
"""Symbol dict: h, v, tl, tr, bl, br, ok, no, ar, dot, fl, sh."""

SPIN = (["\u280b", "\u2819", "\u2839", "\u2838",
         "\u283c", "\u2834", "\u2826", "\u2827",
         "\u2807", "\u280f"] if _UNICODE
        else ["|", "/", "-", "\\"])


# ── Structured messages (always to stderr) ──

def error(msg: str):
    """Print error message to stderr."""
    print(f"{C.RED}error:{C.RST} {msg}", file=sys.stderr)


def warning(msg: str):
    """Print warning message to stderr."""
    print(f"{C.YEL}warning:{C.RST} {msg}", file=sys.stderr)


def fatal(msg: str):
    """Print fatal error message to stderr."""
    print(f"{C.RED}fatal:{C.RST} {msg}", file=sys.stderr)
