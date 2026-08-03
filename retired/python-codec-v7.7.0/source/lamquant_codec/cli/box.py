"""
Typed ASCII box rendering — guaranteed alignment.

Every box in LamQuant goes through this module. No hand-counted spaces.
The Box class computes padding from visible character width (ANSI-aware)
and guarantees that every line has exactly the same visible width.

Usage:
    box = Box(width=72, title="Compress data")
    box.line("What are you compressing?")
    box.line()
    box.line("  [1]  Single file")
    box.line("  [2]  Directory of files")
    box.line()
    box.line("[b] Back    [?] Help")
    print(box.render())

Split box (two columns):
    box = SplitBox(width=72, left_title="Compression", right_title="Signal", split=35)
    box.row("Ratio       2.26 : 1", "Files       1,234 OK")
    print(box.render())
"""
from lamquant_codec.cli.terminal import vlen, C as _C, S as _S


def pad_to(s: str, width: int) -> str:
    """Pad string with spaces so visible length equals width."""
    need = width - vlen(s)
    return s + ' ' * max(0, need)


def tbox(width: int = 72, title: str = "") -> 'Box':
    """Create a Box using the shared terminal symbol set (auto ASCII fallback)."""
    return Box(width=width, title=title,
               h=_S["h"], v=_S["v"], tl=_S["tl"], tr=_S["tr"],
               bl=_S["bl"], br=_S["br"])


class Box:
    """Single-column box with guaranteed alignment.

    All lines render at exactly `width` visible characters between
    the left and right borders. The borders themselves add 4 chars
    (2 indent + left border + space ... right border).
    """

    def __init__(self, width: int = 72, title: str = "",
                 h="─", v="│", tl="┌", tr="┐", bl="└", br="┘"):
        self.width = width
        self.title = title
        self.h = h
        self.v = v
        self.tl = tl
        self.tr = tr
        self.bl = bl
        self.br = br
        self._lines: list = []

    def line(self, content: str = ""):
        """Add a content line. Padding computed automatically."""
        self._lines.append(content)
        return self

    def blank(self):
        """Add an empty line."""
        return self.line("")

    def render(self) -> str:
        """Render the complete box as a string."""
        w = self.width
        parts = []

        # Top border
        if self.title:
            label = f"{self.h} {self.title} "
            fill = self.h * max(0, w - vlen(label))
            parts.append(f"  {self.tl}{label}{fill}{self.tr}")
        else:
            parts.append(f"  {self.tl}{self.h * w}{self.tr}")

        # Content lines
        for content in self._lines:
            padded = pad_to(content, w - 1)  # -1 for the leading space after │
            parts.append(f"  {self.v} {padded}{self.v}")

        # Bottom border
        parts.append(f"  {self.bl}{self.h * w}{self.br}")

        return "\n".join(parts)

    def print(self):
        """Render and print."""
        print(self.render())


class SplitBox:
    """Two-column box with guaranteed alignment.

    left_width + right_width + 3 (for │ borders) = total width.
    """

    def __init__(self, width: int = 72, left_title: str = "",
                 right_title: str = "", split: int = 35,
                 h="─", v="│", tl="┌", tr="┐", bl="└", br="┘",
                 t="┬", b="┴"):
        self.width = width
        self.left_title = left_title
        self.right_title = right_title
        self.lw = split           # left inner width
        self.rw = width - split - 3  # right inner width (3 = │ │ │)
        self.h = h
        self.v = v
        self.tl = tl
        self.tr = tr
        self.bl = bl
        self.br = br
        self.t = t
        self.b = b
        self._rows: list = []

    def row(self, left: str, right: str):
        """Add a row with left and right content."""
        self._rows.append((left, right))
        return self

    def render(self) -> str:
        w = self.width
        parts = []

        # Top border with titles
        lt = f"{self.h} {self.left_title} " if self.left_title else ""
        rt = f"{self.h} {self.right_title} " if self.right_title else ""
        lf = self.h * (self.lw - len(lt))
        rf = self.h * (self.rw - len(rt))
        parts.append(f"  {self.tl}{lt}{lf}{self.t}{rt}{rf}{self.tr}")

        # Content rows
        for left, right in self._rows:
            lp = pad_to(left, self.lw)
            rp = pad_to(right, self.rw)
            parts.append(f"  {self.v}{lp}{self.v}{rp}{self.v}")

        # Bottom border
        parts.append(f"  {self.bl}{self.h * self.lw}{self.b}"
                     f"{self.h * self.rw}{self.br}")

        return "\n".join(parts)

    def print(self):
        print(self.render())


# ────────────────────────────────────────────────────────────────────
# Convenience: header rule (no box, just title + rule)
# ────────────────────────────────────────────────────────────────────

def header(title: str, right: str = "", width: int = 72, h=None,
           dim=None, bold=None, rst=None) -> str:
    """Render a header line with title left, optional right text."""
    h = h or _S["h"]
    dim = dim if dim is not None else _C.DIM
    bold = bold if bold is not None else _C.BLD
    rst = rst if rst is not None else _C.RST
    pad = width - vlen(title) - vlen(right)
    return (f"\n  {dim}{h * width}{rst}\n"
            f"  {bold}{title}{rst}{' ' * max(pad, 1)}{dim}{right}{rst}\n"
            f"  {dim}{h * width}{rst}")
