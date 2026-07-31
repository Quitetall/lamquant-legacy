"""Coverage tests for ``lamquant_codec.cli.box``.

Per ``feedback_futureproof_tests``: pin shape/type invariants, not
implementation-derived values. The box layout invariant is:

    * Every rendered line ends at the same visible column.
    * pad_to() never shrinks input.
    * Width parameter is honored.

We deliberately do not pin exact box-drawing characters because the
module auto-falls-back from unicode to ASCII based on the terminal,
and characters can differ across CI environments.
"""
from __future__ import annotations

import re

import pytest

from lamquant_codec.cli.box import Box, SplitBox, header, pad_to, tbox

_ANSI = re.compile(r"\033\[[0-9;]*m")


def _vlen(s: str) -> int:
    """Visible length (strip ANSI)."""
    return len(_ANSI.sub("", s))


# ----- pad_to -----------------------------------------------------------


def test_pad_to_zero_width_returns_input():
    assert pad_to("", 0) == ""


def test_pad_to_exact_width():
    out = pad_to("abc", 3)
    assert out == "abc"


def test_pad_to_widens_with_spaces():
    out = pad_to("ab", 5)
    assert isinstance(out, str)
    assert out.startswith("ab")
    assert _vlen(out) == 5


def test_pad_to_does_not_shrink():
    # If input is wider than width, pad_to should return input unchanged
    # (max(0, ...) clamp).
    out = pad_to("abcdef", 2)
    assert out == "abcdef"


def test_pad_to_strips_ansi_for_length_calc():
    # ANSI codes should NOT count toward visible width
    raw = "\033[31mX\033[0m"  # one visible char wrapped in red ANSI
    out = pad_to(raw, 3)
    # 2 spaces appended → visible length 3
    assert _vlen(out) == 3


# ----- Box --------------------------------------------------------------


def test_box_construct_defaults():
    b = Box()
    assert b.width == 72
    assert b.title == ""
    assert b._lines == []


def test_box_with_custom_width():
    b = Box(width=40, title="hello")
    assert b.width == 40
    assert b.title == "hello"


def test_box_line_returns_self_for_chaining():
    b = Box()
    assert b.line("a") is b
    assert b.line() is b


def test_box_blank_appends_empty():
    b = Box()
    b.blank()
    assert b._lines == [""]


def test_box_render_returns_str():
    out = Box(width=20, title="t").line("x").render()
    assert isinstance(out, str)
    assert out  # non-empty


def test_box_render_all_lines_same_visible_width():
    """The load-bearing invariant — every line ends at the same column."""
    b = Box(width=30, title="Demo")
    b.line("short")
    b.line("a bit longer line")
    b.blank()
    b.line("[1] option")
    out = b.render()
    lines = out.split("\n")
    visible_widths = {_vlen(ln) for ln in lines}
    # All rendered lines must have the same visible length
    assert len(visible_widths) == 1


def test_box_render_with_no_title():
    out = Box(width=20).line("x").render()
    assert isinstance(out, str)
    assert "x" in out


def test_box_print_does_not_raise(capsys):
    Box(width=20, title="x").line("y").print()
    captured = capsys.readouterr()
    assert "y" in captured.out


def test_tbox_returns_box_instance():
    b = tbox(width=30, title="hi")
    assert isinstance(b, Box)
    assert b.width == 30
    assert b.title == "hi"


# ----- SplitBox ---------------------------------------------------------


def test_splitbox_construct_defaults():
    sb = SplitBox()
    assert sb.width == 72
    assert sb._rows == []


def test_splitbox_row_returns_self():
    sb = SplitBox(width=60, split=30)
    assert sb.row("left", "right") is sb


def test_splitbox_render_returns_str():
    sb = SplitBox(width=60, left_title="L", right_title="R", split=28)
    sb.row("alpha", "beta")
    sb.row("gamma", "delta")
    out = sb.render()
    assert isinstance(out, str)


def test_splitbox_render_all_lines_same_width():
    sb = SplitBox(width=60, left_title="L", right_title="R", split=28)
    sb.row("alpha", "beta")
    sb.row("gamma is longer", "delta")
    out = sb.render()
    lines = out.split("\n")
    widths = {_vlen(ln) for ln in lines}
    assert len(widths) == 1


def test_splitbox_print_does_not_raise(capsys):
    sb = SplitBox(width=40, split=18)
    sb.row("a", "b")
    sb.print()
    captured = capsys.readouterr()
    assert "a" in captured.out and "b" in captured.out


# ----- header -----------------------------------------------------------


def test_header_returns_string():
    out = header("Title")
    assert isinstance(out, str)
    assert "Title" in out


def test_header_with_right():
    out = header("Left", "Right", width=40)
    assert "Left" in out and "Right" in out


def test_header_width_param_used():
    # The leading & trailing rules should be approximately ``width`` chars
    # long. Pin: line count is 3 (two rule lines + the title line).
    out = header("X", width=20)
    lines = out.strip().split("\n")
    assert len(lines) == 3
