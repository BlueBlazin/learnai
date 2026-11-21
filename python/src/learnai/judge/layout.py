"""How a failure detail is laid out: groups one blank line apart, facts in two columns.

A detail is a sequence of groups. A group is a lowercase title, flush left, and its
contents indented two spaces: either facts (label, then value, the values in one
column) or a block of text (an array as numpy prints it, a repr). Text that a test
wrote itself is a group of its own and is never re-laid out.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

#: Spaces between the longest label of a group and the column of values.
GAP = 4


def join(*groups: str | None) -> str:
    """The groups that are present, one blank line apart."""
    return "\n\n".join(group.rstrip("\n") for group in groups if group) + "\n"


def column(labels: Sequence[str]) -> int:
    """The width labels are padded to so that what follows them lines up."""
    return max(len(label) for label in labels) + GAP


def facts(title: str, rows: Sequence[tuple[str, Any]]) -> str:
    """A group of ``label    value`` rows with the values in one column."""
    width = column([label for label, _ in rows])
    lines = [title]
    for label, value in rows:
        first, *rest = str(value).split("\n")
        lines.append(f"  {label:<{width}}{first}")
        lines.extend(" " * (2 + width) + line for line in rest)
    return "\n".join(lines)


def block(title: str, text: str) -> str:
    """A group that is ``text`` as it stands, indented two spaces."""
    return "\n".join([title, *("  " + line for line in text.split("\n"))])


def located(value: str, expected: str, got: str) -> str:
    """A value with where it is, then what was expected and what came back there."""
    return f"{value}   (expected {expected}, got {got})"
