"""Tracebacks trimmed to the frames a learner wrote."""

from __future__ import annotations

import os
import traceback
from collections.abc import Iterable


def _same_file(a: str, b: str) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.abspath(a) == os.path.abspath(b)


def exception_line(exc: BaseException) -> str:
    """``TypeError: message`` on one line."""
    text = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    lines = text.splitlines()
    return lines[-1] if lines else type(exc).__name__


def raised_in(exc: BaseException, path: str) -> bool:
    """True when the innermost frame of the traceback is in ``path``."""
    frames = traceback.extract_tb(exc.__traceback__)
    return bool(frames) and _same_file(frames[-1].filename, path)


def format_user_traceback(exc: BaseException, user_path: str, also_show: Iterable[str] = ()) -> str:
    """Format ``exc`` keeping only frames from the solution file.

    Harness and library frames are noise to a learner, so they are dropped. When the
    exception was raised below the learner's code (inside a framework or the kit), one
    line says where. ``also_show`` lists extra files (the public test file) whose frames
    are kept when the learner's file does not appear at all.
    """
    frames = traceback.extract_tb(exc.__traceback__)
    kept = [frame for frame in frames if _same_file(frame.filename, user_path)]
    if not kept:
        extra = tuple(also_show)
        kept = [f for f in frames if any(_same_file(f.filename, path) for path in extra)]
    lines: list[str] = []
    if kept:
        for frame in kept:
            frame.filename = os.path.basename(frame.filename)
        lines.append("Traceback (most recent call last):\n")
        lines.extend(traceback.format_list(kept))
        innermost = frames[-1]
        if innermost is not kept[-1] and not isinstance(exc, SyntaxError):
            where = os.sep.join(innermost.filename.split(os.sep)[-2:])
            lines.append(
                f"  (raised inside {where}, line {innermost.lineno}, in {innermost.name})\n"
            )
    lines.extend(traceback.format_exception_only(type(exc), exc))
    return "".join(lines).rstrip() + "\n"
