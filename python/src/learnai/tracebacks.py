"""Tracebacks trimmed to the frames a learner wrote."""

from __future__ import annotations

import os
import re
import traceback
from collections.abc import Iterable


def _same_file(a: str, b: str) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.abspath(a) == os.path.abspath(b)


_MEANINGFUL = re.compile(r"\w")


def exception_line(exc: BaseException) -> str:
    """``TypeError: message`` on one line, whatever the exception's text looks like.

    A multi-line message (framework errors often are: a summary, then shapes, hints,
    or a stack of their own) is reduced to the exception type and the first line that
    says something. A first line that only introduces what follows (it ends with a
    colon) is joined with the next one. Nothing is added that the exception did not
    say itself, so the hidden-case rule (ARCHITECTURE 3.6) is unaffected.
    """
    formatted = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    if isinstance(exc, SyntaxError):
        # The file, source line, and caret come first; the last line is the message.
        lines = formatted.splitlines()
        return lines[-1] if lines else type(exc).__name__
    try:
        text = str(exc)
    except Exception:  # noqa: BLE001 - a broken __str__ must not break the report
        text = ""
    lines = [" ".join(line.split()) for line in text.splitlines()]
    meaningful = [line for line in lines if _MEANINGFUL.search(line)]
    # The type as Python prints it (``module.Name`` for non-builtins).
    kind = formatted.split(":", 1)[0].splitlines()[0].strip() if formatted else ""
    if not kind or " " in kind:
        kind = type(exc).__name__
    if not meaningful:
        return kind
    first = meaningful[0]
    if first.endswith(":") and len(meaningful) > 1:
        first = f"{first} {meaningful[1]}"
    return f"{kind}: {first}"


def user_location(exc: BaseException, user_path: str) -> str | None:
    """``solution.py line 12, in softmax``: the innermost frame in the learner's file."""
    for frame in reversed(traceback.extract_tb(exc.__traceback__)):
        if _same_file(frame.filename, user_path):
            return f"{os.path.basename(frame.filename)} line {frame.lineno}, in {frame.name}"
    if isinstance(exc, SyntaxError) and exc.filename and _same_file(exc.filename, user_path):
        return f"{os.path.basename(exc.filename)} line {exc.lineno}"
    return None


def hidden_error_message(exc: BaseException, user_path: str, limit: int = 200) -> str:
    """All a hidden case says about an exception, on one line.

    The learner's own exception type and message (cut to ``limit`` characters) and the
    line in their own file that was running. Nothing from the test: no frames of
    tests.py, no source lines, no argument values beyond what the exception itself
    says.
    """
    text = " ".join(exception_line(exc).split())
    if len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    where = user_location(exc, user_path)
    return f"{text} ({where})" if where else text


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
