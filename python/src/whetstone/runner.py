"""Run the harness as a subprocess and consume its event stream (used by the CLI)."""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from whetstone import protocol


def harness_command(
    problem_dir: Path,
    solution: Path,
    framework: str,
    mode: str,
    case: str | None = None,
    *,
    reveal: bool = False,
) -> list[str]:
    """The harness command for the interpreter that is already running (no uv needed)."""
    command = [
        sys.executable, "-P", "-m", "whetstone.harness",
        "--problem", str(problem_dir.resolve()),
        "--solution", str(solution.resolve()),
        "--framework", framework,
        "--mode", mode,
    ]  # fmt: skip
    if case is not None:
        command += ["--case", case]
    if reveal:
        command.append("--reveal")
    return command


@dataclass
class RunResult:
    start: dict[str, Any] | None = None
    cases: list[dict[str, Any]] = field(default_factory=list)
    end: dict[str, Any] | None = None
    logs: list[str] = field(default_factory=list)
    returncode: int = 0

    @property
    def verdict(self) -> str:
        return str(self.end["status"]) if self.end else "harness_error"


def stream_events(command: list[str]) -> Iterator[dict[str, Any]]:
    """Yield protocol events; non-protocol stdout lines come as ``{"event": "log"}``.

    Always ends with an ``end`` event: one is synthesized if the harness died silently.
    """
    started = time.monotonic()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL)
    saw_end = False
    assert proc.stdout is not None
    try:
        for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            event = protocol.decode(line)
            if event is None:
                yield {"event": "log", "text": line}
                continue
            saw_end = saw_end or event.get("event") == "end"
            yield event
    finally:
        if proc.poll() is None and not saw_end:
            proc.kill()
        proc.wait()
    if not saw_end:
        yield {
            "event": "end",
            "status": "harness_error",
            "passed": 0,
            "total": 0,
            "skipped": 0,
            "duration_ms": round((time.monotonic() - started) * 1000, 3),
            "metrics": [],
            "message": f"the harness exited with code {proc.returncode} without an end event",
        }


def run(command: list[str], on_event: Callable[[dict[str, Any]], None] | None = None) -> RunResult:
    """Run the harness to completion, optionally observing each event as it arrives."""
    result = RunResult()
    for event in stream_events(command):
        kind = event.get("event")
        if kind == "start":
            result.start = event
        elif kind == "case_end":
            result.cases.append(event)
        elif kind == "end":
            result.end = event
        elif kind == "log":
            result.logs.append(str(event.get("text", "")))
        if on_event is not None:
            on_event(event)
    return result
