"""Human-readable terminal output, built from the same event stream the app consumes."""

from __future__ import annotations

import os
import sys
from typing import Any, TextIO

VERDICT_TEXT = {
    "accepted": "Accepted",
    "wrong_answer": "Wrong answer",
    "runtime_error": "Runtime error",
    "time_limit": "Time limit exceeded",
    "import_error": "Import error",
    "harness_error": "Harness error",
    "assets_missing": "Not judged: model asset missing",
    "hardware_missing": "Not accepted: hardware missing",
}
STATUS_TEXT = {
    "pass": "PASS",
    "fail": "FAIL",
    "error": "ERROR",
    "timeout": "TIME",
    "skipped": "SKIP",
}
_COLORS = {
    "pass": "32", "fail": "31", "error": "31", "timeout": "33", "skipped": "2",
    "accepted": "32", "incomplete": "33", "dim": "2", "bold": "1",
}  # fmt: skip


class Renderer:
    """Feed it events with ``handle``; it prints as they arrive."""

    def __init__(self, out: TextIO | None = None, color: bool | None = None) -> None:
        self.out = out or sys.stdout
        if color is None:
            color = self.out.isatty() and "NO_COLOR" not in os.environ
        self.color = color
        self.names: dict[str, str] = {}
        self.width = 8

    def paint(self, text: str, style: str) -> str:
        code = _COLORS.get(style)
        return f"\033[{code}m{text}\033[0m" if self.color and code else text

    def line(self, text: str = "") -> None:
        self.out.write(text + "\n")
        self.out.flush()

    def block(self, title: str, text: str) -> None:
        self.line(f"      {self.paint(title, 'dim')}")
        for row in text.rstrip("\n").splitlines():
            self.line(f"      {self.paint('|', 'dim')} {row}")

    def handle(self, event: dict[str, Any]) -> None:
        kind = event.get("event")
        if kind == "start":
            self.names = {c["id"]: c["name"] for c in event.get("cases", [])}
            self.width = max([8, *(len(case_id) for case_id in self.names)])
            env = event.get("env", {})
            versions = ", ".join(f"{k} {v}" for k, v in env.items() if k not in ("platform",))
            head = f"{event.get('problem')} / {event.get('framework')} / {event.get('mode')}"
            self.line(f"{self.paint(head, 'bold')}  {self.paint(versions, 'dim')}")
        elif kind == "case_end":
            self.case(event)
        elif kind == "end":
            self.end(event)
        elif kind == "log":
            self.line(self.paint(f"  log: {event.get('text', '')}", "dim"))

    def case(self, event: dict[str, Any]) -> None:
        status = str(event.get("status"))
        case_id = str(event.get("id"))
        tag = self.paint(f"{STATUS_TEXT.get(status, status.upper()):5}", status)
        timing = "" if status == "skipped" else f"{event.get('duration_ms', 0):.1f} ms"
        name = self.names.get(case_id, "")
        self.line(f"  {tag} {case_id:<{self.width}}  {name}  {self.paint(timing, 'dim')}")
        for metric in event.get("metrics") or []:
            self.line(f"      {self.paint(format_metric(metric), 'dim')}")
        if status == "pass":
            if event.get("stdout"):
                self.block("output", event["stdout"])
            return
        if event.get("message"):
            self.line(f"      {event['message']}")
        if status == "skipped":
            return
        if event.get("detail"):
            self.block("detail", event["detail"])
        if event.get("stdout"):
            self.block("output", event["stdout"])

    def end(self, event: dict[str, Any]) -> None:
        status = str(event.get("status"))
        verdict = VERDICT_TEXT.get(status, status)
        style = "accepted" if status == "accepted" else "fail"
        if status in ("assets_missing", "hardware_missing"):
            style = "incomplete"
        seconds = float(event.get("duration_ms", 0)) / 1000
        skipped = f", {event['skipped']} skipped" if event.get("skipped") else ""
        summary = (
            f"{event.get('passed', 0)}/{event.get('total', 0)} passed{skipped} in {seconds:.2f} s"
        )
        self.line(f"{self.paint(verdict, style)}  {summary}")
        if event.get("message"):
            self.line(f"  {event['message']}")
        if event.get("detail"):
            self.block("detail", event["detail"])
        for metric in event.get("metrics") or []:
            self.line(f"  {self.paint(format_metric(metric), 'dim')}")


def format_metric(metric: dict[str, Any]) -> str:
    value = metric.get("value")
    text = f"{value:.4g}" if isinstance(value, float) else str(value)
    unit = f" {metric['unit']}" if metric.get("unit") else ""
    limit = f" (limit {metric['limit']:g}{unit})" if "limit" in metric else ""
    return f"{metric.get('name')}: {text}{unit}{limit}"
