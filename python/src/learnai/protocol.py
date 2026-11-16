"""The harness event protocol (docs/ARCHITECTURE.md section 3.6).

Every protocol line on stdout is ``@@WS@@ `` followed by one single-line JSON object.
This module is the only place that knows the wire format; the worker, the supervisor,
and the CLI all go through it. It imports nothing heavy.
"""

from __future__ import annotations

import json
import math
from typing import Any

PREFIX = "@@WS@@ "
# Worker-to-supervisor bookkeeping lines. The supervisor consumes them; they never
# reach core or the CLI.
INTERNAL_PREFIX = "@@WSI@@ "

STDOUT_LIMIT_BYTES = 64 * 1024

CASE_STATUSES = ("pass", "fail", "error", "timeout", "skipped")
VERDICTS = (
    "accepted",
    "wrong_answer",
    "runtime_error",
    "time_limit",
    "import_error",
    "harness_error",
    "assets_missing",
    "hardware_missing",
)
#: Verdicts that mean "nothing failed, but not every case could run on this machine".
INCOMPLETE_VERDICTS = ("assets_missing", "hardware_missing")
#: Why a case was skipped (the ``kind`` of a skipped case's ``skip`` object):
#: its hardware is absent, its asset is not downloaded, or the run stopped before it.
SKIP_KINDS = ("hardware", "asset", "aborted")


def encode(event: dict[str, Any]) -> str:
    """Return the protocol line for ``event`` (without the trailing newline)."""
    return PREFIX + json.dumps(event, separators=(",", ":"), allow_nan=False, default=str)


def encode_internal(event: dict[str, Any]) -> str:
    return INTERNAL_PREFIX + json.dumps(event, separators=(",", ":"), allow_nan=False)


def decode(line: str) -> dict[str, Any] | None:
    """Parse a protocol line. Returns None for anything that is not one."""
    if not line.startswith(PREFIX):
        return None
    try:
        event = json.loads(line[len(PREFIX) :])
    except ValueError:
        return None
    return event if isinstance(event, dict) and "event" in event else None


def decode_internal(line: str) -> dict[str, Any] | None:
    if not line.startswith(INTERNAL_PREFIX):
        return None
    try:
        event = json.loads(line[len(INTERNAL_PREFIX) :])
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


def sanitize_log_line(line: str) -> str:
    """Make a diagnostic line safe to write to stdout: it must never look like protocol."""
    if line.startswith((PREFIX.rstrip(), INTERNAL_PREFIX.rstrip())):
        return " " + line
    return line


def truncate_output(data: bytes, total: int | None = None) -> str:
    """Decode captured output, keeping at most 64 KB and saying so when it was cut."""
    total = len(data) if total is None else total
    if total <= STDOUT_LIMIT_BYTES and len(data) <= STDOUT_LIMIT_BYTES:
        return data.decode("utf-8", errors="replace")
    head = data[:STDOUT_LIMIT_BYTES].decode("utf-8", errors="replace")
    return f"{head}\n... [output truncated: {total} bytes in total, first 64 KB shown]\n"


def verdict_for(
    statuses: list[str],
    *,
    import_failed: bool = False,
    skip_kinds: list[str] | tuple[str, ...] = (),
    hardware_met: bool = True,
) -> str:
    """Final verdict for a list of case statuses.

    ``skip_kinds`` has one entry of ``SKIP_KINDS`` per case that was skipped on
    purpose (``hardware``, ``asset``); a skipped status beyond those was skipped
    because the run stopped early (``aborted``). ``hardware_met`` says whether the
    machine has the hardware the problem's ``requires`` names.

    Precedence: import_error, runtime_error, time_limit, wrong_answer, assets_missing,
    hardware_missing, accepted. What ran decides first: a failure is a failure whatever
    was skipped. A skipped case never yields ``accepted``: cases skipped after a crash
    give ``wrong_answer`` (when nothing worse happened); with no failure at all, a case
    skipped for a missing asset gives ``assets_missing``, and a case skipped for
    missing hardware, or a problem whose required hardware is absent, gives
    ``hardware_missing``.
    """
    if import_failed:
        return "import_error"
    if "error" in statuses:
        return "runtime_error"
    if "timeout" in statuses:
        return "time_limit"
    deliberate = [kind for kind in skip_kinds if kind in ("hardware", "asset")]
    if "fail" in statuses or statuses.count("skipped") > len(deliberate):
        return "wrong_answer"
    if "asset" in deliberate:
        return "assets_missing"
    if "hardware" in deliberate or not hardware_met:
        return "hardware_missing"
    return "accepted"


def clean_metric(
    name: str,
    value: Any,
    unit: str | None = None,
    limit: Any = None,
    *,
    higher_is_better: bool = False,
) -> dict[str, Any] | None:
    """Build a metric object, or None when the value cannot be represented in JSON."""
    try:
        number = float(value)
    except TypeError, ValueError:
        return None
    if not math.isfinite(number):
        return None
    number = round(number, 6)  # drop float noise such as 10.500000000000009
    metric: dict[str, Any] = {
        "name": str(name),
        "value": int(number) if number.is_integer() and abs(number) < 2**53 else number,
    }
    if unit is not None:
        metric["unit"] = str(unit)
    if limit is not None:
        try:
            bound = float(limit)
        except TypeError, ValueError:
            bound = math.nan
        if math.isfinite(bound):
            metric["limit"] = int(bound) if bound.is_integer() and abs(bound) < 2**53 else bound
    if higher_is_better:
        metric["higher_is_better"] = True
    return metric


def summarize_metrics(case_metrics: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Run-level metrics for the ``end`` event: per metric name, the worst value seen.

    Almost every metric the judge reports is "lower is better" (tokens processed,
    slowdown ratio, latency, peak memory), so the worst case across cases is the
    largest value. A metric marked ``higher_is_better`` (tokens per second) is the
    other way around: its worst case is the smallest value.
    """
    worst: dict[str, dict[str, Any]] = {}
    for metrics in case_metrics:
        for metric in metrics:
            name = metric.get("name")
            if not isinstance(name, str):
                continue
            if name not in worst:
                worst[name] = metric
                continue
            value, seen = metric.get("value", 0), worst[name].get("value", 0)
            if (value < seen) if metric.get("higher_is_better") else (value > seen):
                worst[name] = metric
    return list(worst.values())
