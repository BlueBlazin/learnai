"""The harness entry point: ``python -m whetstone.harness`` (ARCHITECTURE 3.5 and 3.6).

This module is the *supervisor*. It imports only the standard library, spawns the
worker (``whetstone._worker``) that actually loads frameworks and runs the solution,
forwards the worker's protocol lines verbatim, and guarantees the two promises the
worker cannot make about itself:

1. Time limits always fire. See "Time limits" below.
2. The stream always finishes with an ``end`` event, even if the solution takes the
   interpreter down (segfault in a C extension, ``os._exit``, out of memory).

Time limits
-----------
Each case has two limits.

* Soft (in the worker): a ``SIGALRM`` timer raises ``CaseTimeout`` on the main thread
  when the case's limit expires. This stops any pure-Python loop, reports the case as
  ``timeout``, and lets the remaining cases run in the same process, so one slow case
  does not cost the learner the rest of their results.
* Hard (here): a signal handler only runs between bytecodes, so the soft limit cannot
  interrupt a long call inside a C extension, and a solution can swallow the exception
  with a bare ``except``. A watchdog thread inside the worker would not help either,
  because a C call that holds the GIL starves every Python thread. The only watchdog
  that cannot be starved is another process. If the worker has not finished the case
  ``HARD_GRACE_S`` seconds after the soft limit, the supervisor kills it, reports the
  case as ``timeout`` (with whatever it printed, read from the capture file), reports
  the remaining cases as ``skipped``, and emits ``end``.

The same watchdog covers importing the solution (a top-level infinite loop).
"""

from __future__ import annotations

import argparse
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

from whetstone import protocol

HARD_GRACE_S = 2.0


class UsageError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise UsageError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="python -m whetstone.harness",
        description="Run a solution against a problem and emit @@WS@@ JSONL events.",
    )
    parser.add_argument("--problem", required=True, help="absolute path of the problem directory")
    parser.add_argument("--solution", required=True, help="absolute path of the solution file")
    parser.add_argument("--framework", required=True, choices=["torch", "jax", "mlx", "python"])
    parser.add_argument("--mode", required=True, choices=["run", "submit"])
    parser.add_argument("--case", default=None, help="run only the case with this id")
    return parser


def _emit(event: dict[str, Any]) -> None:
    sys.stdout.write(protocol.encode(event) + "\n")
    sys.stdout.flush()


def _end_event(status: str, started: float, **extra: Any) -> dict[str, Any]:
    event = {
        "event": "end",
        "status": status,
        "passed": 0,
        "total": 0,
        "duration_ms": round((time.monotonic() - started) * 1000, 3),
        "metrics": [],
    }
    event.update(extra)
    return event


class Supervisor:
    def __init__(self, worker_args: list[str], hard_grace_s: float) -> None:
        self.worker_args = worker_args
        self.hard_grace_s = hard_grace_s
        self.started_at = time.monotonic()
        self.cases: list[dict[str, Any]] = []
        self.statuses: dict[str, str] = {}
        self.case_metrics: list[list[dict[str, Any]]] = []
        self.saw_start = False
        self.final_status: str | None = None
        # What the worker is doing right now: None, or {"phase": "import" | "case", ...}.
        self.activity: dict[str, Any] | None = None
        self.deadline: float | None = None

    # -- line handling ---------------------------------------------------------

    def _handle(self, line: str) -> None:
        internal = protocol.decode_internal(line)
        if internal is not None:
            self._set_activity(internal)
            return
        event = protocol.decode(line)
        if event is None:
            # Diagnostic output; core shows it as a log line.
            sys.stdout.write(protocol.sanitize_log_line(line) + "\n")
            sys.stdout.flush()
            return
        kind = event.get("event")
        if kind == "start":
            self.saw_start = True
            self.cases = [c for c in event.get("cases", []) if isinstance(c, dict)]
        elif kind == "case_end":
            self.statuses[str(event.get("id"))] = str(event.get("status"))
            self.case_metrics.append(event.get("metrics") or [])
            self._set_activity({"phase": "idle"})
        elif kind == "end":
            self.final_status = str(event.get("status"))
            self._set_activity({"phase": "idle"})
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def _set_activity(self, info: dict[str, Any]) -> None:
        if info.get("phase") in ("import", "case"):
            self.activity = info
            self.deadline = time.monotonic() + float(info.get("limit_s", 10.0)) + self.hard_grace_s
        else:
            self.activity = None
            self.deadline = None

    # -- synthesizing what the worker could not say ----------------------------

    def _finish_for_worker(
        self, *, killed_for_timeout: bool, returncode: int, capture: str
    ) -> None:
        how = _describe_exit(returncode)
        if not self.saw_start:
            _emit(
                _end_event(
                    "harness_error",
                    self.started_at,
                    message=f"the harness worker {how} before it could start",
                )
            )
            self.final_status = "harness_error"
            return

        activity = self.activity or {}
        import_failed = False
        end_extra: dict[str, Any] = {}
        skip_reason = "not run: the process stopped during an earlier case"

        if activity.get("phase") == "import":
            import_failed = True
            if killed_for_timeout:
                message = (
                    f"importing the solution did not finish within {activity.get('limit_s'):g} s"
                )
            else:
                message = f"the process {how} while importing the solution"
            end_extra = {"message": message, "detail": _read_capture(capture) or None}
            skip_reason = "not run: the solution failed to import"
        elif activity.get("phase") == "case":
            case_id = str(activity.get("id"))
            limit = float(activity.get("limit_s", 0.0))
            if killed_for_timeout:
                status = "timeout"
                message = (
                    f"time limit exceeded ({limit:g} s); the process had to be stopped, "
                    "so the remaining cases were not run"
                )
                duration_ms = (limit + self.hard_grace_s) * 1000
            else:
                status = "error"
                message = f"the process {how} while running this case"
                duration_ms = 0.0
            self.statuses[case_id] = status
            _emit(
                {
                    "event": "case_end",
                    "id": case_id,
                    "status": status,
                    "duration_ms": round(duration_ms, 3),
                    "message": message,
                    "detail": None,
                    "stdout": _read_capture(capture),
                    "metrics": [],
                }
            )
        else:
            end_extra = {"message": f"the harness worker {how} without finishing the run"}

        for case in self.cases:
            case_id = str(case.get("id"))
            if case_id in self.statuses:
                continue
            self.statuses[case_id] = "skipped"
            _emit(
                {
                    "event": "case_end",
                    "id": case_id,
                    "status": "skipped",
                    "duration_ms": 0.0,
                    "message": skip_reason,
                    "detail": None,
                    "stdout": "",
                    "metrics": [],
                }
            )

        statuses = list(self.statuses.values())
        if activity.get("phase") in ("import", "case"):
            verdict = protocol.verdict_for(statuses, import_failed=import_failed)
        else:
            verdict = "harness_error"
        self.final_status = verdict
        _emit(
            _end_event(
                verdict,
                self.started_at,
                passed=sum(1 for s in statuses if s == "pass"),
                total=len(self.cases),
                metrics=protocol.summarize_metrics(self.case_metrics),
                **end_extra,
            )
        )

    # -- main loop -------------------------------------------------------------

    def run(self) -> int:
        with tempfile.TemporaryDirectory(prefix="whetstone-") as tmp:
            capture = os.path.join(tmp, "capture.txt")
            # -P keeps the working directory off sys.path: a solution is typically
            # named torch.py or jax.py, and running from its directory must not make
            # "import torch" find it.
            command = [sys.executable, "-P", "-m", "whetstone._worker", *self.worker_args]
            command += ["--capture-file", capture]
            proc = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=None,
                env={**os.environ, "PYTHONSAFEPATH": "1"},
            )
            lines: queue.Queue[str | None] = queue.Queue()

            def pump() -> None:
                assert proc.stdout is not None
                for raw in proc.stdout:
                    lines.put(raw.decode("utf-8", errors="replace").rstrip("\n"))
                lines.put(None)

            threading.Thread(target=pump, daemon=True).start()

            killed_for_timeout = False
            try:
                while True:
                    wait = None
                    if self.deadline is not None and not killed_for_timeout:
                        wait = max(0.0, self.deadline - time.monotonic())
                    try:
                        line = lines.get(timeout=wait)
                    except queue.Empty:
                        # Hard limit. Kill, then keep draining: the worker may have
                        # finished the case in the instant before it died.
                        killed_for_timeout = True
                        proc.kill()
                        continue
                    if line is None:
                        break
                    self._handle(line)
            except KeyboardInterrupt:
                proc.kill()
                proc.wait()
                return 130
            returncode = proc.wait()
            if self.final_status is None:
                self._finish_for_worker(
                    killed_for_timeout=killed_for_timeout, returncode=returncode, capture=capture
                )
        return 1 if self.final_status == "harness_error" else 0


def _describe_exit(returncode: int) -> str:
    if returncode < 0:
        try:
            name = signal.Signals(-returncode).name
        except ValueError:
            name = f"signal {-returncode}"
        return f"was killed by {name}"
    return f"exited with code {returncode}"


def _read_capture(path: str) -> str:
    try:
        with open(path, "rb") as handle:
            data = handle.read(protocol.STDOUT_LIMIT_BYTES + 1)
        return protocol.truncate_output(data, os.path.getsize(path))
    except OSError:
        return ""


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        build_parser().parse_args(argv)
    except UsageError as exc:
        _emit(_end_event("harness_error", started, message=f"usage error: {exc}"))
        sys.stderr.write(f"whetstone.harness: {exc}\n")
        return 2
    grace = float(os.environ.get("WHETSTONE_HARD_GRACE_S", HARD_GRACE_S))
    supervisor = Supervisor(argv, grace)
    try:
        return supervisor.run()
    except Exception as exc:  # noqa: BLE001 - last line of defense: always emit `end`
        if supervisor.final_status is None:
            message = f"harness supervisor failed: {exc!r}"
            _emit(_end_event("harness_error", started, message=message))
        return 1


if __name__ == "__main__":
    sys.exit(main())
