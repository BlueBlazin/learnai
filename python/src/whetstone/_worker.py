"""The harness worker: loads the problem, imports the solution, runs the cases.

Run by the supervisor in ``whetstone.harness``; not a public entry point. Everything
the solution prints goes to a capture file (see ``capture.py``); this process writes
protocol lines to a private duplicate of the original stdout.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import signal
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

from whetstone import protocol
from whetstone.capture import FdCapture
from whetstone.problem import Problem, ProblemError, load_problem
from whetstone.tracebacks import exception_line, format_user_traceback, raised_in

MIN_IMPORT_LIMIT_S = 10.0


class CaseTimeout(BaseException):
    """Raised on the main thread by the soft time limit.

    Derives from BaseException so that a solution's ``except Exception`` does not
    swallow it.
    """


class HarnessFailure(Exception):
    """Something outside the solution is wrong (bad problem, missing framework)."""


@contextmanager
def soft_time_limit(seconds: float):
    """Raise CaseTimeout in the main thread after ``seconds`` (see harness.py)."""
    if not hasattr(signal, "setitimer"):  # no SIGALRM here: the hard limit still applies
        yield
        return

    def on_alarm(signum: int, frame: Any) -> None:
        raise CaseTimeout

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def import_file(name: str, path: Path) -> ModuleType:
    """Import the file at ``path`` as module ``name`` without touching sys.path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


class Worker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.capture = FdCapture(args.capture_file)
        self.started_at = time.perf_counter()
        self.ended = False

    # -- emitting --------------------------------------------------------------

    def emit(self, event: dict[str, Any]) -> None:
        self.capture.write_protocol(protocol.encode(event))

    def activity(self, **info: Any) -> None:
        self.capture.write_protocol(protocol.encode_internal(info))

    def end(
        self, status: str, statuses: list[str], total: int, metrics: list, **extra: Any
    ) -> None:
        event = {
            "event": "end",
            "status": status,
            "passed": sum(1 for s in statuses if s == "pass"),
            "total": total,
            "duration_ms": round((time.perf_counter() - self.started_at) * 1000, 3),
            "metrics": metrics,
        }
        event.update({k: v for k, v in extra.items() if v is not None})
        self.emit(event)
        self.ended = True

    def forward_captured_as_log(self) -> None:
        """Pass captured output through as diagnostic (non-protocol) stdout lines."""
        data, total = self.capture.take()
        text = protocol.truncate_output(data, total)
        for line in text.splitlines():
            self.capture.write_protocol(protocol.sanitize_log_line(line))

    # -- the run ---------------------------------------------------------------

    def run(self) -> int:
        # Imported here, after capture is in place, so that anything a framework
        # prints while loading cannot reach the protocol stream.
        from whetstone.judge import frameworks
        from whetstone.judge.context import ModuleStore
        from whetstone.judge.registry import load_cases

        args = self.args
        try:
            problem = load_problem(args.problem)
        except ProblemError as exc:
            raise HarnessFailure(str(exc)) from exc
        if args.framework not in problem.frameworks:
            raise HarnessFailure(
                f"problem {problem.id!r} does not support framework {args.framework!r} "
                f"(supported: {', '.join(problem.frameworks) or 'none'})"
            )
        solution_path = Path(args.solution).resolve()
        if not solution_path.is_file():
            raise HarnessFailure(f"solution file not found: {solution_path}")

        try:
            fw = frameworks.get(args.framework)
            oracle_fw = frameworks.get(problem.oracle_framework)
        except frameworks.FrameworkUnavailable as exc:
            raise HarnessFailure(str(exc)) from exc

        all_cases = [c for c in load_cases(problem.tests_path) if c.applies_to(args.framework)]
        hidden_names: dict[str, str] = {}
        for case in all_cases:
            if not case.public:
                hidden_names[case.id] = f"Hidden case {len(hidden_names) + 1}"
        cases = [c for c in all_cases if c.public or args.mode == "submit"]
        if args.case is not None:
            cases = [c for c in cases if c.id == args.case]
            if not cases:
                raise HarnessFailure(
                    f"no case with id {args.case!r} in {args.mode} mode for {problem.id!r}"
                )

        env = {
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.machine()}",
            "device": fw.device,
        }
        for adapter in {fw.name: fw, oracle_fw.name: oracle_fw}.values():
            if adapter.name != "python":
                env[adapter.name] = adapter.version
        self.emit(
            {
                "event": "start",
                "problem": problem.id,
                "framework": args.framework,
                "mode": args.mode,
                "cases": [
                    {"id": c.id, "name": hidden_names.get(c.id, c.name), "public": c.public}
                    for c in cases
                ],
                "env": env,
            }
        )

        solution, failure = self.import_solution(problem, solution_path)
        if solution is None:
            message, detail = failure
            statuses = []
            for case in cases:
                statuses.append("skipped")
                self.emit(
                    self.case_end(case.id, "skipped", 0.0, "not run: " + message, detail, "", [])
                )
            self.end("import_error", statuses, len(cases), [], message=message, detail=detail)
            return 0

        store = ModuleStore(problem, solution, fw, oracle_fw)
        statuses: list[str] = []
        case_metrics: list[list[dict[str, Any]]] = []
        for case in cases:
            result = self.run_case(problem, case, store, solution_path)
            statuses.append(result["status"])
            case_metrics.append(result["metrics"])
            self.emit(result)
        self.end(
            protocol.verdict_for(statuses),
            statuses,
            len(cases),
            protocol.summarize_metrics(case_metrics),
        )
        return 0

    def import_solution(
        self, problem: Problem, path: Path
    ) -> tuple[ModuleType | None, tuple[str, str | None]]:
        # The override exists so the judge's own tests need not wait ten seconds.
        floor = float(os.environ.get("WHETSTONE_IMPORT_LIMIT_S", MIN_IMPORT_LIMIT_S))
        limit = max(problem.time_limit_s, floor)
        self.capture.begin()
        self.activity(phase="import", limit_s=limit)
        failure: tuple[str, str | None] = ("", None)
        module: ModuleType | None = None
        try:
            with soft_time_limit(limit):
                module = import_file("solution", path)
        except CaseTimeout:
            failure = (f"importing the solution did not finish within {limit:g} s", None)
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # noqa: BLE001 - a solution may raise anything
            module = None
            failure = (
                f"the solution failed to import: {exception_line(exc)}",
                format_user_traceback(exc, str(path)),
            )
        if module is not None:
            missing = [name for name in problem.entry if not hasattr(module, name)]
            if missing:
                names = ", ".join(f"`{name}`" for name in missing)
                detail = (
                    f"{path.name} must define: {', '.join(problem.entry)}\n"
                    f"missing: {', '.join(missing)}\n"
                )
                failure = (f"the solution does not define {names}", detail)
                module = None
        self.activity(phase="idle")
        # Import-time prints belong to no case; surface them as diagnostic log lines.
        self.forward_captured_as_log()
        return module, failure

    def case_end(
        self,
        case_id: str,
        status: str,
        duration_ms: float,
        message: str | None,
        detail: str | None,
        stdout: str,
        metrics: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "event": "case_end",
            "id": case_id,
            "status": status,
            "duration_ms": round(duration_ms, 3),
            "message": message,
            "detail": detail,
            "stdout": stdout,
            "metrics": metrics,
        }

    def run_case(self, problem: Problem, case: Any, store: Any, solution_path: Path) -> dict:
        from whetstone.judge.context import Context
        from whetstone.judge.errors import CheckFailure

        limit = case.time_limit_s if case.time_limit_s is not None else problem.time_limit_s
        ctx = Context(store, case)
        status, message, detail = "pass", None, None
        user_file = str(solution_path)

        self.emit({"event": "case_start", "id": case.id})
        self.capture.begin()
        self.activity(phase="case", id=case.id, limit_s=limit)
        started = time.perf_counter()
        try:
            try:
                with soft_time_limit(limit), ctx.active():
                    case.fn(ctx)
            finally:
                ctx.close()
        except CaseTimeout:
            status, message = "timeout", f"time limit exceeded ({limit:g} s)"
        except KeyboardInterrupt:
            raise
        except CheckFailure as failure:
            status, message, detail = "fail", failure.message, failure.detail
            if not case.public:
                detail = None  # may contain expected values
        except AssertionError as exc:
            if raised_in(exc, user_file):
                # The learner's own assert fired: that is a crash, not a wrong answer.
                status, message = "error", exception_line(exc)
                detail = format_user_traceback(exc, user_file)
            else:
                status = "fail"
                text = str(exc).strip()
                message = text.splitlines()[0] if text else "an assertion in the test failed"
                rest = "\n".join(text.splitlines()[1:]).strip()
                detail = rest or None
                if not case.public:
                    detail = None
        except BaseException as exc:  # noqa: BLE001 - includes SystemExit from the solution
            status, message = "error", exception_line(exc)
            also_show = (str(problem.tests_path),) if case.public else ()
            detail = format_user_traceback(exc, user_file, also_show)
        duration_ms = (time.perf_counter() - started) * 1000
        self.activity(phase="idle")
        data, total = self.capture.take()
        if message is not None:
            message = " ".join(message.split())  # one line
        return self.case_end(
            case.id,
            status,
            duration_ms,
            message,
            detail,
            protocol.truncate_output(data, total),
            ctx.metrics,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="whetstone._worker")
    parser.add_argument("--problem", required=True)
    parser.add_argument("--solution", required=True)
    parser.add_argument("--framework", required=True)
    parser.add_argument("--mode", required=True, choices=["run", "submit"])
    parser.add_argument("--case", default=None)
    parser.add_argument("--capture-file", required=True)
    args = parser.parse_args(argv)

    worker = Worker(args)
    try:
        return worker.run()
    except KeyboardInterrupt:
        return 130
    except BaseException as exc:  # noqa: BLE001 - never die without an `end` event
        if isinstance(exc, HarnessFailure):
            message, detail = str(exc), None
        else:
            message = f"harness failure: {exception_line(exc)}"
            detail = "".join(traceback.format_exception(exc))
        worker.forward_captured_as_log()
        if not worker.ended:
            worker.end("harness_error", [], 0, [], message=message, detail=detail)
        worker.capture.write_diagnostic(f"whetstone.harness: {message}\n")
        return 1


if __name__ == "__main__":
    # os._exit skips interpreter teardown: framework atexit hooks and stray solution
    # threads must not be able to delay or break the exit after `end` is out.
    code = main()
    sys.stdout.flush()
    os._exit(code)
