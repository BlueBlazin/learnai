"""The harness worker: loads the problem, imports the solution, runs the cases.

Run by the supervisor in ``learnai.harness``; not a public entry point. Everything
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

from learnai import assets, hardware, protocol
from learnai.capture import FdCapture
from learnai.problem import Problem, ProblemError, load_problem
from learnai.tracebacks import (
    exception_line,
    format_user_traceback,
    hidden_error_message,
    raised_in,
)

MIN_IMPORT_LIMIT_S = 10.0
#: How long a block that runs off the clock may take before the supervisor gives up on
#: the process. Generous: such a block is library code, never the solution's.
OFF_THE_CLOCK_ALLOWANCE_S = 120.0


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
        #: Real case id -> the id that leaves the harness (opaque for hidden cases).
        self.wire: dict[str, str] = {}
        self.guard: Any = None  # forbidden-call guard, when the problem has a list
        #: Real case id -> (skip object, message) for cases this machine cannot run.
        self.skips: dict[str, tuple[dict[str, str], str]] = {}
        self.hardware_met = True
        #: What the supervisor was last told this process is doing (import or case).
        self.current: dict[str, Any] | None = None
        #: Seconds of the current phase spent with the clock stopped (judge/warmup.py).
        self.paused_s = 0.0
        self._pause_depth = 0

    # -- emitting --------------------------------------------------------------

    def emit(self, event: dict[str, Any]) -> None:
        self.capture.write_protocol(protocol.encode(event))

    def activity(self, **info: Any) -> None:
        self.current = info if info.get("phase") in ("import", "case") else None
        self.paused_s = 0.0
        self.capture.write_protocol(protocol.encode_internal(info))

    @contextmanager
    def off_the_clock(self):
        """Stop the clock of the running phase for the length of the block.

        Used for one-time library costs that are not the solution's doing (see
        ``learnai.judge.warmup``). The soft time limit is suspended and restored
        with what was left of it; the supervisor is told to wait (``deadline_s``: how
        long from now until the soft limit would fire), and again when the block ends;
        the time is added to ``paused_s``, which ``run_case`` subtracts.
        """
        if self._pause_depth or not hasattr(signal, "setitimer"):
            yield
            return
        self._pause_depth += 1
        remaining, _ = signal.setitimer(signal.ITIMER_REAL, 0)
        if self.current is not None:
            waiting = {**self.current, "deadline_s": remaining + OFF_THE_CLOCK_ALLOWANCE_S}
            self.capture.write_protocol(protocol.encode_internal(waiting))
        started = time.perf_counter()
        try:
            yield
        finally:
            self.paused_s += time.perf_counter() - started
            self._pause_depth -= 1
            if self.current is not None:
                resumed = {**self.current, "deadline_s": remaining}
                self.capture.write_protocol(protocol.encode_internal(resumed))
            if remaining > 0:
                signal.setitimer(signal.ITIMER_REAL, remaining)

    def end(
        self, status: str, statuses: list[str], total: int, metrics: list, **extra: Any
    ) -> None:
        event = {
            "event": "end",
            "status": status,
            "passed": sum(1 for s in statuses if s == "pass"),
            "total": total,
            "skipped": sum(1 for s in statuses if s == "skipped"),
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
        from learnai.judge import frameworks, warmup
        from learnai.judge.context import ModuleStore
        from learnai.judge.registry import load_cases, wire_identity

        warmup.install(self.off_the_clock)
        warmup.watch_imports()
        args = self.args
        try:
            problem = load_problem(args.problem)
        except ProblemError as exc:
            raise HarnessFailure(str(exc)) from exc
        if problem.asset_errors:
            raise HarnessFailure("problem.toml: " + "; ".join(problem.asset_errors))
        if args.framework not in problem.frameworks:
            raise HarnessFailure(
                f"problem {problem.id!r} does not support framework {args.framework!r} "
                f"(supported: {', '.join(problem.frameworks) or 'none'})"
            )
        solution_path = Path(args.solution).resolve()
        if not solution_path.is_file():
            raise HarnessFailure(f"solution file not found: {solution_path}")

        if "jax" in (args.framework, problem.oracle_framework):
            cache = assets.cache_dir(assets.root_of(problem.path), "jax")
            frameworks.use_jax_compilation_cache(cache)
        try:
            fw = frameworks.get(args.framework)
            oracle_fw = frameworks.get(problem.oracle_framework)
        except frameworks.FrameworkUnavailable as exc:
            raise HarnessFailure(str(exc)) from exc

        all_cases = load_cases(problem.tests_path)
        # What a hidden case is called and what it checks stays in this process: on
        # the wire it is `hidden-N` / "Hidden case N" (unless an author asks to reveal).
        identity = wire_identity(all_cases, reveal=args.reveal)
        self.wire = {case_id: wire_id for case_id, (wire_id, _) in identity.items()}
        cases = [
            c
            for c in all_cases
            if c.applies_to(args.framework) and (c.public or args.mode == "submit")
        ]
        if args.case is not None:
            cases = [c for c in cases if self.wire[c.id] == args.case]
            if not cases:
                raise HarnessFailure(
                    f"no case with id {args.case!r} in {args.mode} mode for {problem.id!r}"
                )

        models = assets.models_dir(assets.root_of(problem.path))
        self.plan_skips(problem, cases, models)
        runnable = [c for c in cases if c.id not in self.skips]

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
                "cases": [self.case_info(c, identity) for c in cases],
                "env": env,
                "requires": problem.requires,
                "requires_met": self.hardware_met,
            }
        )

        if not runnable:
            # Nothing this machine can run: the solution is not even imported.
            for case in cases:
                self.emit(self.skipped(case))
            self.finish(problem, cases, ["skipped"] * len(cases), [])
            return 0

        if problem.forbidden or problem.forbidden_source:
            from learnai.judge.forbidden import Guard

            # In place before the solution is imported, so that `from torch import
            # softmax` binds the guard, not the function.
            self.guard = Guard(problem.forbidden, solution_path, problem.forbidden_source)
            self.guard.install()
            found = self.guard.scan_source()
            if found is not None:
                # Something that can only be detected in the source: nothing is run,
                # and every case this machine can run fails for that reason.
                detail = self.guard.rule_text()
                statuses = []
                for case in cases:
                    if case.id in self.skips:
                        self.emit(self.skipped(case))
                        statuses.append("skipped")
                        continue
                    wire_id = self.wire[case.id]
                    shown = detail if case.public or args.reveal else None
                    self.emit({"event": "case_start", "id": wire_id})
                    self.emit(self.case_end(wire_id, "fail", 0.0, found.message, shown, "", []))
                    statuses.append("fail")
                self.finish(problem, cases, statuses, [])
                return 0

        solution, failure = self.import_solution(problem, solution_path)
        if solution is None:
            message, detail = failure
            statuses = []
            for case in cases:
                statuses.append("skipped")
                if case.id in self.skips:
                    self.emit(self.skipped(case))
                    continue
                # One line only: the traceback is on the `end` event, once.
                self.emit(
                    self.case_end(
                        self.wire[case.id], "skipped", 0.0, "not run: " + message, None, "", [],
                        skip={"kind": "aborted"},
                    )
                )  # fmt: skip
            self.end("import_error", statuses, len(cases), [], message=message, detail=detail)
            return 0

        store = ModuleStore(problem, solution, fw, oracle_fw, models_dir=models)
        statuses = []
        case_metrics: list[list[dict[str, Any]]] = []
        for case in cases:
            if case.id in self.skips:
                self.emit(self.skipped(case))
                statuses.append("skipped")
                continue
            result = self.run_case(problem, case, store, solution_path)
            statuses.append(result["status"])
            case_metrics.append(result["metrics"])
            self.emit(result)
        self.finish(problem, cases, statuses, protocol.summarize_metrics(case_metrics))
        return 0

    # -- cases this machine cannot run ---------------------------------------------

    def plan_skips(self, problem: Problem, cases: list, models: Path) -> None:
        """Decide, before anything runs, which cases this machine cannot run and why.

        A case is skipped when the hardware it ``requires`` is absent, or when an
        asset it needs is not downloaded. Hardware is checked first: fetching weights
        would not help on a machine that cannot run the case anyway.
        """
        if problem.requires not in hardware.REQUIREMENTS:
            raise HarnessFailure(
                f"problem.toml: requires must be one of {', '.join(hardware.REQUIREMENTS)}"
            )
        self.hardware_met = hardware.available(problem.requires)
        states = {spec.name: assets.state_of(spec, models) for spec in problem.assets}
        for case in cases:
            if case.requires != "cpu":
                if case.requires != problem.requires:
                    raise HarnessFailure(
                        f"case {case.id!r} has requires={case.requires!r}, but problem.toml "
                        f"says requires = {problem.requires!r}; a case may require only "
                        '"cpu" or what the problem requires'
                    )
                if not self.hardware_met:
                    self.skips[case.id] = (
                        {"kind": "hardware", "requires": case.requires},
                        "skipped: " + hardware.missing_message(case.requires),
                    )
                    continue
            for name in case.assets if case.assets is not None else tuple(states):
                if name not in states:
                    raise HarnessFailure(
                        f"case {case.id!r} names the asset {name!r}, which problem.toml "
                        "does not declare in `[[assets]]`"
                    )
                if not states[name].present:
                    self.skips[case.id] = (
                        {"kind": "asset", "asset": name},
                        "skipped: " + assets.missing_message(states[name].spec),
                    )
                    break

    def case_info(self, case: Any, identity: dict[str, tuple[str, str]]) -> dict[str, Any]:
        info: dict[str, Any] = {
            "id": identity[case.id][0],
            "name": identity[case.id][1],
            "public": case.public,
        }
        if case.requires != "cpu":
            info["requires"] = case.requires
        if case.stage is not None:
            info["stage"] = case.stage  # public also for a hidden case
        return info

    def skipped(self, case: Any) -> dict[str, Any]:
        skip, message = self.skips[case.id]
        return self.case_end(self.wire[case.id], "skipped", 0.0, message, None, "", [], skip=skip)

    def finish(self, problem: Problem, cases: list, statuses: list[str], metrics: list) -> None:
        """Emit ``end`` with the verdict, explaining an incomplete run in words."""
        kinds = [self.skips[c.id][0]["kind"] for c in cases if c.id in self.skips]
        verdict = protocol.verdict_for(statuses, skip_kinds=kinds, hardware_met=self.hardware_met)
        extra: dict[str, Any] = {}
        passed = sum(1 for s in statuses if s == "pass")
        ran = len(statuses) - len(kinds)
        if verdict == "assets_missing":
            names = dict.fromkeys(
                self.skips[c.id][0]["asset"]
                for c in cases
                if c.id in self.skips and self.skips[c.id][0]["kind"] == "asset"
            )
            specs = [problem.asset(name) for name in names]
            commands = " and ".join(f"`{spec.fetch_command}`" for spec in specs)
            which = ", ".join(spec.describe() for spec in specs)
            ran_text = f"{passed} of {ran} cases that could run passed; " if ran else ""
            extra["message"] = (
                f"{ran_text}{len(kinds)} of {len(cases)} cases were skipped because a model "
                f"asset is not downloaded: {which}. Fetch with {commands}, then run again."
            )
            extra["missing_assets"] = [
                {
                    "name": spec.name,
                    "repo": spec.repo,
                    "size_mb": spec.size_mb,
                    "fetch_command": spec.fetch_command,
                }
                for spec in specs
            ]
        elif verdict == "hardware_missing":
            need = hardware.describe(problem.requires)
            if kinds:
                ran_text = f"{passed} of {ran} cases that could run passed; " if ran else ""
                extra["message"] = (
                    f"{ran_text}{len(kinds)} of {len(cases)} cases were skipped because they "
                    f"need {need}, which this machine does not have. The solution can only "
                    "be accepted on a machine that runs every case."
                )
            else:
                extra["message"] = (
                    f"every case passed, but this problem requires {need}, which this "
                    "machine does not have, so the run does not count as accepted"
                )
            extra["missing_hardware"] = problem.requires
        self.end(verdict, statuses, len(cases), metrics, **extra)

    def import_solution(
        self, problem: Problem, path: Path
    ) -> tuple[ModuleType | None, tuple[str, str | None]]:
        # The override exists so the judge's own tests need not wait ten seconds.
        floor = float(os.environ.get("LEARNAI_IMPORT_LIMIT_S", MIN_IMPORT_LIMIT_S))
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
            line = exception_line(exc)
            if type(exc).__name__ == "ForbiddenCall":
                line = str(exc)
            failure = (
                f"the solution failed to import: {line}",
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

    @staticmethod
    def describe_error(
        exc: BaseException, user_file: str, also_show: tuple[str, ...], shown: bool
    ) -> tuple[str, str, str | None]:
        """Status, message, and detail for an exception that escaped a case.

        Public case: the exception line, and a traceback of the learner's frames (or
        the test's, when the learner's file is not involved). Hidden case: everything
        the learner needs to debug their own code fits the message (their exception
        type and text, and the line of their file); there is no detail.
        """
        if shown:
            return "error", exception_line(exc), format_user_traceback(exc, user_file, also_show)
        return "error", hidden_error_message(exc, user_file), None

    def case_end(
        self,
        case_id: str,
        status: str,
        duration_ms: float,
        message: str | None,
        detail: str | None,
        stdout: str,
        metrics: list[dict[str, Any]],
        *,
        skip: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        event = {
            "event": "case_end",
            "id": case_id,
            "status": status,
            "duration_ms": round(duration_ms, 3),
            "message": message,
            "detail": detail,
            "stdout": stdout,
            "metrics": metrics,
        }
        if skip is not None:
            event["skip"] = skip  # why a skipped case did not run (protocol.SKIP_KINDS)
        return event

    def run_case(self, problem: Problem, case: Any, store: Any, solution_path: Path) -> dict:
        from learnai.judge.context import Context
        from learnai.judge.errors import CheckFailure

        limit = case.time_limit_s if case.time_limit_s is not None else problem.time_limit_s
        ctx = Context(store, case)
        status, message, detail = "pass", None, None
        user_file = str(solution_path)
        wire_id = self.wire[case.id]
        # Hidden cases keep the one-line message and never carry a detail: it could
        # show expected values, the test's inputs, or its source.
        shown = case.public or self.args.reveal
        if self.guard is not None:
            self.guard.install()  # picks up frameworks the solution imported since
            self.guard.take()

        self.emit({"event": "case_start", "id": wire_id})
        self.capture.begin()
        self.activity(phase="case", id=wire_id, limit_s=limit)
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
        except AssertionError as exc:
            if raised_in(exc, user_file):
                # The learner's own assert fired: that is a crash, not a wrong answer.
                status, message, detail = self.describe_error(exc, user_file, (), shown)
            else:
                status = "fail"
                text = str(exc).strip()
                message = text.splitlines()[0] if text else "an assertion in the test failed"
                rest = "\n".join(text.splitlines()[1:]).strip()
                detail = rest or None
        except BaseException as exc:  # noqa: BLE001 - includes SystemExit from the solution
            also_show = (str(problem.tests_path),)
            status, message, detail = self.describe_error(exc, user_file, also_show, shown)
        duration_ms = max(0.0, time.perf_counter() - started - self.paused_s) * 1000
        self.activity(phase="idle")
        data, total = self.capture.take()
        violations = self.guard.take() if self.guard is not None else []
        if violations and status != "timeout":
            # Also when the solution caught the exception and carried on: the rule of
            # the exercise was broken either way.
            status, message, detail = "fail", violations[0].message, self.guard.rule_text()
        if not shown:
            detail = None
        if message is not None:
            message = " ".join(message.split())  # one line
        return self.case_end(
            wire_id,
            status,
            duration_ms,
            message,
            detail,
            protocol.truncate_output(data, total),
            ctx.metrics,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="learnai._worker")
    parser.add_argument("--problem", required=True)
    parser.add_argument("--solution", required=True)
    parser.add_argument("--framework", required=True)
    parser.add_argument("--mode", required=True, choices=["run", "submit"])
    parser.add_argument("--case", default=None)
    parser.add_argument("--reveal", action="store_true")
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
        worker.capture.write_diagnostic(f"learnai.harness: {message}\n")
        return 1


if __name__ == "__main__":
    # os._exit skips interpreter teardown: framework atexit hooks and stray solution
    # threads must not be able to delay or break the exit after `end` is out.
    code = main()
    sys.stdout.flush()
    os._exit(code)
