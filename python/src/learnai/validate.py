"""``learnai validate``: the quality gate for problems (docs/ARCHITECTURE.md section 3.7).

For every supported framework of a problem: the reference is accepted and the starter
is not. Every mutant is rejected. Metadata is complete and consistent. This module
returns findings; the CLI prints them.

A machine cannot always run every case: a case may need hardware it lacks
(``requires``) or a model asset that is not downloaded (``[[assets]]``). The harness
reports such cases as skipped and ends with ``hardware_missing`` or ``assets_missing``
instead of ``accepted``. Validation then judges what did run and says, as a warning on
the problem, what was not validated here; a problem of which nothing could run is
reported as skipped, never as passed. ``require_assets`` turns a missing asset into an
error.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from learnai import assets, hardware, runner
from learnai.problem import (
    DIFFICULTIES,
    FRAMEWORKS,
    JUDGES,
    REQUIRES,
    TENSOR_FRAMEWORKS,
    Problem,
    load_problem,
    problem_dirs,
)
from learnai.protocol import INCOMPLETE_VERDICTS

ID_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
#: The same pattern as learnai.judge.forbidden.NAME_PATTERN, which is not imported
#: here because the judge package pulls in numpy and the CLI should start fast.
FORBIDDEN_NAME = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)+$")
PLACEHOLDER = "TODO(author)"
#: A full submit should finish in a few seconds. Process startup and framework imports
#: are a fixed cost, so the warning looks at what the author controls: the cases.
SLOW_CASES_S = 3.0
REJECTED = ("wrong_answer", "runtime_error", "time_limit")
#: Judges whose cases depend on real time: `speed` measures it and `service` waits on
#: sockets with timeouts. Their harness runs are not mixed into the parallel batch,
#: where other runs compete for the cores; they run one at a time once it is done.
TIMING_SENSITIVE_JUDGES = ("speed", "service")
#: Importing every framework to resolve `forbidden` names takes a few seconds.
FORBIDDEN_CHECK_TIMEOUT_S = 120

_REQUIRED_KEYS: dict[str, type | tuple[type, ...]] = {
    "id": str,
    "title": str,
    "track": str,
    "order": int,
    "difficulty": str,
    "frameworks": list,
    "judges": list,
    "tags": list,
    "prerequisites": list,
    "requires": str,
    "entry": list,
    "time_limit_s": (int, float),
}


@dataclass
class Report:
    """Findings for one problem."""

    problem_id: str
    path: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)  # what passed, for the printed report
    #: Why nothing of some reference run could be judged on this machine.
    skips: list[str] = field(default_factory=list)
    #: Whether at least one reference ran at least one case.
    ran: bool = False

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def skipped(self) -> bool:
        """Nothing is wrong, but nothing of the behavior could be checked here."""
        return self.ok and bool(self.skips) and not self.ran

    @property
    def status(self) -> str:
        """``ok``, ``fail``, or ``skipped``."""
        if not self.ok:
            return "fail"
        return "skipped" if self.skipped else "ok"


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and bool(path.read_text(encoding="utf-8").strip())
    except OSError, UnicodeDecodeError:
        return False


def check_metadata(directory: Path, catalog_ids: dict[str, list[Path]]) -> tuple[Report, Any]:
    """Static checks. Returns the report and the Problem (None if it cannot be loaded)."""
    report = Report(directory.name, directory)
    err = report.errors.append
    toml_path = directory / "problem.toml"
    try:
        raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        err(f"problem.toml: cannot be read: {exc}")
        return report, None

    for key, kind in _REQUIRED_KEYS.items():
        if key not in raw:
            err(f"problem.toml: missing key `{key}`")
        elif not isinstance(raw[key], kind) or isinstance(raw[key], bool):
            err(f"problem.toml: `{key}` has the wrong type")
    lists = ("frameworks", "judges", "tags", "prerequisites", "entry", "forbidden")
    for key in (*lists, "forbidden_source"):
        value = raw.get(key)
        if isinstance(value, list) and not all(isinstance(item, str) for item in value):
            err(f"problem.toml: `{key}` must be a list of strings")
    if "forbidden" in raw and not isinstance(raw["forbidden"], list):
        err("problem.toml: `forbidden` must be a list of dotted names")
    if "forbidden_source" in raw and not isinstance(raw["forbidden_source"], list):
        err("problem.toml: `forbidden_source` must be a list of identifiers")
    if report.errors:
        return report, None

    problem = load_problem(directory)
    if problem.id != directory.name:
        err(f"problem.toml: id {problem.id!r} must equal the directory name {directory.name!r}")
    if not ID_PATTERN.match(problem.id):
        err(f"problem.toml: id {problem.id!r} must be lowercase words joined by hyphens")
    if problem.track != directory.parent.name:
        err(
            f"problem.toml: track {problem.track!r} must equal the parent directory "
            f"name {directory.parent.name!r}"
        )
    if len(catalog_ids.get(problem.id, [])) > 1:
        others = ", ".join(str(p) for p in catalog_ids[problem.id] if p != directory)
        err(f"problem.toml: id {problem.id!r} is not globally unique (also at {others})")
    if not problem.title.strip():
        err("problem.toml: title is empty")
    if problem.difficulty not in DIFFICULTIES:
        err(f"problem.toml: difficulty must be one of {', '.join(DIFFICULTIES)}")
    if problem.requires not in REQUIRES:
        err(f"problem.toml: requires must be one of {', '.join(REQUIRES)}")
    if problem.time_limit_s <= 0:
        err("problem.toml: time_limit_s must be positive")

    frameworks = problem.frameworks
    if not frameworks:
        err("problem.toml: frameworks is empty")
    if len(set(frameworks)) != len(frameworks):
        err("problem.toml: frameworks contains duplicates")
    for name in frameworks:
        if name not in FRAMEWORKS:
            err(f"problem.toml: unknown framework {name!r}")
    if "python" in frameworks and len(frameworks) > 1:
        err('problem.toml: "python" cannot be combined with tensor frameworks')
    tensor = [f for f in frameworks if f in TENSOR_FRAMEWORKS]
    if tensor and "torch" not in frameworks and frameworks != ("mlx",):
        err(
            'problem.toml: tensor problems must support "torch" (the oracle is written in '
            'it); the one exception is an MLX-only problem, `frameworks = ["mlx"]`, whose '
            "reference/mlx.py is the oracle"
        )
    if problem.requires == "mlx-gpu" and "mlx" not in frameworks:
        err('problem.toml: requires = "mlx-gpu" needs "mlx" in frameworks')
    if problem.requires == "cuda" and "torch" not in frameworks:
        err('problem.toml: requires = "cuda" needs "torch" in frameworks')

    if not problem.judges:
        err("problem.toml: judges is empty")
    for name in problem.judges:
        if name not in JUDGES:
            err(f"problem.toml: unknown judge {name!r}")
    if not problem.entry:
        err("problem.toml: entry is empty")
    for name in problem.entry:
        if not name.isidentifier():
            err(f"problem.toml: entry {name!r} is not a valid Python name")
    if len(set(problem.forbidden)) != len(problem.forbidden):
        err("problem.toml: forbidden contains duplicates")
    for name in problem.forbidden:
        if not FORBIDDEN_NAME.match(name):
            err(
                f"problem.toml: forbidden name {name!r} must be a dotted name such as "
                "`torch.softmax` or `torch.Tensor.softmax`"
            )
    if len(set(problem.forbidden_source)) != len(problem.forbidden_source):
        err("problem.toml: forbidden_source contains duplicates")
    for name in problem.forbidden_source:
        if not name.isidentifier():
            err(
                f"problem.toml: forbidden_source entry {name!r} must be a single identifier "
                "such as `bfloat16` (dotted names that can be called go in `forbidden`)"
            )
    for error in problem.asset_errors:
        err(f"problem.toml: {error}")
    for spec in problem.assets:
        house = _house_models().get(spec.name)
        if house is not None and (spec.repo, spec.revision) != (house.repo, house.revision):
            err(
                f"problem.toml: asset `{spec.name}` is a house model and must be "
                f"{house.repo} at revision {house.revision} (learnai.kit.realmodel)"
            )
    for prerequisite in problem.prerequisites:
        if prerequisite == problem.id:
            err("problem.toml: a problem cannot be its own prerequisite")
        elif prerequisite not in catalog_ids:
            err(f"problem.toml: prerequisite {prerequisite!r} does not exist")

    track_toml = directory.parent / "track.toml"
    try:
        track = tomllib.loads(track_toml.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        err(f"track.toml: cannot be read: {exc}")
    else:
        if track.get("id") != directory.parent.name:
            err(f"track.toml: id must equal the directory name {directory.parent.name!r}")
        for key, kind in (("title", str), ("description", str), ("order", int)):
            if not isinstance(track.get(key), kind) or track.get(key) in ("", None):
                err(f"track.toml: missing or invalid `{key}`")

    for name in ("statement.md", "hints.md", "editorial.md", "tests.py"):
        if not _nonempty(directory / name):
            err(f"{name}: missing or empty")
    if _nonempty(directory / "hints.md"):
        text = (directory / "hints.md").read_text(encoding="utf-8")
        hints = [part for part in re.split(r"(?m)^---[ \t]*$", text)]
        if any(not hint.strip() for hint in hints):
            err("hints.md: contains an empty hint (check the `---` separators)")

    for kind in ("starter", "reference"):
        for name in frameworks:
            if not _nonempty(directory / kind / f"{name}.py"):
                err(f"{kind}/{name}.py: missing or empty")
        folder = directory / kind
        if folder.is_dir():
            for extra in sorted(folder.glob("*.py")):
                if extra.stem not in frameworks:
                    err(f"{kind}/{extra.name}: framework {extra.stem!r} is not in `frameworks`")
    for name in frameworks:
        starter, reference = problem.starter_path(name), problem.reference_path(name)
        if (
            _nonempty(starter)
            and _nonempty(reference)
            and starter.read_text(encoding="utf-8") == reference.read_text(encoding="utf-8")
        ):
            err(f"starter/{name}.py is identical to the reference")

    mutants = problem.mutant_paths()
    if not mutants:
        err("mutants/: at least one mutant is required")
    for mutant in mutants:
        try:
            first = mutant.read_text(encoding="utf-8").splitlines()[0]
        except OSError, IndexError, UnicodeDecodeError:
            first = ""
        if not re.match(r"^# BUG: \S", first):
            err(f"mutants/{mutant.name}: line 1 must be `# BUG: <what is wrong>`")

    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix in (".md", ".py", ".toml"):
            try:
                if PLACEHOLDER in path.read_text(encoding="utf-8"):
                    err(f"{path.relative_to(directory)}: still contains {PLACEHOLDER}")
            except OSError, UnicodeDecodeError:
                err(f"{path.relative_to(directory)}: cannot be read as UTF-8")
    return report, problem


@dataclass(frozen=True)
class _Job:
    kind: str  # reference | starter | mutant
    framework: str
    path: Path

    @property
    def label(self) -> str:
        if self.kind == "mutant":
            return f"mutant {self.path.stem}"
        return f"{self.kind} ({self.framework})"


def unavailable_here(framework: str) -> str | None:
    """Why this platform can never have ``framework``, or None when it can.

    The ``mlx`` extra carries an Apple-silicon marker (pyproject.toml), so everywhere
    else its runs are reported as not validated instead of failing. A framework that
    is only not installed, on a machine that could have it, is still an error.
    """
    apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
    if framework == "mlx" and not apple_silicon:
        return "MLX is installed only on Apple silicon"
    return None


def _checkable(names: tuple[str, ...]) -> tuple[str, ...]:
    """The forbidden names that can be resolved here: not those of an absent framework."""
    return tuple(name for name in names if unavailable_here(name.split(".")[0]) is None)


_IMPORT = re.compile(r"^\s*(?:import|from)\s+([A-Za-z_]\w*)", re.MULTILINE)


def _imports_unavailable(path: Path) -> str | None:
    """Why the file cannot be imported on this platform at all, or None.

    A mutant of a torch problem may borrow another framework's built-in (to show that
    the guard catches it); where that framework cannot exist, the mutant cannot run.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return None
    for module in _IMPORT.findall(source):
        reason = unavailable_here(module)
        if reason is not None:
            return reason
    return None


def _jobs(problem: Problem) -> list[_Job]:
    jobs = []
    for framework in problem.frameworks:
        jobs.append(_Job("reference", framework, problem.reference_path(framework)))
        jobs.append(_Job("starter", framework, problem.starter_path(framework)))
    for mutant in problem.mutant_paths():
        jobs.append(_Job("mutant", problem.oracle_framework, mutant))
    return jobs


def _house_models() -> dict[str, assets.AssetSpec]:
    """The kit's house models. Read from the source file's constants without importing
    the kit package (numpy and every kit module) just to compare two strings."""
    global _HOUSE
    if _HOUSE is None:
        import ast

        _HOUSE = {}
        source = Path(__file__).parent / "kit" / "realmodel.py"
        for node in ast.parse(source.read_text(encoding="utf-8")).body:
            if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
                continue
            if getattr(node.value.func, "id", None) != "AssetSpec":
                continue
            fields = {
                keyword.arg: keyword.value.value
                for keyword in node.value.keywords
                if isinstance(keyword.value, ast.Constant)
            }
            _HOUSE[fields["name"]] = assets.AssetSpec(
                fields["name"], fields["repo"], fields["revision"], (), float(fields["size_mb"])
            )
    return _HOUSE


_HOUSE: dict[str, assets.AssetSpec] | None = None


def _not_run(result: runner.RunResult) -> str:
    """Why cases were skipped, from an incomplete run: one line."""
    reasons = dict.fromkeys(
        str(c.get("message", "")).removeprefix("skipped: ")
        for c in result.cases
        if c.get("status") == "skipped"
    )
    return "; ".join(reason for reason in reasons if reason) or "cases were skipped"


def _skipped_ids(result: runner.RunResult, limit: int = 6) -> str:
    """The ids of the skipped cases (validation runs with --reveal, so real ids)."""
    ids = [str(c.get("id")) for c in result.cases if c.get("status") == "skipped"]
    shown = ", ".join(ids[:limit])
    return shown + (f", and {len(ids) - limit} more" if len(ids) > limit else "")


def _unmet(end: dict[str, Any], problem: Problem) -> str:
    """The hardware the problem requires and this machine lacks, in words."""
    return hardware.describe(str(end.get("missing_hardware") or problem.requires))


def _judge(
    report: Report,
    problem: Problem,
    job: _Job,
    result: runner.RunResult,
    *,
    require_assets: bool = False,
) -> None:
    verdict = result.verdict
    end = result.end or {}
    explanation = f": {end['message']}" if end.get("message") else ""
    if job.kind == "reference":
        _check_stages(report, (result.start or {}).get("cases", []))
    incomplete = verdict in INCOMPLETE_VERDICTS
    passed, skipped = int(end.get("passed", 0)), int(end.get("skipped", 0))
    #: Every case ran and passed; the run is not `accepted` only because the machine
    #: lacks the hardware the problem names in `requires` (nothing was skipped).
    complete_elsewhere = incomplete and skipped == 0 and passed > 0
    if incomplete and verdict == "assets_missing" and require_assets:
        if job.kind == "reference":
            report.errors.append(f"{job.label}: a required model asset is missing{explanation}")
        return
    if incomplete and passed == 0:
        # Nothing ran: there is no behavior to judge, for this file or any other.
        if job.kind == "reference":
            report.skips.append(f"{job.label}: not validated here: {_not_run(result)}")
        return
    if job.kind == "reference":
        if complete_elsewhere:
            cases = (result.start or {}).get("cases", [])
            report.warnings.append(
                f"{job.label}: all {passed} cases ran and passed, but not on the hardware "
                f"the problem requires ({_unmet(end, problem)}), which this machine does "
                "not have"
            )
        elif incomplete:
            cases = (result.start or {}).get("cases", [])
            what = "GPU cases" if verdict == "hardware_missing" else "cases"
            report.warnings.append(
                f"{job.label}: only {passed} of {len(cases)} cases could run on this machine "
                f"(all passed); {skipped} {'was' if skipped == 1 else 'were'} NOT validated: "
                f"{_not_run(result)} ({what} not validated: {_skipped_ids(result)})"
            )
        elif verdict != "accepted":
            failing = [c for c in result.cases if c.get("status") != "pass"]
            # A run that ended before any case (a harness error) has no failing case.
            first = (
                f" (first failing case {failing[0]['id']}: {failing[0].get('message')})"
                if failing
                else ""
            )
            report.errors.append(
                f"{job.label}: must be accepted, got {verdict}{explanation}{first}"
            )
            return
        report.ran = True
        cases = (result.start or {}).get("cases", [])
        public = sum(1 for c in cases if c.get("public"))
        if public == 0:
            report.errors.append(f"tests.py: no public case ({job.framework})")
        if public == len(cases):
            report.errors.append(f"tests.py: no hidden case ({job.framework})")
        seconds = float(end.get("duration_ms", 0)) / 1000
        in_cases = sum(float(c.get("duration_ms", 0)) for c in result.cases) / 1000
        if in_cases > SLOW_CASES_S:
            report.warnings.append(
                f"{job.label}: the cases took {in_cases:.1f} s; a full submit should "
                "finish in a few seconds"
            )
        outcome = f"passed {passed} of {len(cases)} cases" if incomplete else "accepted"
        report.checks.append(
            f"{job.label} {outcome}, {len(cases)} cases, {seconds:.2f} s ({in_cases:.2f} s in cases)"
        )
    elif job.kind == "starter":
        if verdict == "accepted":
            report.errors.append(f"{job.label}: must not be accepted, but it was")
        elif complete_elsewhere:
            report.errors.append(f"{job.label}: passes every case; the tests must reject it")
        elif incomplete:
            report.errors.append(
                f"{job.label}: passes every case that can run on this machine; the cases "
                "that run everywhere must reject the starter"
            )
        elif verdict not in REJECTED:
            report.errors.append(
                f"{job.label}: must run and be rejected by the tests, got {verdict}{explanation}"
            )
        else:
            report.checks.append(f"{job.label} rejected ({verdict})")
    else:
        if verdict == "accepted" or complete_elsewhere:
            # Every case ran: that the machine lacks the hardware changes nothing.
            report.errors.append(
                f"{job.label}: {'was accepted' if verdict == 'accepted' else 'passed every case'}"
                "; no test catches this bug. Add a case that does."
            )
        elif incomplete:
            report.warnings.append(
                f"{job.label}: NOT validated: no case that can run on this machine catches "
                f"it, and {skipped} could not run ({_not_run(result)})"
            )
        elif verdict not in REJECTED:
            report.errors.append(
                f"{job.label}: must run and be rejected by the tests, got {verdict}{explanation}"
            )
        else:
            report.checks.append(f"{job.label} rejected ({verdict})")


def _check_stages(report: Report, cases: list[dict[str, Any]]) -> None:
    """A problem gives every case a stage or none: half-staged cases would be shown
    outside every group, and their stage's progress would be wrong."""
    without = [str(c.get("id")) for c in cases if not c.get("stage")]
    if not without or len(without) == len(cases):
        return
    shown = ", ".join(without[:6]) + (", ..." if len(without) > 6 else "")
    message = (
        f"tests.py: {len(without)} of {len(cases)} cases have no stage ({shown}); "
        "give every @case a stage= or none"
    )
    if message not in report.errors:
        report.errors.append(message)


def is_timing_sensitive(problem: Problem) -> bool:
    """Whether this problem's harness runs must not share the machine with others."""
    return any(judge in TIMING_SENSITIVE_JUDGES for judge in problem.judges)


def check_forbidden(names: tuple[str, ...]) -> dict[str, str]:
    """Resolve and guard every forbidden name in a fresh interpreter.

    Returns ``{name: what is wrong}``. A separate process, because it imports every
    framework the names mention and patches them.
    """
    if not names:
        return {}
    command = [sys.executable, "-P", "-m", "learnai.judge.forbidden", *names]
    try:
        proc = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=FORBIDDEN_CHECK_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return {"forbidden": f"the check did not finish in {FORBIDDEN_CHECK_TIMEOUT_S} s"}
    # Imported late: see FORBIDDEN_NAME.
    from learnai.judge.forbidden import parse_check_output

    result = parse_check_output(proc.stdout)
    if result is None:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:]
        return {"forbidden": "the check failed to run" + (f": {tail[0]}" if tail else "")}
    return {str(name): str(problem) for name, problem in result.items()}


def validate(
    root: Path,
    directories: list[Path],
    jobs: int | None = None,
    *,
    require_assets: bool = False,
) -> list[Report]:
    """Validate the given problem directories against the catalog under ``root``.

    Harness runs execute ``jobs`` at a time, except those of timing-sensitive problems
    (``TIMING_SENSITIVE_JUDGES``), which run one after another once the parallel batch
    has finished, so that nothing else competes with a measurement.

    ``require_assets``: a problem whose model asset is not downloaded is an error
    instead of being skipped (for the machine that is meant to validate everything).
    """
    _, asset_issues = assets.catalog(root)
    conflicts: dict[str, list[str]] = {}
    for problem_id, message in asset_issues:
        conflicts.setdefault(problem_id, []).append(message)
    catalog_ids: dict[str, list[Path]] = {}
    for path in problem_dirs(root):
        try:
            catalog_ids.setdefault(load_problem(path).id, []).append(path)
        except Exception:  # noqa: BLE001 - reported when that problem is validated
            catalog_ids.setdefault(path.name, []).append(path)
    for path in directories:  # problems outside the catalog (tests) still know themselves
        if path not in catalog_ids.get(path.name, []):
            catalog_ids.setdefault(path.name, []).append(path)

    reports: list[Report] = []
    parallel: list[tuple[Report, Problem, _Job]] = []
    serial: list[tuple[Report, Problem, _Job]] = []
    guarded: list[tuple[Report, Problem]] = []
    for directory in directories:
        report, problem = check_metadata(directory, catalog_ids)
        reports.append(report)
        if problem is not None and not problem.asset_errors:
            # Two problems naming one asset must agree on what it is.
            report.errors.extend(
                f"problem.toml: {message}" for message in conflicts.get(problem.id, [])
            )
        if problem is None or not report.ok:
            continue  # behavioral checks on a malformed problem only add noise
        if problem.forbidden:
            guarded.append((report, problem))
        queue = serial if is_timing_sensitive(problem) else parallel
        for job in _jobs(problem):
            reason = unavailable_here(job.framework)
            if reason is not None:
                if job.kind == "reference":
                    report.skips.append(f"{job.label}: not validated here: {reason}")
                continue
            if job.kind == "mutant" and (reason := _imports_unavailable(job.path)):
                report.warnings.append(f"{job.label}: NOT validated: {reason}")
                continue
            queue.append((report, problem, job))

    def execute(item: tuple[Report, Problem, _Job]) -> runner.RunResult:
        _, problem, job = item
        # Authors see hidden cases by their real ids ("first failing case ...").
        command = runner.harness_command(
            problem.path, job.path, job.framework, "submit", reveal=True
        )
        return runner.run(command)

    workers = max(1, jobs or min(4, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        unresolved = pool.map(lambda item: check_forbidden(_checkable(item[1].forbidden)), guarded)
        results = pool.map(execute, parallel)
        for (report, problem), problems in zip(guarded, unresolved, strict=True):
            for name, what in problems.items():
                report.errors.append(f"problem.toml: forbidden name `{name}`: {what}")
            if not problems:
                checked = len(_checkable(problem.forbidden))
                left_out = len(problem.forbidden) - checked
                report.checks.append(
                    f"forbidden: {checked} names resolve and can be guarded"
                    + (f" ({left_out} of a framework this platform cannot have)" if left_out else "")
                )
        for item, result in zip(parallel, results, strict=True):
            _judge(item[0], item[1], item[2], result, require_assets=require_assets)
    # The pool is idle now: timing-sensitive runs get the machine to themselves.
    for item in serial:
        _judge(item[0], item[1], item[2], execute(item), require_assets=require_assets)
    return reports
