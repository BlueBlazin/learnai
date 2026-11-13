"""``ws validate``: the quality gate for problems (docs/ARCHITECTURE.md section 3.7).

For every supported framework of a problem: the reference is accepted and the starter
is not. Every mutant is rejected. Metadata is complete and consistent. This module
returns findings; the CLI prints them.
"""

from __future__ import annotations

import os
import re
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from whetstone import runner
from whetstone.problem import (
    DIFFICULTIES,
    FRAMEWORKS,
    JUDGES,
    REQUIRES,
    TENSOR_FRAMEWORKS,
    Problem,
    load_problem,
    problem_dirs,
)

ID_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
PLACEHOLDER = "TODO(author)"
#: A full submit should finish in a few seconds. Process startup and framework imports
#: are a fixed cost, so the warning looks at what the author controls: the cases.
SLOW_CASES_S = 3.0
REJECTED = ("wrong_answer", "runtime_error", "time_limit")

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

    @property
    def ok(self) -> bool:
        return not self.errors


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
    for key in ("frameworks", "judges", "tags", "prerequisites", "entry"):
        value = raw.get(key)
        if isinstance(value, list) and not all(isinstance(item, str) for item in value):
            err(f"problem.toml: `{key}` must be a list of strings")
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
    if any(f in TENSOR_FRAMEWORKS for f in frameworks) and "torch" not in frameworks:
        err('problem.toml: tensor problems must support "torch" (the oracle is written in it)')

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


def _jobs(problem: Problem) -> list[_Job]:
    jobs = []
    for framework in problem.frameworks:
        jobs.append(_Job("reference", framework, problem.reference_path(framework)))
        jobs.append(_Job("starter", framework, problem.starter_path(framework)))
    for mutant in problem.mutant_paths():
        jobs.append(_Job("mutant", problem.oracle_framework, mutant))
    return jobs


def _judge(report: Report, problem: Problem, job: _Job, result: runner.RunResult) -> None:
    verdict = result.verdict
    end = result.end or {}
    explanation = f": {end['message']}" if end.get("message") else ""
    if job.kind == "reference":
        if verdict != "accepted":
            failing = [c for c in result.cases if c.get("status") != "pass"]
            first = f" (first failing case {failing[0]['id']}: {failing[0].get('message')})"
            report.errors.append(
                f"{job.label}: must be accepted, got {verdict}{explanation}"
                f"{first if failing else ''}"
            )
            return
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
        report.checks.append(
            f"{job.label} accepted, {len(cases)} cases, {seconds:.2f} s ({in_cases:.2f} s in cases)"
        )
    elif job.kind == "starter":
        if verdict == "accepted":
            report.errors.append(f"{job.label}: must not be accepted, but it was")
        elif verdict not in REJECTED:
            report.errors.append(
                f"{job.label}: must run and be rejected by the tests, got {verdict}{explanation}"
            )
        else:
            report.checks.append(f"{job.label} rejected ({verdict})")
    else:
        if verdict == "accepted":
            report.errors.append(
                f"{job.label}: was accepted; no test catches this bug. Add a case that does."
            )
        elif verdict not in REJECTED:
            report.errors.append(
                f"{job.label}: must run and be rejected by the tests, got {verdict}{explanation}"
            )
        else:
            report.checks.append(f"{job.label} rejected ({verdict})")


def validate(root: Path, directories: list[Path], jobs: int | None = None) -> list[Report]:
    """Validate the given problem directories against the catalog under ``root``."""
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
    work: list[tuple[Report, Problem, _Job]] = []
    for directory in directories:
        report, problem = check_metadata(directory, catalog_ids)
        reports.append(report)
        if problem is None or not report.ok:
            continue  # behavioral checks on a malformed problem only add noise
        work.extend((report, problem, job) for job in _jobs(problem))

    def execute(item: tuple[Report, Problem, _Job]) -> runner.RunResult:
        _, problem, job = item
        return runner.run(runner.harness_command(problem.path, job.path, job.framework, "submit"))

    workers = jobs or min(4, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for item, result in zip(work, pool.map(execute, work), strict=True):
            _judge(item[0], item[1], item[2], result)
    return reports
