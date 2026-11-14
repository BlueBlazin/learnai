"""Shared helpers for the judge's own tests."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from whetstone import protocol
from whetstone.judge import frameworks
from whetstone.judge.context import Context, make_context
from whetstone.judge.registry import Case
from whetstone.problem import load_problem

# Harness runs with JAX keep compiled kernels on disk under the content root
# (frameworks.use_jax_compilation_cache). The judge's own tests use fixture catalogs
# inside the repository as roots; their cache goes to a temporary directory instead.
if "WHETSTONE_JAX_CACHE" not in os.environ:
    import atexit
    import shutil
    import tempfile

    _jax_cache = tempfile.mkdtemp(prefix="whetstone-test-jax-cache-")
    os.environ["WHETSTONE_JAX_CACHE"] = _jax_cache
    atexit.register(shutil.rmtree, _jax_cache, ignore_errors=True)

FIXTURES = Path(__file__).parent / "fixtures"
CATALOG = FIXTURES / "catalog"
SOLUTIONS = FIXTURES / "solutions"
REPO_ROOT = Path(__file__).resolve().parents[2]


def fixture_problem(name: str) -> Path:
    matches = list((CATALOG / "problems").glob(f"*/{name}"))
    assert len(matches) == 1, name
    return matches[0]


@dataclass
class HarnessRun:
    returncode: int
    events: list[dict[str, Any]]
    logs: list[str]
    stdout: str
    stderr: str

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["event"] == kind]

    @property
    def start(self) -> dict[str, Any]:
        (event,) = self.of("start")
        return event

    @property
    def end(self) -> dict[str, Any]:
        (event,) = self.of("end")
        return event

    def case(self, case_id: str) -> dict[str, Any]:
        (event,) = [e for e in self.of("case_end") if e["id"] == case_id]
        return event

    @property
    def statuses(self) -> dict[str, str]:
        return {e["id"]: e["status"] for e in self.of("case_end")}


def run_harness(
    problem: Path,
    solution: Path,
    framework: str = "python",
    mode: str = "submit",
    *,
    case: str | None = None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    extra: list[str] | None = None,
) -> HarnessRun:
    command = [
        sys.executable, "-m", "whetstone.harness",
        "--problem", str(problem), "--solution", str(solution),
        "--framework", framework, "--mode", mode,
    ]  # fmt: skip
    if case is not None:
        command += ["--case", case]
    command += extra or []
    proc = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=cwd,
        env={**os.environ, "WHETSTONE_HARD_GRACE_S": "0.5", **(env or {})},
        timeout=120,
    )
    events, logs = [], []
    for line in proc.stdout.splitlines():
        event = protocol.decode(line)
        if event is None:
            logs.append(line)
        else:
            events.append(event)
    return HarnessRun(proc.returncode, events, logs, proc.stdout, proc.stderr)


@pytest.fixture
def sum_list() -> Path:
    return fixture_problem("sum-list")


@pytest.fixture
def ctx() -> Context:
    """An active Context for exercising helpers that report metrics or draw seeds."""
    problem = load_problem(fixture_problem("sum-list"))
    case = Case(id="unit", name="unit", public=True, fn=lambda ctx: None)
    context = make_context(problem, solution=None, framework="python", case=case)
    with context.active():
        yield context
    context.close()


@pytest.fixture(params=["torch", "jax", "mlx", "python"])
def fw(request: pytest.FixtureRequest) -> frameworks.Framework:
    try:
        return frameworks.get(request.param)
    except frameworks.FrameworkUnavailable as exc:
        pytest.skip(str(exc))
