"""The ``@case`` decorator and the loader that collects cases from a tests.py file."""

from __future__ import annotations

import importlib.util
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")

# The list that @case appends to while a tests.py file is being loaded.
_collecting: list[Case] | None = None


@dataclass(frozen=True)
class Case:
    id: str
    name: str
    public: bool
    fn: Callable[[Any], None]
    time_limit_s: float | None = None
    frameworks: tuple[str, ...] | None = None

    def applies_to(self, framework: str) -> bool:
        return self.frameworks is None or framework in self.frameworks


def case(
    name: str,
    *,
    public: bool = False,
    id: str | None = None,
    time_limit_s: float | None = None,
    frameworks: Sequence[str] | None = None,
) -> Callable[[Callable[[Any], None]], Callable[[Any], None]]:
    """Register a test case.

    name          what the learner sees for a public case; write it as a short
                  statement of what is checked ("handles logits near 1e4").
    public        True: runs on Run and Submit, name and details are shown.
                  False (default): runs on Submit only; the learner sees
                  "Hidden case N" and the failure message, never the details.
    id            stable identifier; defaults to the function name.
    time_limit_s  overrides the problem's per-case limit.
    frameworks    restrict the case to these solution frameworks (rarely needed).

    Cases run in the order they are defined. The function takes one argument, the
    ``Context``; it passes by returning and fails by raising ``CheckFailure`` (which is
    what every ``check`` helper does).
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("@case needs a non-empty name")

    def register(fn: Callable[[Any], None]) -> Callable[[Any], None]:
        case_id = id if id is not None else fn.__name__
        if not _ID_PATTERN.match(case_id):
            raise ValueError(f"case id {case_id!r} may only contain letters, digits, _ . -")
        if _collecting is not None:
            if any(existing.id == case_id for existing in _collecting):
                raise ValueError(f"duplicate case id {case_id!r}")
            _collecting.append(
                Case(
                    id=case_id,
                    name=name.strip(),
                    public=bool(public),
                    fn=fn,
                    time_limit_s=time_limit_s,
                    frameworks=tuple(frameworks) if frameworks is not None else None,
                )
            )
        return fn

    return register


def load_cases(tests_path: Path) -> list[Case]:
    """Import ``tests.py`` and return the cases it registers, in definition order."""
    global _collecting
    if not tests_path.is_file():
        raise FileNotFoundError(f"no tests.py at {tests_path}")
    name = f"whetstone_tests_{abs(hash(str(tests_path)))}"
    spec = importlib.util.spec_from_file_location(name, tests_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {tests_path}")
    module = importlib.util.module_from_spec(spec)
    previous, _collecting = _collecting, []
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        cases = _collecting
    finally:
        _collecting = previous
    if not cases:
        raise ValueError(f"{tests_path} defines no @case functions")
    return cases
