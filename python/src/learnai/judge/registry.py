"""The ``@case`` decorator and the loader that collects cases from a tests.py file."""

from __future__ import annotations

import importlib.util
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from learnai.hardware import REQUIREMENTS

_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")

#: A stage label is a heading in the app and in the terminal: keep it short.
MAX_STAGE_LENGTH = 60

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
    #: Hardware the case needs: ``cpu`` (the default: runs everywhere), ``mlx-gpu``, ``cuda``.
    requires: str = "cpu"
    #: Names of the problem's assets the case needs; None means all of them.
    assets: tuple[str, ...] | None = None
    #: The stage of a multi-part problem the case belongs to; None for no stage.
    stage: str | None = None

    def applies_to(self, framework: str) -> bool:
        return self.frameworks is None or framework in self.frameworks


def case(
    name: str,
    *,
    public: bool = False,
    id: str | None = None,
    time_limit_s: float | None = None,
    frameworks: Sequence[str] | None = None,
    requires: str = "cpu",
    assets: Sequence[str] | None = None,
    stage: str | None = None,
) -> Callable[[Callable[[Any], None]], Callable[[Any], None]]:
    """Register a test case.

    name          what the learner sees; write it as a short statement of what is
                  checked ("handles logits near 1e4").
    public        True: runs on Run and Submit. False (default, a "hidden" case):
                  runs on Submit only. Either way the learner sees the name, the
                  failure message, and the details.
    id            stable identifier; defaults to the function name. It seeds the
                  case's random generator.
    time_limit_s  overrides the problem's per-case limit.
    frameworks    restrict the case to these solution frameworks (rarely needed).
    requires      hardware the case needs: ``"cpu"`` (default: the case runs on every
                  machine), or the problem's own ``requires`` (``"mlx-gpu"`` or
                  ``"cuda"``). Where the hardware is absent the case is reported as
                  ``skipped`` with the reason, and the run cannot be ``accepted``.
    assets        names of the problem's ``[[assets]]`` this case needs. Default: all
                  of them. ``assets=[]`` marks a case that needs no download (a
                  synthetic input). A case whose asset is not on disk is ``skipped``.
    stage         the stage of a multi-part problem (a capstone) this case grades, for
                  example ``"2: paged KV cache"``. Cases with the same label form a
                  stage; stages are ordered by the first case that names them. A
                  problem gives every case a stage or none (``learnai validate``
                  enforces it).

    Cases run in the order they are defined. The function takes one argument, the
    ``Context``; it passes by returning and fails by raising ``CheckFailure`` (which is
    what every ``check`` helper does).
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("@case needs a non-empty name")
    if requires not in REQUIREMENTS:
        raise ValueError(f"@case requires= must be one of {', '.join(REQUIREMENTS)}")
    if assets is not None and (
        isinstance(assets, str) or not all(isinstance(item, str) for item in assets)
    ):
        raise ValueError("@case assets= must be a list of asset names")
    if stage is not None:
        if not isinstance(stage, str) or not stage.strip() or "\n" in stage:
            raise ValueError("@case stage= must be a non-empty one-line label")
        if len(stage.strip()) > MAX_STAGE_LENGTH:
            raise ValueError(f"@case stage= is longer than {MAX_STAGE_LENGTH} characters")
        stage = stage.strip()

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
                    requires=requires,
                    assets=tuple(assets) if assets is not None else None,
                    stage=stage,
                )
            )
        return fn

    return register


def stages_of(cases: Sequence[Any]) -> list[str]:
    """The stage labels of ``cases`` in order of first appearance.

    ``cases`` are ``Case`` objects or the case objects of a ``start`` event.
    """
    labels: dict[str, None] = {}
    for item in cases:
        label = item.get("stage") if isinstance(item, dict) else item.stage
        if label is not None:
            labels.setdefault(label)
    return list(labels)


def load_cases(tests_path: Path) -> list[Case]:
    """Import ``tests.py`` and return the cases it registers, in definition order."""
    global _collecting
    if not tests_path.is_file():
        raise FileNotFoundError(f"no tests.py at {tests_path}")
    name = f"learnai_tests_{abs(hash(str(tests_path)))}"
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
