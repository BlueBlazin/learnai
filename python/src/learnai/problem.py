"""Problem and track metadata (docs/ARCHITECTURE.md section 3.3) and catalog discovery."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from learnai.assets import AssetSpec, parse_assets

FRAMEWORKS = ("torch", "jax", "mlx", "python")
TENSOR_FRAMEWORKS = ("torch", "jax", "mlx")
DIFFICULTIES = ("easy", "medium", "hard")
#: Levels within a difficulty: easy 1 is the gentlest problem, hard 3 the most demanding.
LEVELS = (1, 2, 3)
JUDGES = ("numeric", "property", "statistical", "cost", "speed", "simclock", "service")
REQUIRES = ("cpu", "mlx-gpu", "cuda")


class ProblemError(Exception):
    """A problem directory is missing, unreadable, or malformed."""


@dataclass(frozen=True)
class Problem:
    path: Path
    id: str
    title: str
    track: str
    order: int
    difficulty: str
    level: int
    frameworks: tuple[str, ...]
    judges: tuple[str, ...]
    tags: tuple[str, ...]
    prerequisites: tuple[str, ...]
    requires: str
    entry: tuple[str, ...]
    time_limit_s: float
    raw: dict[str, Any]
    #: Optional. Dotted names the solution must not call (see judge/forbidden.py).
    forbidden: tuple[str, ...] = ()
    #: Optional. Identifiers that must not appear in the solution's source.
    forbidden_source: tuple[str, ...] = ()
    #: Optional. Model assets the problem needs (``[[assets]]``, see learnai/assets.py).
    assets: tuple[AssetSpec, ...] = ()
    #: What is wrong with the ``assets`` declaration, for ``learnai validate``.
    asset_errors: tuple[str, ...] = ()

    @property
    def oracle_framework(self) -> str:
        """The framework the oracle reference and the mutants are written in.

        torch when the problem supports it; otherwise the problem's first framework:
        ``python`` for framework-free problems, ``mlx`` for MLX-only problems.
        """
        if "torch" in self.frameworks or not self.frameworks:
            return "torch"
        return self.frameworks[0]

    def asset(self, name: str) -> AssetSpec | None:
        return next((spec for spec in self.assets if spec.name == name), None)

    @property
    def fixtures_path(self) -> Path:
        return self.path / "fixtures"

    @property
    def tests_path(self) -> Path:
        return self.path / "tests.py"

    def starter_path(self, framework: str) -> Path:
        return self.path / "starter" / f"{framework}.py"

    def reference_path(self, framework: str) -> Path:
        return self.path / "reference" / f"{framework}.py"

    def mutant_paths(self) -> list[Path]:
        return sorted((self.path / "mutants").glob("*.py"))


@dataclass(frozen=True)
class Track:
    path: Path
    id: str
    title: str
    description: str
    order: int


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    return ()


def load_problem(path: str | os.PathLike[str]) -> Problem:
    """Load ``problem.toml`` leniently; ``learnai validate`` is what enforces completeness."""
    directory = Path(path).resolve()
    toml_path = directory / "problem.toml"
    if not toml_path.is_file():
        raise ProblemError(f"no problem.toml in {directory}")
    try:
        raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ProblemError(f"cannot read {toml_path}: {exc}") from exc

    def number(key: str, default: float) -> float:
        value = raw.get(key, default)
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else default

    assets, asset_errors = parse_assets(raw.get("assets"))
    return Problem(
        path=directory,
        id=str(raw.get("id", directory.name)),
        title=str(raw.get("title", directory.name)),
        track=str(raw.get("track", directory.parent.name)),
        order=int(number("order", 0)),
        difficulty=str(raw.get("difficulty", "")),
        level=int(number("level", 0)),
        frameworks=_strings(raw.get("frameworks")),
        judges=_strings(raw.get("judges")),
        tags=_strings(raw.get("tags")),
        prerequisites=_strings(raw.get("prerequisites")),
        requires=str(raw.get("requires", "cpu")),
        entry=_strings(raw.get("entry")),
        time_limit_s=float(number("time_limit_s", 10.0)),
        raw=raw,
        forbidden=_strings(raw.get("forbidden")),
        forbidden_source=_strings(raw.get("forbidden_source")),
        assets=tuple(assets),
        asset_errors=tuple(asset_errors),
    )


def load_track(path: str | os.PathLike[str]) -> Track:
    directory = Path(path).resolve()
    toml_path = directory / "track.toml"
    if not toml_path.is_file():
        raise ProblemError(f"no track.toml in {directory}")
    try:
        raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ProblemError(f"cannot read {toml_path}: {exc}") from exc
    order = raw.get("order", 0)
    return Track(
        path=directory,
        id=str(raw.get("id", directory.name)),
        title=str(raw.get("title", directory.name)),
        description=str(raw.get("description", "")),
        order=order if isinstance(order, int) else 0,
    )


def find_root(start: str | os.PathLike[str] | None = None) -> Path:
    """Locate the content root: the directory holding ``python/`` and ``problems/``.

    Order: ``LEARNAI_ROOT``, then the nearest ancestor of ``start`` (default: the
    working directory) that looks like a root, then the checkout this package lives in.
    """
    env = os.environ.get("LEARNAI_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "problems").is_dir() and (candidate / "python").is_dir():
            return candidate
    # python/src/learnai/problem.py -> the repository root is three levels up.
    return Path(__file__).resolve().parents[3]


def problem_dirs(root: Path) -> list[Path]:
    """Every ``problems/<track>/<id>/`` directory that has a problem.toml, sorted."""
    return sorted(p.parent for p in (root / "problems").glob("*/*/problem.toml"))


def discover(root: Path) -> list[Problem]:
    """Load every problem, ordered by track then by position within the track."""
    problems = [load_problem(path) for path in problem_dirs(root)]
    track_order: dict[str, int] = {}
    for track_dir in sorted((root / "problems").glob("*/track.toml")):
        try:
            track = load_track(track_dir.parent)
        except ProblemError:
            continue
        track_order[track_dir.parent.name] = track.order
    return sorted(
        problems, key=lambda p: (track_order.get(p.path.parent.name, 0), p.track, p.order, p.id)
    )


def find_problem(root: Path, problem_id: str) -> Problem:
    """Resolve a problem id (or ``track/id``, or a directory path) to a Problem."""
    as_path = Path(problem_id)
    if (as_path / "problem.toml").is_file():
        return load_problem(as_path)
    wanted = problem_id.split("/")[-1]
    matches = [p for p in problem_dirs(root) if p.name == wanted]
    if "/" in problem_id:
        matches = [p for p in matches if p.parent.name == problem_id.split("/")[0]]
    if not matches:
        raise ProblemError(f"no problem with id {problem_id!r} under {root / 'problems'}")
    if len(matches) > 1:
        where = ", ".join(str(m.relative_to(root)) for m in matches)
        raise ProblemError(f"problem id {problem_id!r} is not unique: {where}")
    return load_problem(matches[0])
