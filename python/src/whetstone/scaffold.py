"""``ws new <track>/<id>``: scaffold a problem directory.

The scaffold is a complete, runnable toy problem (scale a tensor by a factor) in the
house layout. Every place the author must rewrite is marked ``TODO(author)``, and
``ws validate`` refuses a problem that still contains the marker.
"""

from __future__ import annotations

from pathlib import Path

from whetstone.problem import DIFFICULTIES, FRAMEWORKS, TENSOR_FRAMEWORKS
from whetstone.validate import ID_PATTERN, PLACEHOLDER


class ScaffoldError(Exception):
    pass


_IMPORTS = {
    "torch": ("import torch", "torch.Tensor"),
    "jax": ("import jax\nimport jax.numpy as jnp", "jax.Array"),
    "mlx": ("import mlx.core as mx", "mx.array"),
    "python": ("", "list[float]"),
}

_STARTER = '''{imports}def scale(x: {tensor}, factor: float = 2.0) -> {tensor}:
    """{todo}: one-line summary of what to compute.

    x: {todo}: shape and dtype.
    Returns {todo}: shape and dtype.
    """
    raise NotImplementedError
'''

_REFERENCE = """{imports}def scale(x: {tensor}, factor: float = 2.0) -> {tensor}:
    # {todo}: the reference solution. Comment the non-obvious steps.
    return {body}
"""

_MUTANT = """# BUG: {todo}: describe the bug in one line (this one ignores `factor`).
{imports}def scale(x: {tensor}, factor: float = 2.0) -> {tensor}:
    return {body}
"""

_TESTS_TENSOR = '''"""Tests for {id}."""

import numpy as np

from whetstone.judge import case

# {todo}: replace with real cases. At least one public and one hidden case; every
# mutant must fail at least one of them.


@case("a small worked example", public=True)
def worked_example(ctx):
    ctx.check("scale", np.array([1.0, 2.0, 3.0], dtype=np.float32), factor=3.0)


@case("a random batch")
def random_batch(ctx):
    ctx.check("scale", ctx.randn(4, 8), factor=0.5)
'''

_TESTS_PYTHON = '''"""Tests for {id}."""

from whetstone.judge import case, check

# {todo}: replace with real cases. At least one public and one hidden case; every
# mutant must fail at least one of them.


@case("a small worked example", public=True)
def worked_example(ctx):
    xs = [1.0, 2.0, 3.0]
    check.equal(ctx.call("scale", xs, factor=3.0), ctx.expected("scale", xs, factor=3.0))


@case("a longer list")
def longer_list(ctx):
    xs = [float(v) for v in ctx.rng.integers(-9, 9, size=20)]
    check.equal(ctx.call("scale", xs, factor=0.5), ctx.expected("scale", xs, factor=0.5))
'''

_STATEMENT = """{todo}: one or two sentences stating the task and what makes it interesting.

## Specification

```python
def scale(x, factor=2.0): ...
```

{todo}: the exact definition, with math where it helps: $y_i = c \\cdot x_i$.

- `x`: {todo}.
- Returns {todo}.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(n,)` | float32 |
| result | `(n,)` | float32 |

## Example

```python
scale([1.0, 2.0, 3.0], factor=3.0)   # [3.0, 6.0, 9.0]
```

## Constraints

- {todo}: input ranges, sizes, tolerances, what is not allowed.

## Why this matters

{todo}: one short paragraph connecting the problem to real inference systems.
"""

_HINTS = """{todo}: first hint. A nudge toward the key observation, no code.

---

{todo}: second hint. The approach, stated concretely.

---

{todo}: third hint. Close to the solution: the formula or the key line.
"""

_EDITORIAL = """## The idea

{todo}: explain the approach of the reference and why it is correct.

## Reference solution (PyTorch)

```python
{todo}
```

## Common mistakes

- {todo}: each mutant should correspond to one of these.

## Complexity

{todo}
"""


def create(
    root: Path,
    target: str,
    *,
    title: str | None = None,
    frameworks: list[str] | None = None,
    difficulty: str = "easy",
) -> Path:
    """Create ``problems/<track>/<id>/`` (and track.toml if the track is new)."""
    parts = target.strip("/").split("/")
    if len(parts) != 2 or not all(ID_PATTERN.match(part) for part in parts):
        raise ScaffoldError("expected <track>/<id>, both lowercase words joined by hyphens")
    track, problem_id = parts
    frameworks = frameworks or list(TENSOR_FRAMEWORKS)
    for name in frameworks:
        if name not in FRAMEWORKS:
            raise ScaffoldError(f"unknown framework {name!r}")
    if "python" in frameworks and len(frameworks) > 1:
        raise ScaffoldError('"python" cannot be combined with tensor frameworks')
    if "python" not in frameworks and "torch" not in frameworks:
        raise ScaffoldError('tensor problems must include "torch" (the oracle)')
    if difficulty not in DIFFICULTIES:
        raise ScaffoldError(f"difficulty must be one of {', '.join(DIFFICULTIES)}")
    existing = [p for p in (root / "problems").glob(f"*/{problem_id}") if p.is_dir()]
    if existing:
        raise ScaffoldError(f"problem id {problem_id!r} already exists at {existing[0]}")

    track_dir = root / "problems" / track
    directory = track_dir / problem_id
    siblings = [p for p in track_dir.glob("*/problem.toml")] if track_dir.is_dir() else []
    for folder in ("starter", "reference", "mutants"):
        (directory / folder).mkdir(parents=True, exist_ok=True)

    def write(relative: str, text: str) -> None:
        (directory / relative).write_text(text, encoding="utf-8")

    if not (track_dir / "track.toml").exists():
        n_tracks = len(list((root / "problems").glob("*/track.toml")))
        (track_dir / "track.toml").write_text(
            f'id = "{track}"\n'
            f'title = "{track.replace("-", " ").title()}"\n'
            f'description = "{PLACEHOLDER}: one or two sentences about this track."\n'
            f"order = {10 * (n_tracks + 1)}\n",
            encoding="utf-8",
        )

    title = title or problem_id.replace("-", " ").title()
    listed = ", ".join(f'"{name}"' for name in frameworks)
    write(
        "problem.toml",
        f'id = "{problem_id}"\n'
        f'title = "{title}"\n'
        f'track = "{track}"\n'
        f"order = {10 * (len(siblings) + 1)}\n"
        f'difficulty = "{difficulty}"\n'
        f"frameworks = [{listed}]\n"
        'judges = ["numeric"]\n'
        f'tags = ["{PLACEHOLDER}"]\n'
        "prerequisites = []\n"
        'requires = "cpu"\n'
        'entry = ["scale"]\n'
        "time_limit_s = 10\n",
    )
    for name in frameworks:
        imports, tensor = _IMPORTS[name]
        imports = f"{imports}\n\n\n" if imports else ""
        good = "[v * factor for v in x]" if name == "python" else "x * factor"
        write(
            f"starter/{name}.py",
            _STARTER.format(imports=imports, tensor=tensor, todo=PLACEHOLDER),
        )
        write(
            f"reference/{name}.py",
            _REFERENCE.format(imports=imports, tensor=tensor, todo=PLACEHOLDER, body=good),
        )
    oracle = "python" if "python" in frameworks else "torch"
    imports, tensor = _IMPORTS[oracle]
    write(
        "mutants/ignores_factor.py",
        _MUTANT.format(
            imports=f"{imports}\n\n\n" if imports else "\n\n",
            tensor=tensor,
            todo=PLACEHOLDER,
            body="[v * 2.0 for v in x]" if oracle == "python" else "x * 2.0",
        ),
    )
    tests = _TESTS_PYTHON if oracle == "python" else _TESTS_TENSOR
    write("tests.py", tests.format(id=problem_id, todo=PLACEHOLDER))
    write("statement.md", _STATEMENT.format(todo=PLACEHOLDER))
    write("hints.md", _HINTS.format(todo=PLACEHOLDER))
    write("editorial.md", _EDITORIAL.format(todo=PLACEHOLDER))
    return directory
