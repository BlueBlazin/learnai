"""The object every test case receives: ``def my_case(ctx): ...``."""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import importlib.util
import inspect
import json
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from whetstone import assets, protocol
from whetstone.judge.frameworks import Framework, get
from whetstone.problem import Problem

_current: contextvars.ContextVar[Context | None] = contextvars.ContextVar(
    "whetstone_context", default=None
)


def current() -> Context:
    """The context of the case that is running (used by ``check`` helpers)."""
    ctx = _current.get()
    if ctx is None:
        raise RuntimeError("no test case is running; this helper needs an active Context")
    return ctx


def current_or_none() -> Context | None:
    return _current.get()


class ModuleStore:
    """Per-run state shared by all cases: the solution and lazily loaded references."""

    def __init__(
        self,
        problem: Problem,
        solution: ModuleType,
        fw: Framework,
        oracle_fw: Framework,
        models_dir: Path | None = None,
    ) -> None:
        self.problem = problem
        self.solution = solution
        self.fw = fw
        self.oracle_fw = oracle_fw
        #: Where model assets live (``.whetstone/models`` under the content root).
        self.models_dir = models_dir or assets.models_dir(assets.root_of(problem.path))
        self._references: dict[str, ModuleType] = {}

    def reference(self, framework: str) -> ModuleType:
        if framework not in self._references:
            path = self.problem.reference_path(framework)
            name = f"whetstone_reference_{framework}"
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None or not path.is_file():
                raise FileNotFoundError(f"no reference implementation at {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            self._references[framework] = module
        return self._references[framework]


class Context:
    """Everything a test needs. See docs/PROBLEM_AUTHORING.md for the guided tour.

    Attributes:
        problem     the Problem (metadata).
        framework   name of the framework the solution is written in.
        fw          adapter for the solution's framework.
        oracle_fw   adapter for the oracle's framework (torch; python for a
                    framework-free problem; mlx for an MLX-only problem).
        solution    the learner's module.
        oracle      the oracle reference module (reference/<oracle framework>.py).
        reference   the reference module in the solution's own framework (speed baseline).
        seed, rng   per-case seed and numpy Generator; identical for every framework.
    """

    def __init__(self, store: ModuleStore, case: Any) -> None:
        self._store = store
        self.case = case
        self.problem = store.problem
        self.fw = store.fw
        self.oracle_fw = store.oracle_fw
        self.framework = store.fw.name
        self.solution = store.solution
        digest = hashlib.sha256(f"{self.problem.id}:{case.id}".encode()).digest()
        self.seed = int.from_bytes(digest[:8], "big")
        self.rng = np.random.default_rng(self.seed)
        self.metrics: list[dict[str, Any]] = []
        self._cleanups: list[Callable[[], None]] = []

    # -- modules ---------------------------------------------------------------

    @property
    def oracle(self) -> ModuleType:
        return self._store.reference(self.oracle_fw.name)

    @property
    def reference(self) -> ModuleType:
        return self._store.reference(self.fw.name)

    # -- calling ---------------------------------------------------------------

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Call the solution's ``name`` with numpy inputs; get numpy outputs back.

        Arrays anywhere in the arguments become the solution's tensors; tensors
        anywhere in the result become numpy arrays. Everything else passes through.

        Lists, tuples, and dicts are rebuilt on the way in (that is how the arrays
        inside them get converted), so the solution never receives the very object the
        test holds. To test in-place mutation of an argument, call the solution
        directly: ``ctx.solution.push(heap, 3)`` and then look at ``heap``.
        """
        fn = getattr(self.solution, name)
        out = fn(*self.fw.to_framework(args), **self.fw.to_framework(kwargs))
        return self.fw.to_numpy(out, what=f"`{name}`")

    def expected(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Call the oracle reference's ``name`` the same way; numpy in, numpy out."""
        return self._reference_call(self.oracle, self.oracle_fw, name, args, kwargs)

    def baseline(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Call the same-framework reference's ``name``; numpy in, numpy out."""
        return self._reference_call(self.reference, self.fw, name, args, kwargs)

    @staticmethod
    def _reference_call(
        module: ModuleType, fw: Framework, name: str, args: tuple, kwargs: dict
    ) -> Any:
        fn = getattr(module, name)
        out = fn(*fw.to_framework(args, fresh_kit=True), **fw.to_framework(kwargs, fresh_kit=True))
        return fw.to_numpy(out, what=f"the reference `{name}`")

    def check(
        self,
        name: str,
        *args: Any,
        rtol: float | None = None,
        atol: float | None = None,
        dtype: bool = True,
        **kwargs: Any,
    ) -> Any:
        """The one-line numeric test: call solution and oracle, compare, return the result.

        Float outputs are compared with ``check.allclose`` (float32 defaults, or the
        ``rtol``/``atol`` given), integer and bool outputs exactly, tuples and dicts
        element by element. ``dtype=True`` also requires matching dtype kinds and
        float widths. On failure the detail names the call and its inputs.
        """
        from whetstone.judge import check

        got = self.call(name, *args, **kwargs)
        want = self.expected(name, *args, **kwargs)
        check.match(
            got,
            want,
            rtol=rtol,
            atol=atol,
            dtype=dtype,
            label=name,
            context=describe_call(getattr(self.oracle, name), name, args, kwargs),
        )
        return got

    def check_speed(
        self,
        name: str,
        *args: Any,
        max_ratio: float,
        rounds: int = 7,
        warmup: int = 2,
        tokens: int | None = None,
        **kwargs: Any,
    ) -> float:
        """Speed judge: the solution must be within ``max_ratio`` of the reference.

        Times the solution's ``name`` against the same-framework reference's, in this
        process, interleaved, after ``warmup`` untimed calls each, taking the median of
        ``rounds`` measurements and forcing lazy frameworks to finish before the clock
        stops (``fw.sync``: ``mx.eval`` plus ``mx.synchronize`` on MLX). Reports the
        ``slowdown`` metric and returns the ratio.

        ``tokens``: how many tokens one call produces or processes. When given, the
        solution's throughput is also reported as the ``tokens_per_s`` metric
        (informational: it has no limit, because an absolute rate means nothing across
        machines).
        """
        fw_args = self.fw.to_framework(args)
        fw_kwargs = self.fw.to_framework(kwargs)
        solution_fn = getattr(self.solution, name)
        reference_fn = getattr(self.reference, name)
        return self.compare_speed(
            lambda: solution_fn(*fw_args, **fw_kwargs),
            lambda: reference_fn(*fw_args, **fw_kwargs),
            max_ratio=max_ratio,
            label=name,
            rounds=rounds,
            warmup=warmup,
            tokens=tokens,
        )

    def compare_speed(
        self,
        run_solution: Callable[[], Any],
        run_reference: Callable[[], Any],
        *,
        max_ratio: float,
        label: str = "solution",
        rounds: int = 7,
        warmup: int = 2,
        tokens: int | None = None,
    ) -> float:
        """Speed judge for anything that is not one call of one entry point.

        ``run_solution`` and ``run_reference`` are zero-argument callables; whatever
        they return is forced (``fw.sync``) before the clock stops, so return the
        arrays the work produces. Use this when the timed thing is built first: a
        function compiled with ``mx.compile``, a generation loop over a kit model, a
        step applied to a prepared cache. Build (and compile) outside the callables;
        the ``warmup`` calls then absorb tracing and kernel compilation, so only
        steady-state time is compared. Same procedure and metrics as ``check_speed``.
        """
        from whetstone.judge import speed

        result = speed.compare(
            lambda: self.fw.sync(run_solution()),
            lambda: self.fw.sync(run_reference()),
            rounds=rounds,
            warmup=warmup,
        )
        speed.require(result, max_ratio=max_ratio, label=label, ctx=self, tokens=tokens)
        return result.ratio

    # -- inputs ----------------------------------------------------------------

    def randn(self, *shape: int, scale: float = 1.0, loc: float = 0.0) -> np.ndarray:
        """Standard normal float32 array of ``shape`` (times ``scale``, plus ``loc``)."""
        values = self.rng.standard_normal(shape, dtype=np.float32)
        return (values * np.float32(scale) + np.float32(loc)).astype(np.float32)

    def uniform(self, *shape: int, low: float = 0.0, high: float = 1.0) -> np.ndarray:
        """Uniform float32 array in [low, high). The "randomness is an input" source."""
        values = self.rng.random(shape, dtype=np.float32)
        return (values * np.float32(high - low) + np.float32(low)).astype(np.float32)

    def randint(self, low: int, high: int, shape: int | tuple[int, ...] = ()) -> np.ndarray:
        """Integers in [low, high) as an int64 array (adapters narrow as needed)."""
        return self.rng.integers(low, high, size=shape, dtype=np.int64)

    def child_rng(self, index: int) -> np.random.Generator:
        """An independent generator for trial ``index`` of this case (property tests)."""
        return np.random.default_rng([self.seed, int(index)])

    # -- reporting -------------------------------------------------------------

    def metric(
        self,
        name: str,
        value: float,
        unit: str | None = None,
        limit: float | None = None,
        *,
        higher_is_better: bool = False,
    ) -> None:
        """Report a number with the case (shown in the UI; ``limit`` draws the bar).

        Reporting the same name again replaces the earlier value. Metrics are "lower
        is better" unless ``higher_is_better`` says otherwise (a throughput); that
        decides which case's value the run reports as its worst.
        """
        metric = protocol.clean_metric(name, value, unit, limit, higher_is_better=higher_is_better)
        if metric is None:
            return
        self.metrics[:] = [m for m in self.metrics if m["name"] != metric["name"]]
        self.metrics.append(metric)

    # -- files: model assets and fixtures ------------------------------------------

    def asset(self, name: str, file: str | None = None) -> Path:
        """The directory of a model asset, or the path of one file in it.

            ctx.asset("qwen3-0.6b")                        # the directory
            ctx.asset("qwen3-0.6b", "model.safetensors")   # one file

        ``name`` is an asset declared in ``[[assets]]`` of problem.toml. A case that
        needs an asset which is not downloaded is skipped before it starts, so inside
        a case the asset is there, unless the case opted out with ``assets=[...]`` and
        asks anyway: that raises ``AssetMissing``.
        """
        spec = self.problem.asset(name)
        if spec is None:
            declared = ", ".join(s.name for s in self.problem.assets) or "none"
            raise LookupError(
                f"asset {name!r} is not declared in `[[assets]]` of problem.toml "
                f"(declared: {declared})"
            )
        return assets.resolve(spec, self._store.models_dir, file)

    def fixture_path(self, name: str) -> Path:
        """The path of a file in the problem's ``fixtures/`` directory."""
        path = (self.problem.fixtures_path / name).resolve()
        if self.problem.fixtures_path.resolve() not in path.parents or not path.is_file():
            raise FileNotFoundError(f"no fixture {name!r} in {self.problem.fixtures_path}")
        return path

    def fixture(self, name: str) -> Any:
        """Load a committed reference output from the problem's ``fixtures/`` directory.

        By extension: ``.npz`` gives a dict of numpy arrays, ``.npy`` an array,
        ``.json`` the parsed JSON, ``.txt`` a string; anything else comes back as
        ``bytes``. Pickled data is never loaded.
        """
        path = self.fixture_path(name)
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                return {key: data[key] for key in data.files}
        if path.suffix == ".npy":
            return np.load(path, allow_pickle=False)
        if path.suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        if path.suffix == ".txt":
            return path.read_text(encoding="utf-8")
        return path.read_bytes()

    # -- judge-specific helpers --------------------------------------------------

    def simclock(self) -> Any:
        """A fresh ``SimClock`` (virtual time), closed automatically after the case."""
        from whetstone.judge.simclock import SimClock

        sim = SimClock()
        self.defer(sim.close)
        return sim

    def serve(self, app: Any = None, **options: Any) -> Any:
        """Serve an ASGI app (default: the solution's ``app``) on an ephemeral port.

        Use as ``with ctx.serve() as client:``; ``client`` is an ``httpx.Client`` bound
        to the server. The server is always shut down, including on failure or timeout.
        """
        from whetstone.judge.service import serve

        if app is None:
            app = self.solution.app
        return serve(app, **options)

    def defer(self, cleanup: Callable[[], None]) -> None:
        """Run ``cleanup`` after the case finishes, whatever the outcome."""
        self._cleanups.append(cleanup)

    # -- lifecycle (used by the harness) -----------------------------------------

    @contextmanager
    def active(self) -> Iterator[Context]:
        token = _current.set(self)
        try:
            yield self
        finally:
            _current.reset(token)

    def close(self) -> None:
        while self._cleanups:
            cleanup = self._cleanups.pop()
            # Cleanup must not mask the outcome of the case itself.
            with contextlib.suppress(Exception):
                cleanup()


def make_context(problem: Problem, solution: ModuleType, framework: str, case: Any) -> Context:
    """Build a Context outside the harness (unit tests of judge helpers)."""
    store = ModuleStore(problem, solution, get(framework), get(problem.oracle_framework))
    return Context(store, case)


def _summarize(value: Any) -> str:
    if isinstance(value, np.ndarray):
        head = f"{value.dtype} array, shape {tuple(value.shape)}"
        if value.size <= 16:
            body = np.array2string(value, precision=6, separator=", ", max_line_width=88)
            return f"{head}: {body}" if "\n" not in body else f"{head}:\n{body}"
        return head
    text = repr(value)
    return text if len(text) <= 200 else text[:197] + "..."


def describe_call(fn: Callable[..., Any], name: str, args: tuple, kwargs: dict) -> str:
    """Human-readable description of a call, for failure details."""
    try:
        bound = inspect.signature(fn).bind(*args, **kwargs)
        items = list(bound.arguments.items())
    except TypeError, ValueError:
        items = [(f"arg{i}", a) for i, a in enumerate(args)] + list(kwargs.items())
    lines = [f"call: {name}({', '.join(key for key, _ in items)})"]
    for key, value in items:
        text = _summarize(value)
        if "\n" in text:
            first, rest = text.split("\n", 1)
            lines.append(f"  {key} = {first}")
            lines.extend(f"    {line}" for line in rest.splitlines())
        else:
            lines.append(f"  {key} = {text}")
    return "\n".join(lines)
