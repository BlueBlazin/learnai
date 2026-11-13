"""The object every test case receives: ``def my_case(ctx): ...``."""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import importlib.util
import inspect
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import ModuleType
from typing import Any

import numpy as np

from whetstone import protocol
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
        self, problem: Problem, solution: ModuleType, fw: Framework, oracle_fw: Framework
    ) -> None:
        self.problem = problem
        self.solution = solution
        self.fw = fw
        self.oracle_fw = oracle_fw
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
        oracle_fw   adapter for the oracle's framework (torch, or python).
        solution    the learner's module.
        oracle      the oracle reference module (reference/torch.py or python.py).
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
        **kwargs: Any,
    ) -> float:
        """Speed judge: the solution must be within ``max_ratio`` of the reference.

        Times the solution's ``name`` against the same-framework reference's, in this
        process, interleaved, after ``warmup`` untimed calls each, taking the median of
        ``rounds`` measurements and forcing lazy frameworks to finish before the clock
        stops. Reports the ``slowdown`` metric and returns the ratio.
        """
        from whetstone.judge import speed

        fw_args = self.fw.to_framework(args)
        fw_kwargs = self.fw.to_framework(kwargs)
        solution_fn = getattr(self.solution, name)
        reference_fn = getattr(self.reference, name)
        result = speed.compare(
            lambda: self.fw.sync(solution_fn(*fw_args, **fw_kwargs)),
            lambda: self.fw.sync(reference_fn(*fw_args, **fw_kwargs)),
            rounds=rounds,
            warmup=warmup,
        )
        speed.require(result, max_ratio=max_ratio, label=name, ctx=self)
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
        self, name: str, value: float, unit: str | None = None, limit: float | None = None
    ) -> None:
        """Report a number with the case (shown in the UI; ``limit`` draws the bar).

        Reporting the same name again replaces the earlier value.
        """
        metric = protocol.clean_metric(name, value, unit, limit)
        if metric is None:
            return
        self.metrics[:] = [m for m in self.metrics if m["name"] != metric["name"]]
        self.metrics.append(metric)

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
