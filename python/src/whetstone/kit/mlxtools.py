"""Instruments for MLX problems: where evaluation happens, how often a compiled
function is traced, and how much memory a piece of work takes.

    from whetstone.kit import EvalSpy, TraceCounter, mlx_memory, pending_ops, watch_load, LazyClock

MLX builds a graph and computes nothing until something forces it: ``mx.eval``,
``mx.async_eval``, or a read that needs the numbers (``.item()``, ``.tolist()``,
``np.array(x)``, ``print(x)``). Where a program forces evaluation decides its speed
and its memory, and none of it shows in the result. These tools make it visible so a
test can assert on it.

This module imports ``mlx`` when a tool is used, not when the kit is imported.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Self

from whetstone.kit.counters import Counters

#: Calls that have to compute an array to return: ``array.<method>()``.
_READ_METHODS = frozenset({"item", "tolist"})
#: Builtins and numpy functions that compute their first argument when it is an array.
_READ_BUILTINS = frozenset(
    {"float", "int", "bool", "complex", "str", "repr", "format", "print", "memoryview", "bytes"}
)
_READ_NUMPY = frozenset({"array", "asarray", "asanyarray", "ascontiguousarray"})


@dataclass(frozen=True)
class SpyEvent:
    """One evaluation the spy saw.

    ``kind`` is ``"eval"`` (``mx.eval``), ``"async_eval"`` (``mx.async_eval``), or
    ``"read"`` (an implicit evaluation). ``what`` names the call (``"item"``,
    ``"np.array"``, ``"print"``); ``line`` is the line of the watched file that made it.
    """

    kind: str
    what: str
    line: int


class EvalSpy:
    """Records every evaluation an MLX solution triggers, in order, with its line.

        with EvalSpy() as spy:
            tokens = ctx.solution.generate(model, prompt, 8)
        check.that(spy.evals == 8, f"expected one mx.eval per token, saw {spy.evals}")
        check.that(spy.implicit == 0, "the loop reads arrays (.item(), print, ...)")

    ``spy.events`` is the ordered list of ``SpyEvent``; ``spy.evals``,
    ``spy.async_evals``, and ``spy.implicit`` count the three kinds; ``spy.kinds()`` is
    the sequence of kinds, handy for asserting on order ("each ``async_eval`` of step
    ``t + 1`` comes before the read of step ``t``").

    Only calls made *by code in the watched file* count: by default the solution being
    judged, or pass ``file=``. Work the kit or MLX does on its own is not the
    solution's doing. Calls are observed with ``sys.monitoring``, so nothing in MLX is
    patched and ``from mlx.core import eval`` is seen like ``mx.eval``.

    What it sees as a read: ``x.item()``, ``x.tolist()``; ``np.array(x)`` and
    ``np.asarray(x)``; ``float(x)``, ``int(x)``, ``bool(x)``, ``str(x)``, ``repr(x)``,
    ``format(x)``, ``print(x)``, ``memoryview(x)``, with the array as the first
    argument. What it cannot see, because Python makes no call for it: ``if x:`` and
    ``while x:`` on an array, and an array interpolated in an f-string. Write a mutant
    for a read that the spy does see.
    """

    def __init__(self, file: str | None = None) -> None:
        if file is None:
            from whetstone.judge.context import current_or_none

            ctx = current_or_none()
            file = getattr(getattr(ctx, "solution", None), "__file__", None)
            if file is None:
                raise ValueError("EvalSpy needs file= when no case with a solution is running")
        self.file = str(file)
        self.events: list[SpyEvent] = []
        self._tool: int | None = None

    # -- results ---------------------------------------------------------------------

    def count(self, kind: str) -> int:
        return sum(1 for event in self.events if event.kind == kind)

    @property
    def evals(self) -> int:
        """Number of ``mx.eval`` calls."""
        return self.count("eval")

    @property
    def async_evals(self) -> int:
        """Number of ``mx.async_eval`` calls."""
        return self.count("async_eval")

    @property
    def implicit(self) -> int:
        """Number of reads that forced an evaluation without asking for one."""
        return self.count("read")

    def kinds(self) -> list[str]:
        return [event.kind for event in self.events]

    def describe(self) -> str:
        """One line per event, for a failure detail."""
        return "".join(f"line {e.line}: {e.kind} ({e.what})\n" for e in self.events)

    # -- observing -------------------------------------------------------------------

    def __enter__(self) -> Self:
        import mlx.core as mx

        self._mx = mx
        monitoring = sys.monitoring
        for tool in (4, 3, 2, 1):
            try:
                monitoring.use_tool_id(tool, "whetstone-evalspy")
            except ValueError:
                continue
            self._tool = tool
            break
        else:
            raise RuntimeError("EvalSpy: no free sys.monitoring tool id (nested spies?)")
        monitoring.register_callback(self._tool, monitoring.events.CALL, self._on_call)
        monitoring.set_events(self._tool, monitoring.events.CALL)
        monitoring.restart_events()  # call sites an earlier spy switched off
        return self

    def __exit__(self, *exc: object) -> None:
        monitoring = sys.monitoring
        if self._tool is not None:
            monitoring.set_events(self._tool, 0)
            monitoring.register_callback(self._tool, monitoring.events.CALL, None)
            monitoring.free_tool_id(self._tool)
            self._tool = None

    def _on_call(self, code: Any, offset: int, function: Any, arg0: Any) -> Any:
        if code.co_filename != self.file:
            return sys.monitoring.DISABLE  # never ask about this call site again
        mx = self._mx
        kind = what = None
        if function is mx.eval:
            kind, what = "eval", "mx.eval"
        elif function is mx.async_eval:
            kind, what = "async_eval", "mx.async_eval"
        else:
            name = getattr(function, "__name__", "")
            owner = getattr(function, "__self__", None)
            if name in _READ_METHODS and (
                isinstance(arg0, mx.array) or isinstance(owner, mx.array)
            ):
                kind, what = "read", name
            elif isinstance(arg0, mx.array):
                module = getattr(function, "__module__", "") or ""
                if name in _READ_NUMPY and module.split(".")[0] == "numpy":
                    kind, what = "read", f"np.{name}"
                elif name in _READ_BUILTINS and (
                    module == "builtins" or isinstance(function, type)
                ):
                    kind, what = "read", name
        if kind is not None:
            self.events.append(SpyEvent(kind, what, sys._getframe(1).f_lineno))
        return None


class TraceCounter:
    """Counts how many times the Python body of a function runs.

        tc = TraceCounter()

        def step(x):
            return tc.mark(x) * 2          # mark is the identity

        fast = mx.compile(step)
        for _ in range(50):
            fast(mx.array(1.0))
        tc.count                           # 1: traced once, replayed 49 times

    ``mx.compile`` runs the Python function only to *trace* it; later calls with the
    same input shapes and types replay the recorded graph without running Python. So
    the number of times the body runs is the number of traces. Hand ``tc.mark`` (or a
    kit model, whose ``counters.forward_calls`` counts the same thing) to the solution
    and assert on ``tc.count``: a retrace per call, the classic mistake, shows as
    ``count == calls``.
    """

    def __init__(self) -> None:
        self.count = 0

    def mark(self, x: Any) -> Any:
        """Return ``x`` unchanged; the call itself is what is counted."""
        self.count += 1
        return x

    def wrap(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """``fn`` with every run of its body counted."""

        def counted(*args: Any, **kwargs: Any) -> Any:
            self.count += 1
            return fn(*args, **kwargs)

        return counted

    def reset(self) -> None:
        self.count = 0


class MemoryMeasurement:
    """What ``mlx_memory`` measured, in bytes (see there)."""

    def __init__(self) -> None:
        self.baseline_bytes = 0
        self.peak_bytes = 0
        self.active_bytes = 0
        self.cache_bytes = 0

    @property
    def peak_mib(self) -> float:
        return self.peak_bytes / 2**20

    @property
    def active_mib(self) -> float:
        return self.active_bytes / 2**20

    def __repr__(self) -> str:
        return f"MemoryMeasurement(peak={self.peak_mib:.2f} MiB, active={self.active_mib:.2f} MiB)"


@contextmanager
def mlx_memory() -> Iterator[MemoryMeasurement]:
    """Measure the MLX memory a block of work takes.

        with mlx_memory() as mem:
            out = ctx.call("forward_f16", params, x)   # ctx.call forces the result
        check.memory(mem, at_most_mib=24)

    On entry the buffer cache is emptied (``mx.clear_cache``), the high-water mark is
    reset (``mx.reset_peak_memory``), and the memory already in use is noted
    (``mx.get_active_memory``). On exit, after ``mx.synchronize``:

    ``peak_bytes``    ``mx.get_peak_memory()`` minus what was in use on entry: the most
                      the block held at one time, over and above its inputs.
    ``active_bytes``  ``mx.get_active_memory()`` minus the same: what the block left
                      allocated (its results, if they are still referenced).
    ``cache_bytes``   ``mx.get_cache_memory()``: freed buffers MLX keeps for reuse.

    MLX is lazy: a result that was never evaluated has allocated nothing, so force the
    work *inside* the block (``ctx.call`` does; otherwise ``ctx.fw.sync(out)``).
    Inputs created before the block are not counted; arrays created inside it are.
    Sizes are exact for the arrays themselves, but the allocator rounds and a kernel
    may use scratch space, so compare with a limit that leaves a margin
    (``check.memory``), never for equality.
    """
    import mlx.core as mx

    measurement = MemoryMeasurement()
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    measurement.baseline_bytes = baseline = mx.get_active_memory()
    try:
        yield measurement
    finally:
        mx.synchronize()
        measurement.peak_bytes = max(0, mx.get_peak_memory() - baseline)
        measurement.active_bytes = max(0, mx.get_active_memory() - baseline)
        measurement.cache_bytes = mx.get_cache_memory()


def pending_ops(*trees: Any) -> int:
    """Number of operations not yet computed behind the arrays in ``trees``.

        check.that(pending_ops(model.parameters()) == 0, "the parameters are still lazy")

    MLX builds a graph and computes it when asked; an array that has been evaluated
    has nothing behind it. ``trees`` may be arrays, or lists, tuples, and dicts of
    them (``module.parameters()``, a cache); anything else is ignored. Zero means
    every array is evaluated. The count comes from ``mx.export_to_dot``, which lists
    one node per pending operation; it is meant for "zero or not" and "bounded"
    checks, not for exact graph sizes.
    """
    import io

    import mlx.core as mx
    from mlx.utils import tree_flatten

    arrays = [leaf for _, leaf in tree_flatten(list(trees)) if isinstance(leaf, mx.array)]
    if not arrays:
        return 0
    dot = io.StringIO()
    mx.export_to_dot(dot, *arrays)
    return dot.getvalue().count("shape=rectangle")


@contextmanager
def watch_load() -> Iterator[dict[str, Any]]:
    """Collect the arrays ``mx.load`` returns while the block runs: ``{name: array}``.

        with watch_load() as loaded:
            ctx.solution.load_into(model, path)
        check.that(pending_ops(loaded["unused.weight"]) > 0, "an unused tensor was read")

    ``mx.load`` of a safetensors or npz file returns lazy arrays: nothing is read from
    disk until an array is evaluated. Holding the very arrays the solution was given
    lets a test see which of them were read (``pending_ops``). Only ``mx.load`` looked
    up at call time is seen, which is how solutions call it.
    """
    import mlx.core as mx

    seen: dict[str, Any] = {}
    original = mx.load

    def load(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        tensors = result[0] if isinstance(result, tuple) else result
        if isinstance(tensors, dict):
            seen.update(tensors)
        return result

    mx.load = load
    try:
        yield seen
    finally:
        mx.load = original


class LazyClock:
    """A simulated clock on which time passes only when lazy work is really computed.

        clock = LazyClock()
        logits = model(tokens)                 # a lazy graph
        clock.spend(0.0001)                    # building it cost this much, at once
        clock.owe(logits, 0.004)               # computing it will cost this much
        clock()                                # 0.0001: nothing has been computed yet
        mx.eval(logits)
        clock()                                # 0.0041: the debt was collected

    For problems about *measuring* MLX code. Real timings are noisy; this clock makes
    the mistakes exact. ``clock()`` returns simulated seconds, as ``time.perf_counter``
    would. ``spend(seconds)`` advances it immediately. ``owe(array, seconds, group=None)``
    attaches a cost to a lazy array: the clock collects it at the first reading after
    the array has been evaluated (``pending_ops(array) == 0``). An interval timed
    around an unevaluated graph therefore contains none of its work, and the work
    lands in whichever interval ends after someone evaluates.

    ``group`` ties debts together (the calls of one generation run): a group's debts
    are collected in order, and a later debt waits for the earlier ones. ``slow``
    multiplies the cost of the k-th group to be collected by ``slow[k]`` (warm-up:
    the first computed run is slower); groups beyond the list cost their plain price.

    ``clock.reads`` counts the readings; ``clock.now`` is the time without reading.
    """

    def __init__(self, slow: Any = ()) -> None:
        self.now = 0.0
        self.slow = list(slow)
        self.reads = 0
        self._debts: list[tuple[Any, Any, float]] = []
        self._order: dict[Any, int] = {}
        self._anonymous = 0

    def spend(self, seconds: float) -> None:
        self.now += seconds

    def owe(self, array: Any, seconds: float, group: Any = None) -> None:
        if group is None:
            self._anonymous += 1
            group = ("anonymous", self._anonymous)
        self._debts.append((array, group, seconds))

    def __call__(self) -> float:
        self.reads += 1
        waiting = []
        blocked = set()  # groups with an uncomputed debt: the debts after it depend on it
        for array, group, seconds in self._debts:
            if group in blocked or pending_ops(array):
                blocked.add(group)
                waiting.append((array, group, seconds))
                continue
            position = self._order.setdefault(group, len(self._order))
            self.now += seconds * (self.slow[position] if position < len(self.slow) else 1.0)
        self._debts = waiting
        return self.now


class CountingFile:
    """A read-only file that counts what is read from it, as a file or as a mapping.

        with CountingFile(path) as f:
            header_len = int.from_bytes(f[:8], "little")       # mapping style
            header = json.loads(bytes(f[8 : 8 + header_len]))
            f.seek(offset); chunk = f.read(4096)                # file style
        f.counters.bytes_read      # bytes asked for, in total
        f.counters.reads           # number of reads and slices
        f.pages_touched            # distinct pages those ranges cover

    The file is memory-mapped. A slice ``f[a:b]`` is a zero-copy ``memoryview`` of
    the mapping (``np.frombuffer(f[a:b], dtype)`` wraps it without copying), and it
    counts as reading ``b - a`` bytes: the pages a program would fault in by using
    that range. So a loader that takes ``f[:]`` or ``f.read()`` and picks out one
    tensor is charged for the whole file, and one that slices only the tensors it
    needs is charged for those. ``page_size`` defaults to 16 KiB, the page size of
    Apple silicon. Counting is by the ranges requested, not by faults the operating
    system observed, so it is exact and the same on every machine.
    """

    def __init__(self, path: Any, page_size: int = 16384) -> None:
        import mmap

        self.path = str(path)
        self.page_size = int(page_size)
        self.counters = Counters("bytes_read", "reads")
        with open(self.path, "rb") as handle:
            self._map = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        self._view = memoryview(self._map)
        self._position = 0
        self._pages: set[int] = set()

    def _charge(self, start: int, stop: int) -> None:
        if stop <= start:
            return
        self.counters.add("bytes_read", stop - start)
        self.counters.add("reads")
        self._pages.update(range(start // self.page_size, (stop - 1) // self.page_size + 1))

    @property
    def pages_touched(self) -> int:
        """Number of distinct pages covered by everything read so far."""
        return len(self._pages)

    # -- as a mapping ----------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._view)

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self._view))
            if step != 1:
                raise ValueError("CountingFile slices must be contiguous (step 1)")
            self._charge(start, stop)
            return self._view[start:stop]
        position = index if index >= 0 else len(self._view) + index
        self._charge(position, position + 1)
        return self._view[position]

    # -- as a file -------------------------------------------------------------------

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self._position, 2: len(self._view)}[whence]
        self._position = max(0, base + offset)
        return self._position

    def tell(self) -> int:
        return self._position

    def read(self, size: int = -1) -> bytes:
        stop = len(self._view) if size is None or size < 0 else self._position + size
        stop = min(stop, len(self._view))
        start = min(self._position, stop)
        self._charge(start, stop)
        self._position = stop
        return bytes(self._view[start:stop])

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def close(self) -> None:
        # Views handed out may still be alive; the mapping closes when they are gone.
        self._view = memoryview(b"")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"CountingFile({self.path!r}, bytes_read={self.counters.bytes_read})"
