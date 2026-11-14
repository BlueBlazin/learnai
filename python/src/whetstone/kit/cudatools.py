"""Instruments for CUDA problems that run on the CPU.

    from whetstone.kit import (
        SyncSpy, DeviceSpy, FakeDevice, RealDevice, OOMModel, FakeCudaGraph,
        CompileCounter, TransferSpy,
    )

A GPU hides its mistakes. A decode loop that calls ``.item()`` every step returns the
right tokens; a timer without ``torch.cuda.synchronize`` returns a number; a CUDA graph
whose placeholder was rebound replays without complaint. The cost is time, memory, or
stale data, and none of it shows on a CPU, where every operation finishes before the
next line runs. Each tool here makes one of those mistakes visible *on CPU tensors*, in
the way it would bite on a GPU, so that the logic of a CUDA problem can be judged on a
machine without one:

    SyncSpy          counts host synchronizations made by the solution file
    DeviceSpy        finds tensors created without an explicit ``device=``
    FakeDevice       a device whose work finishes only at ``sync()``, on a virtual clock
    RealDevice       the same interface over ``torch.cuda.synchronize`` and ``perf_counter``
    OOMModel         a model that raises ``torch.OutOfMemoryError`` above a batch size
    FakeCudaGraph    ``torch.cuda.CUDAGraph``'s surface: capture, then replay on the
                     captured memory
    CompileCounter   how many graphs ``torch.compile`` built, and how many graph breaks
    TransferSpy      ``pin`` and ``to_device`` stand-ins that record what happened when

``torch`` is imported when a tool is used, not when the kit is imported. The API names
and behaviors these tools imitate were checked against torch 2.15 (see
docs/PROBLEM_AUTHORING.md, "CUDA problems", for what was verified and what cannot be
verified without a GPU).
"""

from __future__ import annotations

import sys
import time
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Self

from whetstone.kit.counters import Counters


def _solution_file(file: str | None, who: str) -> str:
    if file is not None:
        return str(file)
    from whetstone.judge.context import current_or_none

    ctx = current_or_none()
    found = getattr(getattr(ctx, "solution", None), "__file__", None)
    if found is None:
        raise ValueError(f"{who} needs file= when no case with a solution is running")
    return str(found)


class _Patches:
    """Attributes replaced for the length of a ``with`` block, restored exactly."""

    def __init__(self) -> None:
        self._undo: list[tuple[Any, str, bool, Any]] = []

    def set(self, owner: Any, name: str, value: Any) -> None:
        own = vars(owner) if isinstance(owner, type) else owner.__dict__
        had = name in own
        self._undo.append((owner, name, had, own.get(name)))
        setattr(owner, name, value)

    def restore(self) -> None:
        while self._undo:
            owner, name, had, old = self._undo.pop()
            if had:
                setattr(owner, name, old)
            else:
                delattr(owner, name)  # the inherited attribute shows through again


# --------------------------------------------------------------------------------------
# SyncSpy: host synchronizations
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SyncEvent:
    """One host synchronization the spy saw.

    ``kind`` is ``"read"`` (a value was pulled to Python: ``.item()``, ``bool(x)``,
    ``print(x)``), ``"transfer"`` (``.cpu()``, ``.numpy()``: the whole tensor was
    copied to the host), or ``"shape"`` (an operation whose output size depends on the
    data, such as ``nonzero`` or a boolean-mask index, which makes the host wait for
    the result to learn the size). ``what`` names the call; ``line`` is the line of
    the watched file that made it.
    """

    kind: str
    what: str
    line: int


#: Tensor methods that hand a value to Python: the host waits for the GPU to finish.
_READS = {
    "item": "item",
    "tolist": "tolist",
    "__bool__": "bool",
    "__int__": "int",
    "__float__": "float",
    "__complex__": "complex",
    "__index__": "index",
    "__repr__": "print",  # print(x), str(x), repr(x)
    "__format__": "format",  # f"{x}"
    "__contains__": "in",
    "equal": "equal",
    "allclose": "allclose",
}
#: Tensor methods that copy the whole tensor to the host.
_TRANSFERS = {"cpu": "cpu", "numpy": "numpy", "__array__": "np.asarray"}
#: Operations whose output shape depends on the values of the input.
_DYNAMIC = ("nonzero", "argwhere", "masked_select", "unique", "unique_consecutive")


class SyncSpy:
    """Counts the host synchronizations a torch solution makes, on CPU tensors.

        with SyncSpy() as spy:
            tokens = ctx.solution.decode(model, prompt, 64, uniforms, torch.device("cpu"))
        check.at_most("host_syncs", spy.syncs, 64 // check_every + 1)

    On a GPU every one of these calls makes Python wait until all queued work is done
    and copies data across; in a decode loop that is once per token. On the CPU they
    cost nothing, which is why the mistake survives until the code meets a GPU. The
    spy replaces the methods on ``torch.Tensor`` for the length of the block, so it
    also sees what makes no visible call: ``if x:``, ``while x:``, ``x in seen``, an
    f-string, ``range(x)``.

    ``spy.events`` is the ordered list of ``SyncEvent``; ``spy.syncs`` is its length;
    ``spy.reads``, ``spy.transfers``, and ``spy.shape_syncs`` count the kinds;
    ``spy.describe()`` is one line per event for a failure detail.

    Only calls made *by code in the watched file* count (the solution being judged,
    or ``file=``): what the kit, the test, or torch do internally is not the
    solution's doing. One call counts once: ``print(x)`` is one event although torch
    reads many values to format it.

    Counted: ``x.item()``, ``x.tolist()``, ``bool(x)`` (also ``if``, ``while``,
    ``and``, ``or``, ``not``, ``max(a, b)``), ``int(x)``, ``float(x)``, using ``x`` as
    an index or in ``range``, ``print(x)``, ``str(x)``, ``repr(x)``, f-strings,
    ``y in x``, ``x.equal(y)``, ``x.allclose(y)`` and the ``torch.`` forms; ``x.cpu()``,
    ``x.numpy()``, ``np.asarray(x)``; ``nonzero``, ``argwhere``, ``masked_select``,
    ``unique``, ``torch.where(cond)`` with one argument, and ``x[mask]`` with a
    boolean mask.

    Not counted: ``x.to(device)``. With ``device`` a CPU device it is a no-op here,
    and whether it would synchronize on a GPU depends on where ``x`` lives, which a
    CPU run cannot know. Specify ``.cpu()`` for the final transfer in the statement.
    """

    def __init__(self, file: str | None = None) -> None:
        self.file = _solution_file(file, "SyncSpy")
        self.events: list[SyncEvent] = []
        self._patches = _Patches()

    # -- results ---------------------------------------------------------------------

    def count(self, kind: str) -> int:
        return sum(1 for event in self.events if event.kind == kind)

    @property
    def syncs(self) -> int:
        """Number of host synchronizations of any kind."""
        return len(self.events)

    @property
    def reads(self) -> int:
        return self.count("read")

    @property
    def transfers(self) -> int:
        return self.count("transfer")

    @property
    def shape_syncs(self) -> int:
        return self.count("shape")

    def describe(self) -> str:
        """One line per event, for a failure detail."""
        return "".join(f"line {e.line}: {e.kind} ({e.what})\n" for e in self.events)

    # -- observing -------------------------------------------------------------------

    def _record(self, kind: str, what: str, depth: int = 2) -> None:
        frame = sys._getframe(depth)
        if frame.f_code.co_filename == self.file:
            self.events.append(SyncEvent(kind, what, frame.f_lineno))

    def _wrap(self, original: Callable[..., Any], kind: str, what: str) -> Callable[..., Any]:
        def spied(*args: Any, **kwargs: Any) -> Any:
            self._record(kind, what)
            return original(*args, **kwargs)

        spied.__name__ = getattr(original, "__name__", what)
        return spied

    def __enter__(self) -> Self:
        import torch

        tensor = torch.Tensor
        for name, what in _READS.items():
            self._patches.set(tensor, name, self._wrap(getattr(tensor, name), "read", what))
        for name, what in _TRANSFERS.items():
            self._patches.set(tensor, name, self._wrap(getattr(tensor, name), "transfer", what))
        for name in _DYNAMIC:
            self._patches.set(tensor, name, self._wrap(getattr(tensor, name), "shape", name))
            self._patches.set(torch, name, self._wrap(getattr(torch, name), "shape", name))
        for name in ("equal", "allclose"):
            self._patches.set(torch, name, self._wrap(getattr(torch, name), "read", name))

        getitem, where = tensor.__getitem__, torch.where

        def spied_getitem(self_tensor: Any, index: Any) -> Any:
            parts = index if isinstance(index, tuple) else (index,)
            if any(isinstance(p, tensor) and p.dtype == torch.bool for p in parts):
                self._record("shape", "boolean-mask index")
            return getitem(self_tensor, index)

        def spied_where(*args: Any, **kwargs: Any) -> Any:
            if len(args) + len(kwargs) == 1:
                self._record("shape", "where(condition)")
            return where(*args, **kwargs)

        self._patches.set(tensor, "__getitem__", spied_getitem)
        self._patches.set(torch, "where", spied_where)
        return self

    def __exit__(self, *exc: object) -> None:
        self._patches.restore()


# --------------------------------------------------------------------------------------
# DeviceSpy: tensors created on the default device
# --------------------------------------------------------------------------------------

#: ``torch`` factory functions that take ``device=`` and use the default device without it.
_FACTORIES = (
    "arange", "zeros", "ones", "full", "empty", "eye", "linspace", "logspace", "rand",
    "randn", "randint", "randperm", "tensor", "tril_indices", "triu_indices",
)  # fmt: skip


@dataclass(frozen=True)
class FactoryEvent:
    """A tensor created by ``torch.<what>(...)`` at ``line`` of the watched file."""

    what: str
    line: int
    explicit: bool  # whether ``device=`` was passed (and was not None)


class DeviceSpy:
    """Finds tensors a solution creates without saying where.

        with DeviceSpy() as spy:
            out = ctx.solution.run(model, inputs, torch.device("cpu"))
        check.that(not spy.implicit, f"{len(spy.implicit)} tensors were created without "
                   "device=", spy.describe())

    ``torch.arange(n)`` puts its result on the default device, the CPU. Next to CUDA
    inputs that is an error at best and a silent host round trip at worst; on a CPU
    run it is invisible, because everything is on the CPU. The spy wraps the factory
    functions (``arange``, ``zeros``, ``ones``, ``full``, ``empty``, ``eye``,
    ``linspace``, ``rand``, ``randn``, ``randint``, ``randperm``, ``tensor``,
    ``tril_indices``, ``triu_indices``) and records every call made by the watched
    file: ``spy.events`` is all of them, ``spy.implicit`` those without an explicit
    ``device=``. The ``*_like`` functions and ``x.new_zeros(...)`` inherit the device
    of their argument and are not watched.

    Only ``torch.<name>`` looked up at call time is seen; a solution that did ``from
    torch import arange`` before the block escapes (the forbidden-call guard of
    section 4.1 has the same limit and the same remedy: forbid nothing, test more).
    """

    def __init__(self, file: str | None = None) -> None:
        self.file = _solution_file(file, "DeviceSpy")
        self.events: list[FactoryEvent] = []
        self._patches = _Patches()

    @property
    def implicit(self) -> list[FactoryEvent]:
        """Creations that fell back to the default device."""
        return [event for event in self.events if not event.explicit]

    def describe(self) -> str:
        return "".join(
            f"line {e.line}: torch.{e.what}(...) "
            f"{'with device=' if e.explicit else 'WITHOUT device='}\n"
            for e in self.events
        )

    def __enter__(self) -> Self:
        import torch

        def wrap(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
            def spied(*args: Any, **kwargs: Any) -> Any:
                frame = sys._getframe(1)
                if frame.f_code.co_filename == self.file:
                    explicit = kwargs.get("device") is not None
                    self.events.append(FactoryEvent(name, frame.f_lineno, explicit))
                return original(*args, **kwargs)

            spied.__name__ = name
            return spied

        for name in _FACTORIES:
            self._patches.set(torch, name, wrap(name, getattr(torch, name)))
        return self

    def __exit__(self, *exc: object) -> None:
        self._patches.restore()


# --------------------------------------------------------------------------------------
# FakeDevice: asynchronous work on a virtual clock
# --------------------------------------------------------------------------------------


class FakeDevice:
    """A device that works asynchronously, with a clock that only it advances.

        dev = FakeDevice()
        fn = dev.kernel(0.004, first_call_s=0.050)   # each call queues 4 ms of device work
        seconds = ctx.solution.time_region(fn, dev, 5)
        check.allclose(seconds, 0.004, rtol=1e-9, label="median seconds per call")

    A CUDA call returns as soon as the work is *queued*; the GPU does it later.
    ``time.perf_counter()`` right after the call therefore measures the queueing, not
    the work, unless ``torch.cuda.synchronize()`` waits first. Here:

    - ``dev.kernel(seconds, first_call_s=None)`` returns a function standing for a GPU
      operation. Calling it queues ``seconds`` of device work and returns at once (it
      costs the host ``launch_s``, zero by default). Its first call queues
      ``first_call_s`` instead, when given: warm-up (allocation, kernel compilation).
    - ``dev.launch(seconds)`` queues work directly (stale work left by "someone else").
    - ``dev.sync()`` stands in for ``torch.cuda.synchronize``: the host waits until the
      device queue is empty, and the clock moves to that moment.
    - ``dev.clock()`` stands in for ``time.perf_counter``: host time in seconds. It
      never waits for the device.

    The device runs its queue in order, concurrently with the host: work queued at
    host time ``t`` starts at ``max(t, when the device becomes free)``. So a timer read
    without a sync measures nothing of the work it was meant to time; a region not
    preceded by a sync pays for whatever was queued before it; and a first call
    included in the measurement shows as warm-up time.

    ``dev.counters`` has ``syncs``, ``clock_reads``, and ``launches``;
    ``dev.pending_s`` is the queued work not yet waited for; ``dev.events`` is the
    ordered log of ``("launch" | "sync" | "clock", host seconds)``. Time is kept in
    integer nanoseconds, so sums are exact; compare seconds with ``rtol=1e-9``.

    The solution receives ``dev`` as its ``device_api`` and calls only ``sync()`` and
    ``clock()``. On a GPU the same solution gets ``RealDevice``.
    """

    def __init__(self, launch_s: float = 0.0) -> None:
        self._launch_ns = self._ns(launch_s)
        self._now = 0  # host time
        self._free_at = 0  # when the device finishes everything queued so far
        self.counters = Counters("syncs", "clock_reads", "launches")
        self.events: list[tuple[str, float]] = []

    @staticmethod
    def _ns(seconds: float) -> int:
        if seconds < 0:
            raise ValueError("a duration cannot be negative")
        return round(seconds * 1e9)

    def launch(self, seconds: float) -> None:
        """Queue ``seconds`` of device work; returns without waiting for it."""
        self._free_at = max(self._free_at, self._now) + self._ns(seconds)
        self._now += self._launch_ns
        self.counters.add("launches")
        self.events.append(("launch", self._now / 1e9))

    def kernel(self, seconds: float, first_call_s: float | None = None) -> Callable[[], None]:
        """A zero-argument function that queues ``seconds`` of work each time it is called."""
        state = {"first": first_call_s is not None}

        def run() -> None:
            first, state["first"] = state["first"], False
            self.launch(first_call_s if first else seconds)  # type: ignore[arg-type]

        return run

    def sync(self) -> None:
        """Wait for the device: the stand-in for ``torch.cuda.synchronize()``."""
        self._now = max(self._now, self._free_at)
        self.counters.add("syncs")
        self.events.append(("sync", self._now / 1e9))

    def clock(self) -> float:
        """Host time in seconds: the stand-in for ``time.perf_counter()``."""
        self.counters.add("clock_reads")
        self.events.append(("clock", self._now / 1e9))
        return self._now / 1e9

    @property
    def pending_s(self) -> float:
        """Device work queued and not yet waited for, in seconds."""
        return max(0, self._free_at - self._now) / 1e9

    def __repr__(self) -> str:
        return f"FakeDevice(now={self._now / 1e9:g} s, pending={self.pending_s:g} s)"


class RealDevice:
    """``sync()`` and ``clock()`` for a real torch device: what ``FakeDevice`` imitates.

    ``sync()`` is ``torch.cuda.synchronize(device)`` on a CUDA device and does nothing
    on the CPU, where work is synchronous; ``clock()`` is ``time.perf_counter()``.
    """

    def __init__(self, device: Any = "cuda") -> None:
        import torch

        self.device = torch.device(device)
        self._torch = torch

    def sync(self) -> None:
        if self.device.type == "cuda":
            self._torch.cuda.synchronize(self.device)

    def clock(self) -> float:
        return time.perf_counter()


# --------------------------------------------------------------------------------------
# OOMModel: out of memory above a batch size
# --------------------------------------------------------------------------------------


class OOMModel:
    """A model that runs out of memory when its batch is larger than ``max_items``.

        model = OOMModel(lambda x: x * 2, max_items=3)
        out = ctx.solution.run_batched(model, items, 8, model.empty_cache)
        check.equal(model.batch_sizes, [8, 4, 2, 2, 2, 2])

    ``model(batch)`` returns ``fn(batch)`` when ``len(batch) <= max_items`` and
    otherwise raises ``torch.OutOfMemoryError`` with the text CUDA produces. That
    class is the one torch 2.15 raises; it is a subclass of ``RuntimeError``, and
    ``torch.cuda.OutOfMemoryError`` is an alias of it.

    What the test can read afterwards:

    - ``model.batch_sizes``: the size of every batch tried, in order, failed or not;
    - ``model.events``: the ordered log, ``("call", size, "ok" | "oom" | "error")``
      and ``("empty_cache", live)`` entries, to check that ``empty_cache`` ran between
      a failure and the retry;
    - ``model.counters``: ``calls``, ``ooms``, ``items`` (rows processed successfully),
      ``empty_cache_calls``, and ``live_at_empty_cache``;
    - ``model.empty_cache``: the stand-in for ``torch.cuda.empty_cache`` to hand to the
      solution.

    ``live_at_empty_cache`` counts ``empty_cache`` calls that could not have freed
    anything: the batch that failed was still referenced. That is the real trap. The
    exception holds the traceback, the traceback holds the frame of the failed call,
    and the frame holds the batch, so ``empty_cache()`` called inside the ``except``
    block finds the memory still in use. The fix is to leave the ``except`` block (and
    drop one's own references to the batch) first. Tracking needs a batch that can be
    weakly referenced: a tensor, not a list.

    ``errors={call_index: exception}`` makes the model raise something else on a
    given call (0-based), for example a shape error that a correct solution must not
    swallow just because ``OutOfMemoryError`` is a ``RuntimeError``.
    """

    def __init__(
        self,
        fn: Callable[[Any], Any],
        max_items: int,
        *,
        errors: dict[int, BaseException] | None = None,
    ) -> None:
        self.fn = fn
        self.max_items = max_items
        self.errors = dict(errors or {})
        self.counters = Counters(
            "calls", "ooms", "items", "empty_cache_calls", "live_at_empty_cache"
        )
        self.batch_sizes: list[int] = []
        self.events: list[tuple[Any, ...]] = []
        self._failed: list[weakref.ref[Any]] = []

    def __call__(self, batch: Any) -> Any:
        import torch

        size = len(batch)
        index = self.counters.calls
        self.counters.add("calls")
        self.batch_sizes.append(size)
        if index in self.errors:
            self.events.append(("call", size, "error"))
            raise self.errors[index]
        if size > self.max_items:
            self.counters.add("ooms")
            self.events.append(("call", size, "oom"))
            try:
                self._failed.append(weakref.ref(batch))
            except TypeError:  # a list or tuple: cannot be tracked
                pass
            raise torch.OutOfMemoryError(
                f"CUDA out of memory. Tried to allocate {size} items; this model fits "
                f"{self.max_items} (whetstone OOMModel)"
            )
        self.counters.add("items", size)
        self.events.append(("call", size, "ok"))
        return self.fn(batch)

    def empty_cache(self) -> None:
        """The stand-in for ``torch.cuda.empty_cache()``; records what was still alive."""
        live = sum(1 for ref in self._failed if ref() is not None)
        self._failed = []
        self.counters.add("empty_cache_calls")
        if live:
            self.counters.add("live_at_empty_cache")
        self.events.append(("empty_cache", live))

    def __repr__(self) -> str:
        return f"OOMModel(max_items={self.max_items}, {self.counters!r})"


# --------------------------------------------------------------------------------------
# FakeCudaGraph: capture and replay on the captured memory
# --------------------------------------------------------------------------------------


class FakeCudaGraph:
    """``torch.cuda.CUDAGraph``'s capture-and-replay, on CPU tensors.

        graph = FakeCudaGraph()
        step = ctx.solution.GraphedStep(step_fn, example_inputs, graph)
        out = step(new_inputs)

    A CUDA graph records the GPU kernels launched between ``capture_begin()`` and
    ``capture_end()`` together with the *memory addresses* they read and wrote.
    ``replay()`` launches the same kernels on the same addresses. Python is not
    involved: no code of the captured function runs again. Hence the rules of using
    one, each of which this class enforces the way the GPU does:

    - New inputs must be **copied into** the tensors that were inputs during capture
      (``static_in.copy_(x)``). Rebinding a name (``static_in = x``) changes nothing
      the graph can see: replay reads the old memory and returns the old answer.
    - Outputs are overwritten in place by every replay. A caller that keeps an output
      without ``clone()`` sees it change at the next call.
    - Nothing may synchronize with the host during capture: ``.item()``, ``bool(x)``,
      ``print(x)`` raise, as they do on a capturing CUDA stream.
    - Shapes are frozen: a replay whose recomputed shape differs raises.

    How: between ``capture_begin`` and ``capture_end`` a ``TorchDispatchMode`` records
    every operation with the tensor objects it read and returned. ``replay`` runs the
    recorded operations again on those same tensor objects and writes each result
    into the tensor that held it during capture.

    Surface, as in torch 2.15: ``capture_begin(pool=None, capture_error_mode="global")``,
    ``capture_end()``, ``replay()``, ``reset()``. Extras for tests: ``graph.capturing``,
    ``graph.captured``, ``graph.n_ops`` (operations recorded), and ``graph.counters``
    (``captures``, ``replays``). ``with graph.patched():`` makes
    ``torch.cuda.graph(graph)`` work for this object, so a solution written with the
    context manager, the form torch recommends, runs unchanged.

    Not imitated: streams, memory pools, and the requirement to warm up before
    capturing. To check the warm-up, wrap the step function in the test and look at
    ``graph.capturing`` on each call: a correct solution calls it with ``False``
    (warm-up) before the one call with ``True``, and never after.
    """

    def __init__(self, keep_graph: bool = False) -> None:
        self.counters = Counters("captures", "replays")
        self._ops: list[tuple[Any, tuple, dict, list[Any]]] = []
        self._mode: Any = None
        self.captured = False

    @property
    def capturing(self) -> bool:
        """True between ``capture_begin()`` and ``capture_end()``."""
        return self._mode is not None

    @property
    def n_ops(self) -> int:
        return len(self._ops)

    def capture_begin(self, pool: Any = None, capture_error_mode: str = "global") -> None:
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode

        if self.capturing:
            raise RuntimeError("This CUDAGraph is already capturing.")
        if self.captured:
            raise RuntimeError(
                "This CUDAGraph instance already owns a captured graph. "
                "To capture a new graph, create a new instance."
            )
        graph = self

        class Recorder(TorchDispatchMode):
            def __torch_dispatch__(
                self, func: Any, types: Any, args: tuple = (), kwargs: dict | None = None
            ) -> Any:
                kwargs = kwargs or {}
                if "_local_scalar_dense" in str(func):
                    raise RuntimeError(
                        "CUDA error: operation not permitted when stream is capturing "
                        "(a host read such as .item(), bool(x), or print(x) inside a "
                        "CUDA graph capture; whetstone FakeCudaGraph)"
                    )
                out = func(*args, **kwargs)
                graph._ops.append((func, args, kwargs, _tensors_of(out, torch)))
                return out

        self._mode = Recorder()
        self._mode.__enter__()

    def capture_end(self) -> None:
        if not self.capturing:
            raise RuntimeError("capture_end() called before capture_begin().")
        mode, self._mode = self._mode, None
        mode.__exit__(None, None, None)
        self.captured = True
        self.counters.add("captures")

    def replay(self) -> None:
        """Run the captured operations again, on the captured tensors."""
        import torch

        if self.capturing:
            raise RuntimeError("replay() called while capturing.")
        if not self.captured:
            raise RuntimeError("replay() called before capture_end().")
        self.counters.add("replays")
        # Inference mode: in-place writes are allowed whatever mode the capture ran in.
        with torch.inference_mode():
            for func, args, kwargs, outs in self._ops:
                new = _tensors_of(func(*args, **kwargs), torch)
                for old, fresh in zip(outs, new, strict=True):
                    if fresh is old:
                        continue  # an in-place operation: it wrote where it wrote before
                    if fresh.shape != old.shape:
                        raise RuntimeError(
                            f"FakeCudaGraph.replay: `{_op_name(func)}` produced shape "
                            f"{tuple(fresh.shape)}, but {tuple(old.shape)} was captured. A "
                            "CUDA graph replays fixed kernels on fixed memory: shapes "
                            "cannot depend on the data."
                        )
                    if _same_memory(old, fresh):
                        continue  # a view of memory that is already up to date
                    old.copy_(fresh)

    def reset(self) -> None:
        """Delete the captured graph."""
        if self.capturing:
            self.capture_end()
        self._ops = []
        self.captured = False

    @contextmanager
    def patched(self) -> Iterator[Self]:
        """Inside the block, ``torch.cuda.graph(g)`` accepts a ``FakeCudaGraph``."""
        import torch

        real = torch.cuda.graph

        class graph:
            def __new__(cls, cuda_graph: Any, *args: Any, **kwargs: Any) -> Any:
                if isinstance(cuda_graph, FakeCudaGraph):
                    return object.__new__(cls)
                return real(cuda_graph, *args, **kwargs)

            def __init__(self, cuda_graph: Any, *args: Any, **kwargs: Any) -> None:
                self.cuda_graph = cuda_graph

            def __enter__(self) -> None:
                self.cuda_graph.capture_begin()

            def __exit__(self, *exc: object) -> None:
                self.cuda_graph.capture_end()

        torch.cuda.graph = graph
        try:
            yield self
        finally:
            torch.cuda.graph = real

    def __repr__(self) -> str:
        state = "capturing" if self.capturing else "captured" if self.captured else "empty"
        return f"FakeCudaGraph({state}, ops={self.n_ops}, {self.counters!r})"


def _tensors_of(out: Any, torch: Any) -> list[Any]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        return [t for item in out for t in _tensors_of(item, torch)]
    return []


def _op_name(func: Any) -> str:
    return str(func).removeprefix("aten.").split(".")[0]


def _same_memory(a: Any, b: Any) -> bool:
    return (
        a.data_ptr() == b.data_ptr()
        and a.shape == b.shape
        and a.stride() == b.stride()
        and a.dtype == b.dtype
    )


# --------------------------------------------------------------------------------------
# CompileCounter: how many graphs torch.compile built
# --------------------------------------------------------------------------------------


class CompileCounter:
    """Counts what ``torch.compile`` did inside a block: compilations and graph breaks.

        with CompileCounter() as cc:
            step = ctx.solution.make_step(model, cache)       # torch.compile(..., backend="eager")
            for pos in range(64):
                logits = step(token, torch.tensor(pos))
        check.at_most("compilations", cc.compilations, 1)
        check.that(cc.graph_breaks == 0, "the step has a graph break", cc.describe())

    ``torch.compile`` traces a function the first time it is called and guards the
    trace with conditions on its inputs (shapes, dtypes, the values of Python
    numbers). A call that fails a guard is traced again. Code that cannot be traced
    (a Python ``if`` on a tensor's value) splits the function at a *graph break* into
    several separately compiled pieces. Both cost far more than the step they wrap.

    - ``cc.compilations``: frames Dynamo compiled (each piece of a broken function is
      one). A function compiled once, with no break, is 1.
    - ``cc.graph_breaks``: number of graph breaks; ``cc.break_reasons`` their first
      lines.
    - ``cc.graphs``: distinct graphs handed to the backend.
    - ``cc.describe()``: the above as text for a failure detail.

    Entering the block calls ``torch._dynamo.reset()``, so compilations made earlier
    in the process do not hide later ones, and the counters start at zero. Use
    ``backend="eager"`` on the CPU: Dynamo traces and guards exactly as it would for
    any backend, and no C++ compiler is needed.

    The numbers come from ``torch._dynamo.utils.counters``, which is not public API;
    they are what torch's own tests read. Checked on torch 2.15 (see
    PROBLEM_AUTHORING, "CUDA problems"); re-check when the pin moves.
    """

    def __init__(self) -> None:
        self.compilations = 0
        self.graphs = 0
        self.graph_breaks = 0
        self.break_reasons: list[str] = []

    def __enter__(self) -> Self:
        import torch._dynamo
        from torch._dynamo.utils import counters

        torch._dynamo.reset()
        counters.clear()
        self._counters = counters
        return self

    def _read(self) -> None:
        counters = self._counters
        self.compilations = int(counters["frames"]["ok"])
        self.graphs = int(counters["stats"]["unique_graphs"])
        reasons = counters["unimplemented"]
        self.graph_breaks = int(sum(reasons.values()))
        self.break_reasons = [str(reason).strip().splitlines()[0] for reason in reasons]

    def snapshot(self) -> Self:
        """Update the numbers without leaving the block; returns self."""
        self._read()
        return self

    def __exit__(self, *exc: object) -> None:
        import torch._dynamo

        self._read()
        # Compiled code holds guards on objects of this case; drop it before the next.
        torch._dynamo.reset()

    def describe(self) -> str:
        lines = [
            f"compilations: {self.compilations}\n",
            f"distinct graphs: {self.graphs}\n",
            f"graph breaks: {self.graph_breaks}\n",
        ]
        lines += [f"  break: {reason}\n" for reason in self.break_reasons]
        return "".join(lines)


# --------------------------------------------------------------------------------------
# TransferSpy: pinning and host-to-device transfers
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TransferEvent:
    """``kind`` is ``"pin"``, ``"transfer"``, or ``"mark"``.

    ``label`` is the label of the tensor the event is about (``spy.label``), or the
    name given to ``spy.mark``. For a transfer, ``pinned`` says whether the tensor
    came out of ``spy.pin`` and ``non_blocking`` what the caller passed.
    """

    kind: str
    label: Any
    pinned: bool = False
    non_blocking: bool = False


class TransferSpy:
    """Stand-ins for ``pin_memory`` and ``.to(device)`` that record the order of events.

        spy = TransferSpy()
        batches = [spy.label(torch.randn(4, 8), i) for i in range(5)]
        fetcher = ctx.solution.Prefetcher(batches, device, spy.to_device, spy.pin)
        for i, batch in enumerate(fetcher):
            spy.mark(("consumed", i))
        # transfer i + 1 was started before batch i was consumed:
        order = [(e.kind, e.label) for e in spy.events]

    A host-to-GPU copy with ``non_blocking=True`` returns at once and overlaps with
    the model's work. Torch documents that for *pinned* (page-locked) memory; from
    ordinary memory there is no such promise. So a prefetcher has to pin first,
    transfer with ``non_blocking=True``, and start the next transfer before handing
    out the current batch. None of that changes any value, so the test checks the
    sequence instead.

    - ``spy.label(tensor, label)`` names a source tensor and returns it. Tensors
      derived from it by ``pin`` and ``to_device`` carry the label.
    - ``spy.pin(tensor)`` returns a copy that counts as pinned.
    - ``spy.to_device(tensor, device, non_blocking=False)`` returns the tensor moved
      to ``device`` (a real ``.to``; a no-op on the CPU) and records whether its
      argument was pinned and whether the call was non-blocking.
    - ``spy.mark(name)`` records a point in time chosen by the test.
    - ``spy.events``: the ordered ``TransferEvent`` list; ``spy.unpinned`` and
      ``spy.blocking`` are the transfers made from pageable memory and without
      ``non_blocking=True``; ``spy.order()`` is ``[(kind, label), ...]``.

    On a GPU pass the real things: ``pin=torch.Tensor.pin_memory`` and
    ``to_device=lambda t, device, non_blocking=False: t.to(device,
    non_blocking=non_blocking)``.
    """

    def __init__(self) -> None:
        self.events: list[TransferEvent] = []
        self._labels: dict[int, Any] = {}
        self._pinned: set[int] = set()
        self._keep: list[Any] = []  # ids stay unique while the tensors are alive

    def _remember(self, tensor: Any, label: Any, pinned: bool = False) -> Any:
        self._keep.append(tensor)
        self._labels[id(tensor)] = label
        if pinned:
            self._pinned.add(id(tensor))
        return tensor

    def label(self, tensor: Any, label: Any) -> Any:
        return self._remember(tensor, label)

    def label_of(self, tensor: Any) -> Any:
        """The label ``tensor`` carries, or None."""
        return self._labels.get(id(tensor))

    def pin(self, tensor: Any) -> Any:
        label = self.label_of(tensor)
        self.events.append(TransferEvent("pin", label))
        return self._remember(tensor.clone(), label, pinned=True)

    def to_device(self, tensor: Any, device: Any, non_blocking: bool = False) -> Any:
        label = self.label_of(tensor)
        pinned = id(tensor) in self._pinned
        self.events.append(TransferEvent("transfer", label, pinned, bool(non_blocking)))
        # A fresh tensor object, as a transfer to another device would give.
        moved = tensor.to(device, non_blocking=non_blocking).view_as(tensor)
        return self._remember(moved, label)

    def mark(self, name: Any) -> None:
        self.events.append(TransferEvent("mark", name))

    @property
    def transfers(self) -> list[TransferEvent]:
        return [event for event in self.events if event.kind == "transfer"]

    @property
    def unpinned(self) -> list[TransferEvent]:
        """Transfers whose source was not pinned."""
        return [event for event in self.transfers if not event.pinned]

    @property
    def blocking(self) -> list[TransferEvent]:
        """Transfers made without ``non_blocking=True``."""
        return [event for event in self.transfers if not event.non_blocking]

    def order(self) -> list[tuple[str, Any]]:
        return [(event.kind, event.label) for event in self.events]

    def describe(self) -> str:
        lines = []
        for e in self.events:
            extra = ""
            if e.kind == "transfer":
                extra = f" (pinned={e.pinned}, non_blocking={e.non_blocking})"
            lines.append(f"{e.kind}: {e.label!r}{extra}\n")
        return "".join(lines)
