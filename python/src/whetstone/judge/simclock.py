"""Simulated-clock judge: a deterministic discrete-event simulator with virtual time.

``SimClock`` runs an asyncio event loop whose clock is virtual. Whenever nothing is
ready to run, time jumps straight to the next scheduled timer instead of sleeping, so
a simulated hour of traffic takes milliseconds and every run is exactly reproducible.

Solutions use ordinary asyncio (``asyncio.sleep``, ``asyncio.Queue``,
``asyncio.wait_for``, ``loop.call_later``, ``loop.time()``) or the clock object the
test hands them; either way they are driven by virtual time. What they must not do is
touch the real clock (``time.sleep``, ``time.monotonic``) or real I/O.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import selectors
from collections.abc import Awaitable, Callable, Coroutine, Iterable
from dataclasses import dataclass, field
from typing import Any, Self

from whetstone.judge.errors import CheckFailure


class _VirtualSelector(selectors.BaseSelector):
    """A selector that never waits: asked to block for ``timeout``, it advances time."""

    def __init__(self) -> None:
        self.loop: _VirtualLoop | None = None
        self._keys: dict[Any, selectors.SelectorKey] = {}

    def register(self, fileobj: Any, events: int, data: Any = None) -> selectors.SelectorKey:
        key = selectors.SelectorKey(fileobj, fileobj if isinstance(fileobj, int) else -1,
                                    events, data)  # fmt: skip
        self._keys[fileobj] = key
        return key

    def unregister(self, fileobj: Any) -> selectors.SelectorKey:
        return self._keys.pop(fileobj)

    def select(self, timeout: float | None = None) -> list:
        # asyncio calls this once per loop iteration with the time until the next
        # timer: 0 if callbacks are ready to run now, None if there are no timers.
        loop = self.loop
        assert loop is not None
        if timeout is None:
            # Nothing is ready and no timer is pending: the simulation is idle. A real
            # loop would block forever here; ours stops.
            loop.stop()
        elif timeout > 0:
            horizon = loop.horizon
            if horizon is not None and loop.virtual_time + timeout > horizon:
                # The next event lies beyond the requested end of the run.
                loop.virtual_time = max(loop.virtual_time, horizon)
                loop.stop()
            else:
                loop.advance_to_next_timer(timeout)
        return []

    def get_map(self) -> dict[Any, selectors.SelectorKey]:
        return self._keys


class _VirtualLoop(asyncio.SelectorEventLoop):
    def __init__(self) -> None:
        selector = _VirtualSelector()
        super().__init__(selector)
        selector.loop = self
        self.virtual_time = 0.0
        #: While set, the loop stops instead of advancing past this virtual time.
        self.horizon: float | None = None

    def time(self) -> float:
        return self.virtual_time

    def advance_to_next_timer(self, timeout: float) -> None:
        """Advance virtual time by ``timeout``, the wait asyncio computed for the next timer.

        ``timeout`` is ``deadline - now``; adding it back can miss the deadline by a
        rounding error, so land exactly on the deadline when that is what was meant.
        """
        target = self.virtual_time + timeout
        scheduled = getattr(self, "_scheduled", None)
        if scheduled:
            when = getattr(scheduled[0], "_when", None)
            if isinstance(when, (int, float)) and math.isclose(when, target, rel_tol=1e-12):
                target = float(when)
        self.virtual_time = max(self.virtual_time, target)


@dataclass(frozen=True)
class TraceEvent:
    time: float
    kind: str
    fields: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.fields[key]


class Trace:
    """An append-only log of what happened and when (in virtual seconds)."""

    def __init__(self, clock: SimClock) -> None:
        self._clock = clock
        self.events: list[TraceEvent] = []

    def record(self, kind: str, **fields: Any) -> TraceEvent:
        """Append an event stamped with the current virtual time."""
        event = TraceEvent(self._clock.now(), kind, fields)
        self.events.append(event)
        return event

    def of(self, kind: str) -> list[TraceEvent]:
        """All events of one kind, in order."""
        return [event for event in self.events if event.kind == kind]

    def values(self, kind: str, name: str) -> list[Any]:
        """Field ``name`` of every event of ``kind`` (for example all batch sizes)."""
        return [event.fields[name] for event in self.of(kind)]

    def times(self, kind: str) -> list[float]:
        return [event.time for event in self.of(kind)]

    def latencies(self, start: str, end: str, key: str = "id") -> list[float]:
        """Time from each ``start`` event to the ``end`` event with the same ``key`` field.

        Fails if a started item never ended (a dropped request).
        """
        ended = {event.fields[key]: event.time for event in self.of(end)}
        result = []
        for event in self.of(start):
            ident = event.fields[key]
            if ident not in ended:
                raise CheckFailure(
                    f"{start} {ident!r} never reached {end!r}",
                    f"{len(self.of(start))} {start} events, {len(ended)} {end} events\n",
                )
            result.append(ended[ident] - event.time)
        return result


def percentile(values: Iterable[float], q: float) -> float:
    """The ``q``-th percentile (0 to 100) by the nearest-rank method.

    Nearest rank returns an actual observation and involves no interpolation, so the
    result is exactly reproducible. p50 of [1, 2, 3, 4] is 2; p100 is the maximum.
    """
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of an empty sequence")
    if not 0 <= q <= 100:
        raise ValueError("q must be between 0 and 100")
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


class SimClock:
    """A virtual-time event loop plus a trace recorder.

        sim = ctx.simclock()
        sim.call_at(0.010, batcher.submit, "request-1")      # schedule an arrival
        sim.spawn(batcher.run())                             # start a coroutine
        sim.run()                                            # run until idle
        sim.now()                                            # virtual seconds elapsed

    ``sim.loop`` is the asyncio loop, for solutions that want it explicitly.
    """

    def __init__(self) -> None:
        self.loop = _VirtualLoop()
        self.trace = Trace(self)
        self._failure: BaseException | None = None
        self._closed = False
        self.loop.set_exception_handler(self._on_loop_exception)

    # -- time and scheduling -----------------------------------------------------

    def now(self) -> float:
        """Current virtual time in seconds; starts at 0."""
        return self.loop.time()

    def call_at(self, when: float, fn: Callable[..., Any], *args: Any) -> asyncio.TimerHandle:
        """Call ``fn(*args)`` at virtual time ``when``. Returns a handle with ``cancel()``."""
        return self.loop.call_at(when, fn, *args)

    def call_later(self, delay: float, fn: Callable[..., Any], *args: Any) -> asyncio.TimerHandle:
        """Call ``fn(*args)`` after ``delay`` virtual seconds."""
        return self.loop.call_later(delay, fn, *args)

    def sleep(self, delay: float) -> Awaitable[None]:
        """Awaitable that completes after ``delay`` virtual seconds."""
        return asyncio.sleep(delay)

    def spawn(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
        """Start a coroutine as a task. An exception escaping it fails ``run()``."""
        task = self.loop.create_task(coro)
        task.add_done_callback(self._on_task_done)
        return task

    def spawn_at(self, when: float, make: Callable[[], Coroutine[Any, Any, Any]]) -> None:
        """At virtual time ``when``, call ``make()`` and start the coroutine it returns."""
        self.loop.call_at(when, lambda: self.spawn(make()))

    # -- running -----------------------------------------------------------------

    def run(self, until: float | None = None) -> None:
        """Run the simulation until it is idle, or until virtual time ``until``.

        Idle means no callback is ready and no timer is pending; tasks blocked forever
        on a queue do not keep the simulation alive. With ``until``, everything
        scheduled at or before ``until`` runs (including whatever it triggers at that
        same instant), and time ends at exactly ``until`` even if the simulation went
        idle earlier.
        """
        self._check_open()
        if until is not None and until < self.now():
            raise ValueError("cannot run backwards in time")
        self.loop.horizon = until
        try:
            self.loop.run_forever()
        finally:
            self.loop.horizon = None
        self._raise_failure()
        if until is not None:
            self.loop.virtual_time = max(self.loop.virtual_time, until)

    def run_until_complete(self, awaitable: Awaitable[Any]) -> Any:
        """Run until ``awaitable`` finishes and return its result.

        Fails with a clear message if the simulation goes idle first (a deadlock:
        the awaitable waits for something that will never happen).
        """
        self._check_open()
        future = asyncio.ensure_future(awaitable, loop=self.loop)
        future.add_done_callback(lambda _: self.loop.stop())
        self.loop.run_forever()
        if not future.done():
            future.cancel()
            self._raise_failure()
            raise CheckFailure(
                "the simulation went idle before the awaited operation finished "
                f"(deadlock at virtual time {self.now():g} s)",
                "Nothing was scheduled to run, yet the operation was still waiting.\n",
            )
        self._raise_failure()
        return future.result()

    def close(self) -> None:
        """Cancel whatever is still pending and close the loop. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        try:
            # Best effort: a task that refuses to be canceled must not block closing.
            with contextlib.suppress(Exception):
                pending = [task for task in asyncio.all_tasks(self.loop) if not task.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        finally:
            self.loop.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- failure propagation -------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("this SimClock is closed")

    def _on_task_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None and self._failure is None:
            self._failure = exc
            self.loop.stop()

    def _on_loop_exception(self, loop: asyncio.AbstractEventLoop, context: dict) -> None:
        exc = context.get("exception")
        if self._failure is None:
            self._failure = exc or RuntimeError(str(context.get("message")))
            loop.stop()

    def _raise_failure(self) -> None:
        if self._failure is not None:
            failure, self._failure = self._failure, None
            raise failure
