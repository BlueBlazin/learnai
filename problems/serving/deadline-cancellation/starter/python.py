import asyncio
from collections import deque


class Canceled(Exception):
    """The request was canceled by its client before it finished."""


class DeadlineExceeded(Exception):
    """The request had not finished when its deadline came."""


class Handle:
    """What `Scheduler.submit` returns: the client's side of one request."""

    def __init__(self, scheduler, request, future):
        self.request = request
        self._scheduler = scheduler
        self._future = future
        self._tokens = 0  # output tokens produced so far

    def cancel(self) -> bool:
        """Cancel the request now. Takes effect before this call returns.

        Returns True if the request was still pending (waiting or running): it is
        removed from the scheduler, released from the engine if it was running, and
        `result()` raises `Canceled`. Returns False, and changes nothing, if the
        request had already finished, expired, or been canceled.
        """
        raise NotImplementedError

    async def result(self) -> int:
        """Wait for the request to end.

        Returns the number of output tokens if it finished. Raises `Canceled` if it
        was canceled and `DeadlineExceeded` if its deadline came first.
        """
        return await self._future


class Scheduler:
    """A continuous-batching scheduler. As given, it serves every request to the end,
    first come first served; cancellation and deadlines are yours to add."""

    def __init__(self, engine, max_running: int):
        self.engine = engine
        self.max_running = max_running
        self._waiting = deque()  # handles without a slot, oldest first
        self._running = []  # handles of live sequences, in admission order
        self._wake = asyncio.Event()

    def submit(self, request) -> Handle:
        """Accept a request at its arrival instant and return its handle at once.

        request: a `SimRequest`. `request.deadline_s` is None or an absolute virtual
            time in milliseconds: the request expires at that instant unless it has
            finished before. A deadline that is not in the future expires it here.

        Called from a plain callback inside the event loop, so
        `asyncio.get_running_loop()` works: `loop.time()` is the virtual time in
        milliseconds and `loop.call_at(when, fn, *args)` schedules a timer.
        """
        loop = asyncio.get_running_loop()
        handle = Handle(self, request, loop.create_future())
        self._waiting.append(handle)
        self._wake.set()
        return handle

    async def run(self) -> None:
        """The scheduling loop; it never returns."""
        while True:
            if not self._waiting and not self._running:
                self._wake.clear()
                await self._wake.wait()  # idle: block until `submit` wakes the loop
                continue

            decode = [handle.request.id for handle in self._running]
            prefill = []
            while self._waiting and len(self._running) < self.max_running:
                handle = self._waiting.popleft()
                self.engine.admit(handle.request)
                prefill.append((handle.request.id, handle.request.prompt_tokens))
                self._running.append(handle)

            result = await self.engine.step(prefill=prefill, decode=decode)

            by_id = {handle.request.id: handle for handle in self._running}
            for seq_id in result.tokens:
                by_id[seq_id]._tokens += 1
            for seq_id in result.finished:
                handle = by_id[seq_id]
                self._running.remove(handle)
                handle._future.set_result(handle._tokens)
