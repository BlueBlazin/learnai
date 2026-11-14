# BUG: submit does not check whether the deadline has already passed; the request joins the waiting list and is admitted and prefilled before the overdue timer removes it.
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
        self._future = future  # resolves to a token count or to an exception instance
        self._timer = None  # the deadline timer, while the request is pending
        self._tokens = 0

    def cancel(self) -> bool:
        return self._scheduler._end(self, Canceled(f"request {self.request.id!r} was canceled"))

    async def result(self) -> int:
        outcome = await self._future
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class Scheduler:
    def __init__(self, engine, max_running: int):
        self.engine = engine
        self.max_running = max_running
        self._waiting = deque()  # handles without a slot, oldest first
        self._running = []  # handles of live sequences, in admission order
        self._wake = asyncio.Event()

    def submit(self, request) -> Handle:
        loop = asyncio.get_running_loop()
        handle = Handle(self, request, loop.create_future())
        deadline = request.deadline_s
        if deadline is not None:
            # A plain timer, not a sleeping task: at a shared instant it fires before
            # the engine completes a step, so a token due at the deadline is not produced.
            expired = DeadlineExceeded(f"request {request.id!r} missed its deadline")
            handle._timer = loop.call_at(deadline, self._end, handle, expired)
        self._waiting.append(handle)
        self._wake.set()
        return handle

    def _end(self, handle: Handle, outcome) -> bool:
        """Finish, cancel, or expire `handle` right now. False if it had already ended."""
        if handle._future.done():
            return False
        if handle._timer is not None:
            handle._timer.cancel()
        if handle in self._running:
            self._running.remove(handle)  # the slot is free for the next step
            if isinstance(outcome, Exception):
                # Also in the middle of a step: the sequence gets nothing from it.
                self.engine.release(handle.request.id)
        else:
            self._waiting.remove(handle)
        handle._future.set_result(outcome)
        return True

    async def run(self) -> None:
        while True:
            if not self._waiting and not self._running:
                self._wake.clear()
                await self._wake.wait()
                continue

            decode = [handle.request.id for handle in self._running]
            prefill = []
            while self._waiting and len(self._running) < self.max_running:
                handle = self._waiting.popleft()
                self.engine.admit(handle.request)
                prefill.append((handle.request.id, handle.request.prompt_tokens))
                self._running.append(handle)

            result = await self.engine.step(prefill=prefill, decode=decode)

            # Anything canceled or expired during the step has already left
            # `self._running` and is in neither `result.tokens` nor `result.finished`.
            by_id = {handle.request.id: handle for handle in self._running}
            for seq_id in result.tokens:
                by_id[seq_id]._tokens += 1
            for seq_id in result.finished:
                self._end(by_id[seq_id], by_id[seq_id]._tokens)
