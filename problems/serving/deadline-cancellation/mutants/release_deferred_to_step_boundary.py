# BUG: a canceled or expired running request is only marked; engine.release is called when the loop plans its next step, so the step that was in flight still produces a token for it after the cancellation instant.
import asyncio
from collections import deque


class Canceled(Exception):
    """The request was canceled by its client before it finished."""


class DeadlineExceeded(Exception):
    """The request had not finished when its deadline came."""


class Handle:
    def __init__(self, scheduler, request, future):
        self.request = request
        self._scheduler = scheduler
        self._future = future
        self._timer = None
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
        self._waiting = deque()
        self._running = []
        self._to_release = []  # ids to drop from the engine before the next step
        self._wake = asyncio.Event()

    def submit(self, request) -> Handle:
        loop = asyncio.get_running_loop()
        handle = Handle(self, request, loop.create_future())
        deadline = request.deadline_s
        if deadline is not None and deadline <= loop.time():
            handle._future.set_result(DeadlineExceeded(f"request {request.id!r} arrived too late"))
            return handle
        if deadline is not None:
            expired = DeadlineExceeded(f"request {request.id!r} missed its deadline")
            handle._timer = loop.call_at(deadline, self._end, handle, expired)
        self._waiting.append(handle)
        self._wake.set()
        return handle

    def _end(self, handle: Handle, outcome) -> bool:
        if handle._future.done():
            return False
        if handle._timer is not None:
            handle._timer.cancel()
        if handle in self._running:
            self._running.remove(handle)
            if isinstance(outcome, Exception):
                self._to_release.append(handle.request.id)
                self._wake.set()
        else:
            self._waiting.remove(handle)
        handle._future.set_result(outcome)
        return True

    async def run(self) -> None:
        while True:
            for seq_id in self._to_release:
                if seq_id in self.engine.live:
                    self.engine.release(seq_id)
            self._to_release = []
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

            by_id = {handle.request.id: handle for handle in self._running}
            for seq_id in result.tokens:
                if seq_id in by_id:
                    by_id[seq_id]._tokens += 1
            for seq_id in result.finished:
                if seq_id in by_id:
                    self._end(by_id[seq_id], by_id[seq_id]._tokens)
