# BUG: deadlines are checked only when a waiting request is about to be admitted; there is no timer, so a running request generates past its deadline and a waiting one is told late.
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
        self._wake = asyncio.Event()

    def submit(self, request) -> Handle:
        loop = asyncio.get_running_loop()
        handle = Handle(self, request, loop.create_future())
        self._waiting.append(handle)
        self._wake.set()
        return handle

    def _end(self, handle: Handle, outcome) -> bool:
        if handle._future.done():
            return False
        if handle in self._running:
            self._running.remove(handle)
            if isinstance(outcome, Exception):
                self.engine.release(handle.request.id)
        else:
            self._waiting.remove(handle)
        handle._future.set_result(outcome)
        return True

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            if not self._waiting and not self._running:
                self._wake.clear()
                await self._wake.wait()
                continue

            decode = [handle.request.id for handle in self._running]
            prefill = []
            while self._waiting and len(self._running) < self.max_running:
                handle = self._waiting[0]
                deadline = handle.request.deadline_s
                if deadline is not None and deadline <= loop.time():
                    self._end(handle, DeadlineExceeded(f"request {handle.request.id!r} expired"))
                    continue
                self._waiting.popleft()
                self.engine.admit(handle.request)
                prefill.append((handle.request.id, handle.request.prompt_tokens))
                self._running.append(handle)
            if not prefill and not decode:
                continue

            result = await self.engine.step(prefill=prefill, decode=decode)

            by_id = {handle.request.id: handle for handle in self._running}
            for seq_id in result.tokens:
                by_id[seq_id]._tokens += 1
            for seq_id in result.finished:
                self._end(by_id[seq_id], by_id[seq_id]._tokens)
