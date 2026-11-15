import asyncio
from collections import deque


class Unavailable(Exception):
    """The server is draining or has shut down and takes no new requests."""


class Terminated(Exception):
    """The request was cut off at the grace deadline of a drain."""

    def __init__(self, tokens: int):
        super().__init__(f"terminated after {tokens} tokens")
        self.tokens = tokens  # output tokens the request had produced


class Handle:
    """What `Server.submit` returns: the client's side of one request."""

    def __init__(self, request, future):
        self.request = request
        self._future = future  # resolves to a token count or to a Terminated instance
        self._tokens = 0

    async def result(self) -> int:
        outcome = await self._future
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class Server:
    def __init__(self, engine, max_running: int):
        self.engine = engine
        self.max_running = max_running
        self.ready = True
        self.closed = False
        self._waiting = deque()  # handles without a slot, oldest first
        self._running = []  # handles of live sequences, in admission order
        self._wake = asyncio.Event()
        self._drained = None  # the future `drain` returned, once it has been called
        self._deadline = None  # the timer of the grace deadline
        self._terminated = 0

    def submit(self, request) -> Handle:
        if not self.ready:
            raise Unavailable("the server is draining")
        handle = Handle(request, asyncio.get_running_loop().create_future())
        self._waiting.append(handle)
        self._wake.set()
        return handle

    def drain(self, grace_ms: float) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        self.ready = False  # before this call returns: the next submit is refused
        self._drained = loop.create_future()
        if grace_ms <= 0:
            # A timer for "now" would fire after a step that ends at this instant.
            self._terminate_pending()
        else:
            # A plain timer: at a shared instant it fires before the engine completes
            # a step, so a token due at the deadline is not produced.
            self._deadline = loop.call_at(loop.time() + grace_ms, self._terminate_pending)
        self._wake.set()  # an idle loop has to notice that it may stop
        return self._drained

    def _terminate_pending(self) -> None:
        """The grace period is over: end everything that is still pending, now."""
        for handle in self._running:
            self.engine.release(handle.request.id)  # also in the middle of a step
        pending = [*self._running, *self._waiting]
        self._running.clear()
        self._waiting.clear()
        for handle in pending:
            handle._future.set_result(Terminated(handle._tokens))
        self._terminated += len(pending)
        self._wake.set()

    async def run(self) -> None:
        while True:
            if not self._waiting and not self._running:
                if self._drained is not None:
                    # Nothing pending, and the engine is idle because this loop is
                    # the only caller of `step` and is not inside one.
                    if self._deadline is not None:
                        self._deadline.cancel()
                    self.closed = True
                    self._drained.set_result(self._terminated)
                    return
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

            # Sequences terminated during the step have left `self._running` and are
            # in neither `result.tokens` nor `result.finished`.
            by_id = {handle.request.id: handle for handle in self._running}
            for seq_id in result.tokens:
                by_id[seq_id]._tokens += 1
            for seq_id in result.finished:
                handle = by_id[seq_id]
                self._running.remove(handle)
                handle._future.set_result(handle._tokens)
