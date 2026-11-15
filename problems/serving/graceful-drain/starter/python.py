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
        self._future = future
        self._tokens = 0  # output tokens produced so far

    async def result(self) -> int:
        """Wait for the request to end.

        Returns the number of output tokens if it finished. Raises `Terminated`, with
        `.tokens` set to the number of tokens produced, if a drain cut it off.
        """
        return await self._future


class Server:
    """A continuous-batching server. As given, it serves every request to the end,
    first come first served, and never stops; draining is yours to add."""

    def __init__(self, engine, max_running: int):
        self.engine = engine
        self.max_running = max_running
        self.ready = True  # the readiness probe: False from the start of a drain
        self.closed = False  # True once a drain has completed
        self._waiting = deque()  # handles without a slot, oldest first
        self._running = []  # handles of live sequences, in admission order
        self._wake = asyncio.Event()

    def submit(self, request) -> Handle:
        """Accept a request at its arrival instant and return its handle at once.

        Raises `Unavailable` if a drain has started; the request is then not accepted.
        Called from a plain callback inside the event loop.
        """
        handle = Handle(request, asyncio.get_running_loop().create_future())
        self._waiting.append(handle)
        self._wake.set()
        return handle

    def drain(self, grace_ms: float) -> asyncio.Future:
        """Start shutting down. A plain method: it takes effect before it returns.

        grace_ms: how long pending requests may keep running, in virtual
            milliseconds; 0 terminates them inside this call.

        Returns a future that resolves, once nothing is pending and the engine is
        idle, to the number of requests that were terminated. Called from a plain
        callback inside the event loop, so `asyncio.get_running_loop()` works:
        `loop.time()` is the virtual time in milliseconds and
        `loop.call_at(when, fn, *args)` schedules a timer.
        """
        raise NotImplementedError

    async def run(self) -> None:
        """The serving loop. Returns when a drain has completed."""
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
