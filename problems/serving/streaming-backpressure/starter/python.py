import asyncio
from collections import deque


class Stream:
    """What `Server.submit` returns: the client's side of one streamed request."""

    def __init__(self, server, request):
        self.request = request
        self.state = "open"  # "open", "finished", "stalled", or "closed"
        self._server = server
        self._buffer = []  # token indices produced and not yet taken

    def take(self) -> list[int]:
        """Remove and return every buffered token index, oldest first, as one chunk.

        Never blocks; returns [] when nothing is buffered. Called by the transport
        each time the client can receive data.
        """
        chunk, self._buffer = self._buffer, []
        return chunk

    def close(self) -> None:
        """The client disconnected. Takes effect before this call returns: an open
        stream leaves the server (and the engine, if it was running) and its state
        becomes "closed". A stream that is not open is left unchanged."""
        raise NotImplementedError


class Server:
    """A continuous-batching server that streams tokens into per-request buffers.
    As given, the buffers are unbounded and nobody is ever paused, cut off, or
    closed; backpressure is yours to add."""

    def __init__(self, engine, max_running: int, max_buffered: int, stall_ms: float):
        self.engine = engine
        self.max_running = max_running
        self.max_buffered = max_buffered  # tokens a stream may hold unread
        self.stall_ms = stall_ms  # how long a buffer may stay full
        self._waiting = deque()  # streams without a slot, oldest first
        self._running = []  # streams of live sequences, in admission order
        self._wake = asyncio.Event()

    def submit(self, request) -> Stream:
        """Accept a request at its arrival instant and return its stream at once.

        Called from a plain callback inside the event loop, as are `Stream.take` and
        `Stream.close`: `asyncio.get_running_loop()` works there, `loop.time()` is the
        virtual time in milliseconds, and `loop.call_at(when, fn, *args)` schedules a
        timer.
        """
        stream = Stream(self, request)
        self._waiting.append(stream)
        self._wake.set()
        return stream

    async def run(self) -> None:
        """The serving loop; it never returns."""
        produced = {}  # request id -> tokens produced so far
        while True:
            if not self._waiting and not self._running:
                self._wake.clear()
                await self._wake.wait()  # idle: block until `submit` wakes the loop
                continue

            decode = [stream.request.id for stream in self._running]
            prefill = []
            while self._waiting and len(self._running) < self.max_running:
                stream = self._waiting.popleft()
                self.engine.admit(stream.request)
                prefill.append((stream.request.id, stream.request.prompt_tokens))
                self._running.append(stream)

            result = await self.engine.step(prefill=prefill, decode=decode)

            by_id = {stream.request.id: stream for stream in self._running}
            for seq_id in result.tokens:
                index = produced.get(seq_id, 0)
                produced[seq_id] = index + 1
                by_id[seq_id]._buffer.append(index)
            for seq_id in result.finished:
                stream = by_id[seq_id]
                self._running.remove(stream)
                stream.state = "finished"
