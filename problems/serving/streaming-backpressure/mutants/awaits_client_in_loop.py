# BUG: each stream has a bounded asyncio.Queue and the engine loop does `await queue.put(token)`; when one client's queue is full the whole loop waits for that client, and every other request stops with it.
import asyncio
from collections import deque


class Stream:
    """What `Server.submit` returns: the client's side of one streamed request."""

    def __init__(self, server, request):
        self.request = request
        self.state = "open"  # "open", "finished", "stalled", or "closed"
        self._server = server
        self._queue = None  # asyncio.Queue(maxsize=max_buffered), created by the server
        self._produced = 0

    def take(self) -> list[int]:
        return self._server._take(self)

    def close(self) -> None:
        self._server._end(self, "closed")


class Server:
    def __init__(self, engine, max_running: int, max_buffered: int, stall_ms: float):
        self.engine = engine
        self.max_running = max_running
        self.max_buffered = max_buffered
        self.stall_ms = stall_ms
        self._waiting = deque()  # streams without a slot, oldest first
        self._running = []  # streams of live sequences, in admission order
        self._wake = asyncio.Event()

    def submit(self, request) -> Stream:
        stream = Stream(self, request)
        stream._queue = asyncio.Queue(maxsize=self.max_buffered)
        self._waiting.append(stream)
        self._wake.set()
        return stream

    def _take(self, stream: Stream) -> list[int]:
        chunk = []
        while not stream._queue.empty():
            chunk.append(stream._queue.get_nowait())  # unblocks a pending put
        return chunk

    def _end(self, stream: Stream, state: str) -> None:
        """The client disconnected ("closed") or stopped reading ("stalled")."""
        if stream.state != "open":
            return
        stream.state = state
        stream._queue = asyncio.Queue(maxsize=self.max_buffered)
        if stream in self._running:
            self._running.remove(stream)
            self.engine.release(stream.request.id)  # also in the middle of a step
        else:
            self._waiting.remove(stream)
        self._wake.set()

    async def run(self) -> None:
        while True:
            decode = [stream.request.id for stream in self._running]
            prefill = []
            while self._waiting and len(self._running) < self.max_running:
                stream = self._waiting.popleft()
                self.engine.admit(stream.request)
                prefill.append((stream.request.id, stream.request.prompt_tokens))
                self._running.append(stream)

            if not prefill and not decode:
                # Idle, or every running sequence is paused. `submit`, `take`,
                # `close`, and the stall timers set the event; the loop never waits
                # for one particular client.
                self._wake.clear()
                await self._wake.wait()
                continue

            result = await self.engine.step(prefill=prefill, decode=decode)

            # Streams closed or stalled during the step have left `self._running`
            # and are in neither `result.tokens` nor `result.finished`.
            by_id = {stream.request.id: stream for stream in self._running}
            for seq_id in result.tokens:
                stream = by_id.get(seq_id)
                if stream is None or stream.state != "open":
                    continue
                try:
                    # Blocks while the client's queue is full: the whole loop waits.
                    await asyncio.wait_for(stream._queue.put(stream._produced), self.stall_ms)
                    stream._produced += 1
                except TimeoutError:
                    self._end(stream, "stalled")
            for seq_id in result.finished:
                stream = by_id[seq_id]
                if stream in self._running:
                    self._running.remove(stream)
                    stream.state = "finished"
