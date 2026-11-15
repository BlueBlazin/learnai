# BUG: a sequence whose buffer is full keeps decoding and the new tokens are thrown away, so a slow client receives a stream with holes.
import asyncio
from collections import deque


class Stream:
    """What `Server.submit` returns: the client's side of one streamed request."""

    def __init__(self, server, request):
        self.request = request
        self.state = "open"  # "open", "finished", "stalled", or "closed"
        self._server = server
        self._buffer = []  # token indices produced and not yet taken
        self._produced = 0
        self._stall_timer = None  # armed while the buffer is full

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
        self._waiting.append(stream)
        self._wake.set()
        return stream

    def _take(self, stream: Stream) -> list[int]:
        chunk, stream._buffer = stream._buffer, []  # everything buffered, as one chunk
        if stream._stall_timer is not None:
            stream._stall_timer.cancel()  # the client is reading again
            stream._stall_timer = None
        self._wake.set()  # a paused sequence can be decoded again
        return chunk

    def _end(self, stream: Stream, state: str) -> None:
        """The client disconnected ("closed") or stopped reading ("stalled")."""
        if stream.state != "open":
            return
        stream.state = state
        stream._buffer = []
        if stream._stall_timer is not None:
            stream._stall_timer.cancel()
            stream._stall_timer = None
        if stream in self._running:
            self._running.remove(stream)
            self.engine.release(stream.request.id)  # also in the middle of a step
        else:
            self._waiting.remove(stream)
        self._wake.set()

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            # A sequence whose buffer is full is paused: it keeps its slot and its KV
            # cache but is left out of the step, so the buffer cannot grow.
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
                stream = by_id[seq_id]
                if len(stream._buffer) < self.max_buffered:
                    stream._buffer.append(stream._produced)
                stream._produced += 1
            for seq_id in result.finished:
                stream = by_id[seq_id]
                self._running.remove(stream)
                stream.state = "finished"
            for stream in self._running:
                if len(stream._buffer) >= self.max_buffered and stream._stall_timer is None:
                    stream._stall_timer = loop.call_at(
                        loop.time() + self.stall_ms, self._end, stream, "stalled"
                    )
