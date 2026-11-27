## The idea

Backpressure ensures that when a consumer falls behind a producer, the producer slows down instead of letting unread data pile up. In a token-streaming server, the excess tokens can be absorbed in one of three places:

| where excess goes | what it costs | who pays |
|---|---|---|
| an unbounded buffer | memory and GPU time for unread tokens | the server |
| the engine loop (`await send`) | every request waits for the slowest client | all other clients |
| the slow request itself | it generates no faster than the client reads | the slow client |

Only the third option isolates slow clients from the server and from other users, and a bounded per-stream buffer implements it. The buffer absorbs normal network jitter by coalescing a few tokens into one chunk. Once the buffer is full, the *sequence* pauses rather than the engine loop. The loop never awaits a client and only checks the length of each buffer.

This design introduces three requirements:

- **Pausing is a planning decision.** Before each step, include a running sequence in `decode` only if `len(buffer) < max_buffered`. Because a sequence produces at most one token per step, a buffer below the limit at planning time can at most reach `max_buffered` after the step. No overflow check is needed when appending, and tokens are never dropped.
- **The loop can run out of work while sequences are live.** If every running sequence is paused and no new request can be admitted, there is no step to run. Running an empty step raises a `ProtocolViolation`, and polling wastes time and resumes late. Instead, the loop waits on an `asyncio.Event` that is set by every operation that can create work: `submit`, `take`, `close`, and a stall timeout.
- **A paused sequence still holds resources.** It occupies an engine slot and its KV cache. To prevent an abandoned client from holding that slot forever, filling a buffer starts a timer for `stall_ms`. If the timer expires before a `take()`, the server releases the sequence immediately, using the same cleanup path as a client disconnect (`close()`).

## Reference solution (Python)

```python
class Stream:
    def __init__(self, server, request):
        self.request = request
        self.state = "open"
        self._server = server
        self._buffer = []
        self._produced = 0
        self._stall_timer = None

    def take(self):
        return self._server._take(self)

    def close(self):
        self._server._end(self, "closed")


class Server:
    def _take(self, stream):
        chunk, stream._buffer = stream._buffer, []
        if stream._stall_timer is not None:
            stream._stall_timer.cancel()
            stream._stall_timer = None
        self._wake.set()
        return chunk

    def _end(self, stream, state):
        if stream.state != "open":
            return
        stream.state = state
        stream._buffer = []
        if stream._stall_timer is not None:
            stream._stall_timer.cancel()
            stream._stall_timer = None
        if stream in self._running:
            self._running.remove(stream)
            self.engine.release(stream.request.id)  # also mid-step
        else:
            self._waiting.remove(stream)
        self._wake.set()

    async def run(self):
        loop = asyncio.get_running_loop()
        while True:
            decode = [s.request.id for s in self._running if len(s._buffer) < self.max_buffered]
            prefill = []
            while self._waiting and len(self._running) < self.max_running:
                stream = self._waiting.popleft()
                self.engine.admit(stream.request)
                prefill.append((stream.request.id, stream.request.prompt_tokens))
                self._running.append(stream)
            if not prefill and not decode:
                self._wake.clear()
                await self._wake.wait()
                continue
            result = await self.engine.step(prefill=prefill, decode=decode)
            by_id = {s.request.id: s for s in self._running}
            for seq_id in result.tokens:
                stream = by_id[seq_id]
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
```

Clearing `_wake` and immediately calling `await self._wake.wait()` cannot miss a wake-up because there is no `await` between planning and `clear()`, so no callback can run in between.

## Common mistakes

- **Calling `await queue.put(token)` in the engine loop.** A bounded `asyncio.Queue` applies backpressure to the shared engine loop instead of the individual sequence. In a batch of six running requests, one stalled client freezes the other five.
- **Leaving buffers unbounded.** Never pausing a sequence produces the right token order but keeps generating for a client that is not reading. In the peak-buffer test, this reaches 24 unread tokens against a limit of 8, and without a stall timeout it would buffer the entire response.
- **Dropping tokens when the buffer is full.** Dropping bounds memory, but the client receives a corrupted stream with missing tokens (such as `[0, 1, 2]` followed by `[7, 8, 9]`).
- **Pausing one token late (`<=` instead of `<`).** The check runs before a step that appends a token, so `len(s._buffer) <= self.max_buffered` lets the buffer reach `max_buffered + 1`.
- **Marking a stream `"stalled"` without calling `engine.release`.** The sequence stays paused in the engine and leaks its slot permanently. After `max_running` stalled clients, the server can no longer admit new requests.
- **Not propagating `close()` to the engine.** Flipping `stream.state` without releasing the sequence lets it keep generating until the buffer fills and then hold its slot until it stalls.
- **Leaving a stale stall timer active.** If `take()` does not cancel `_stall_timer`, a timer armed during an earlier full buffer can fire later on a client that is actively reading.
- **Forgetting to wake the loop in `take()`, or polling with `sleep`.** When every running sequence is paused, a client read is the only event that can resume generation. Without `_wake.set()`, the loop hangs until another request arrives. Polling with `sleep` wakes up late.

## Complexity

Planning takes $O(R)$ time per step for $R$ running sequences. `take()` takes $O(1)$ time plus the size of the returned chunk. Each stream buffers at most `max_buffered` tokens in memory, and the engine generates at most `max_buffered` tokens ahead of what the client has read.

## In the wild

- **vLLM** (`vllm/v1/engine/output_processor.py`) places a `RequestOutputCollector` between its output loop and each request's generator. Its docstring states the same coalescing rule used here: "When streaming deltas, RequestOutputs are merged if the producer gets ahead of the consumer." The collector holds a single merged slot, `put` never blocks the engine loop, and the consumer waits on an `asyncio.Event`. vLLM does not pause generation on a full buffer. Instead, per-request token limits bound the merged output, and disconnect detection stops abandoned requests just like `close()`.
- **ASGI and asyncio transports** expose backpressure on the socket side. `await send(...)` and `StreamWriter.drain()` return only when the transport's write buffer has room. A per-request handler task that awaits them applies backpressure cleanly, whereas awaiting them in a shared engine loop causes head-of-line blocking. `sse-token-stream` covers the per-request side, including detecting disconnects by watching `receive()`.
- **Reactive Streams, gRPC, and HTTP/2 flow control** express this bound as credit, where the consumer grants the producer permission to send $n$ more items. Here, `max_buffered - len(buffer)` is that credit, and `take()` refills it.
- Pausing a sequence keeps its KV cache allocated. Under memory pressure, production servers may preempt paused sequences (freeing their blocks and recomputing later, as in `preemption-recompute`) or shorten `stall_ms`. The buffer bound provides the signal to make that decision.
