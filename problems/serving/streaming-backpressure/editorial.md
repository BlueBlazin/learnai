## The idea

Backpressure means that when the consumer is slower than the producer, the producer
finds out and slows down, instead of the difference piling up in between. A token
stream has three places where the difference can go:

| where | what it costs | who pays |
|---|---|---|
| an unbounded buffer | memory, and GPU time for tokens nobody has read | the server |
| the engine loop (`await send`) | every request waits for the slowest client | all other clients |
| the slow request itself | it generates no faster than it is read | the slow client |

Only the third is fair, and a bounded buffer is how to get it. The buffer absorbs
the jitter of a healthy client (a chunk of a few tokens instead of one), and when it
is full the *sequence* stops, not the loop: it is left out of the next step. The
engine loop never touches a client; it only reads the length of a list.

Three consequences have to be handled.

**Pausing is a planning decision.** Whether a sequence takes part in a step is
decided when the step is planned, from `len(buffer) < max_buffered`. A sequence gets
at most one token per step, so a buffer that was below the limit at planning time
is at most at the limit afterwards. No check is needed when appending, and nothing
is ever dropped.

**The loop can run out of work while sequences are live.** If every running
sequence is paused and nobody can be admitted, there is no step to run. An empty
step is a protocol violation and polling wastes time and resumes late, so the loop
waits on an event. Everything that can create work sets it: a new request, a
`take()` that empties a buffer, a `close()` or a stall that frees a slot.

**A paused sequence is not free.** It holds a slot and its KV cache. A client that
never comes back would hold them forever, so a full buffer starts a clock, and when
it reaches `stall_ms` the request is treated exactly like a disconnect: released
from the engine at that instant. Disconnects themselves must reach the engine too;
otherwise the buffer bound only limits the damage of a client that is gone.

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

Clearing the event and then waiting cannot lose a wake-up: between planning (which
found nothing to do) and `clear()` there is no `await`, so no callback can run in
between.

## Common mistakes

- **`await queue.put(token)` in the engine loop.** A bounded `asyncio.Queue` looks
  like backpressure in one line, and it is, on the wrong party: the loop that serves
  everyone blocks on one client. In the test with six running requests, one stalled
  client freezes the other five.
- **An unbounded buffer.** The starter's behavior: nobody is ever paused. The output
  is correct, and the server keeps generating for a client that reads nothing until
  something else stops it; in the cost case the peak is 24 unread tokens against a
  limit of 8, and without a stall limit it would be the whole answer.
- **Dropping tokens when the buffer is full.** Bounded memory, and a client that
  reads `[0, 1, 2]` and then `[7, 8, 9]`. Text is not telemetry: nothing may be
  skipped.
- **Pausing one token late** (`<=` for `<`). The check runs before a step that adds
  a token, so "not above the limit" lets the buffer reach `max_buffered + 1`.
- **Marking a stalled stream without releasing it.** The client is told, the
  sequence stays paused in the engine, and its slot is gone for good; after
  `max_running` stalled clients the server admits nothing.
- **Not propagating a disconnect.** `close()` that only flips the state leaves the
  sequence generating into the void until the buffer bound pauses it, and then
  holding its slot.
- **A stale stall timer.** If `take()` does not cancel the timer, the timer of an
  earlier full buffer fires on a client that has been reading since.
- **A `take()` that does not wake the loop,** or a loop that polls. When every
  running sequence is paused, the only event that can restart the engine is the
  client's read. Without the wake-up the server hangs until an unrelated request
  arrives; with polling it restarts late.

## Complexity

Planning is $O(R)$ per step for $R$ running sequences, as before. `take` is $O(1)$
plus the size of the chunk it returns. Memory per stream is at most `max_buffered`
tokens, and the engine generates at most `max_buffered` tokens beyond what a client
has read.

## In the wild

- **vLLM** (`vllm/v1/engine/output_processor.py`) puts a `RequestOutputCollector`
  between its output loop and each request's generator. Its docstring states the
  coalescing rule of this problem: "When streaming deltas, RequestOutputs are merged
  if the producer gets ahead of the consumer." The collector is a single slot that
  merges, `put` never blocks the loop, and the consumer waits on an `asyncio.Event`.
  It does not pause generation: what bounds the merged output there is the request's
  own token limit, and what stops a request whose client has gone is disconnect
  detection, the `close()` of this problem.
- **ASGI and asyncio transports** expose the same signal from the other side: `await
  send(...)` (and `StreamWriter.drain()`) returns only when the transport's write
  buffer has room. A handler that awaits it per request, in its own task, is
  backpressured correctly; a shared loop that awaits it is the head-of-line bug.
  `sse-token-stream` covers the per-request side, including the fact that a
  disconnect has to be detected by watching `receive()`.
- **Reactive Streams** (and gRPC and HTTP/2 flow control) make the bound explicit as
  credit: the consumer grants the producer permission for $n$ more items.
  `max_buffered` minus the buffer's length is that credit, and `take()` refills it.
- Pausing has a price: a paused sequence keeps its KV cache. Servers under memory
  pressure may prefer to preempt a paused sequence (free its blocks and recompute
  later, `preemption-recompute`) or to shorten `stall_ms`; the buffer bound is what
  gives them the signal to act on.
