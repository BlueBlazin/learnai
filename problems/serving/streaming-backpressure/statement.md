Put a bounded buffer between the engine loop and each client, so that a client that
reads slowly, stops reading, or disconnects costs the server a known, small amount
and costs the other requests nothing. The starter is a working streaming server whose
buffers grow without limit; the work is the bound: who pauses, who is cut off, and
how the loop keeps going when the only thing it could do is wait for a client.

## Specification

```python
class Server:
    def __init__(self, engine, max_running: int, max_buffered: int, stall_ms: float): ...
    def submit(self, request) -> Stream: ...
    async def run(self) -> None: ...


class Stream:  # returned by submit; the class itself is yours
    state: str  # "open", "finished", "stalled", or "closed"

    def take(self) -> list[int]: ...
    def close(self) -> None: ...
```

`engine` is a [`SimEngine`](../../../python/src/learnai/kit/simengine.py) with
`max_running` slots and requests are
[`SimRequest`](../../../python/src/learnai/kit/workloads.py) objects. Time is virtual
and counted in **milliseconds**. A step over $P$ prompt tokens and $D$ decoded
sequences takes $8 + 0.25\,P + D$ ms.

**Streams.** `submit(request)` is called by the test at the request's arrival
instant; it puts the request at the end of the waiting list and returns its stream,
in state `"open"`. Each token the engine produces for the request is appended to the
stream's **buffer** as its index within the request (0, 1, 2, ...). The test plays
the transport: whenever the client can receive data it calls `take()`, which removes
and returns everything buffered, oldest first, as one list (several tokens coalesce
into one chunk; `[]` if nothing is buffered). `take()` never blocks.

**The bound.** A buffer never holds more than `max_buffered` tokens. A running
sequence whose buffer holds `max_buffered` tokens is **paused**: it keeps its slot
and its place in the admission order, but it is left out of every step until a
`take()` has emptied the buffer.

**The loop.** `run()` never returns. Before every step:

1. **Admit.** While a slot is free and a request is waiting, admit the oldest waiting
   request; its whole prompt is prefilled in this step. A paused sequence still
   occupies its slot.
2. **Decode.** Every sequence that was running before (1) and is not paused is
   decoded, in admission order.
3. **Nothing to do.** If (1) and (2) leave both lists empty, there is no step: block
   until a `submit`, `take`, `close`, or stall changes that, then plan again at that
   instant. Never run an empty step, never sleep or poll, and never wait for one
   particular client.

When a step returns, each token is appended to its stream's buffer, a sequence that
produced its last token is finished (its stream's state becomes `"finished"` and its
slot is free), and the next step is planned at the same instant.

**How a stream ends.** A stream is `"open"` from `submit` until the first of:

| it | happens when | then |
|---|---|---|
| finishes | the engine produces its last token | state `"finished"`; buffered tokens stay available to `take()` |
| stalls | its buffer has held `max_buffered` tokens for `stall_ms` without a `take()` | state `"stalled"` |
| is closed | `close()` is called while it is open | state `"closed"` |

- **Stall.** The clock starts at the instant the token that fills the buffer is
  produced and is stopped by any `take()`. If it reaches `stall_ms`, the client is
  given up on at that instant. A stream that finishes is never stalled, even if its
  last token filled the buffer.
- **Close.** The client disconnected. `close()` on a stream that is not open changes
  nothing.
- **Both take effect at once**, inside the `close()` call or at the stall instant: a
  running sequence is dropped from the engine with `engine.release(id)`, also when a
  step is in flight (the engine then gives it nothing from that step), and its slot
  is free when the next step is planned; a waiting request leaves the waiting list.
  The buffer is discarded: `take()` returns `[]` from then on.

**Events at one instant.** The test calls `submit`, `take`, and `close` from ordinary
timers, and the engine completes a step after every ordinary timer of the same
instant. So a `take()` at the instant a token is due does not return that token, and
the buffer it emptied counts as empty when the next step is planned. A `take()` at
the very instant a stall is due comes first and saves the stream. A stall that is
due at the instant a step ends frees its slot before the next step is planned. The
tests never call `take` and `close` of one stream at the same instant.

## Example

`max_running = 2`, `max_buffered = 2`, `stall_ms = 30`. Requests `a` and `b` arrive
at 0 with 8 prompt tokens each; `a` wants 5 output tokens and `b` wants 8. The client
of `a` takes every 10 ms from 15.5 on; the client of `b` takes at 25.5 and at 80.5
and never again.

| time | event |
|---|---|
| 0 | step 0: prefill `a`, `b` (12 ms) |
| 12 | first tokens. Step 1: decode `a`, `b` (10 ms) |
| 15.5 | `a.take()` returns `[0]` |
| 22 | second tokens: `b` holds `[0, 1]`, is full, and is paused; its stall is due at 52. Step 2: decode `a` only (9 ms) |
| 25.5 | `a.take()` returns `[1]`; `b.take()` returns `[0, 1]`, which stops the clock |
| 31 | step 3: decode `a`, `b` (10 ms): `b` runs again |
| 41 | step 4: decode `a`, `b` (10 ms); `b` holds `[2]` |
| 51 | `a` finishes with 5 tokens. `b` holds `[2, 3]`, is full, stall due at 81. Nothing can be decoded: the loop blocks |
| 55.5 | `a.take()` returns `[4]`; `a.state` is `"finished"` |
| 80.5 | `b.take()` returns `[2, 3]`, half a millisecond before the stall. Step 5 starts at once: decode `b` (9 ms) |
| 89.5 | step 6: decode `b` (9 ms) |
| 98.5 | `b` holds `[4, 5]`, is full, stall due at 128.5; the loop blocks |
| 128.5 | `b` stalls: `engine.release("b")`, state `"stalled"`, after 6 of its 8 tokens |

The engine ran 7 steps. Between 51 and 80.5 and after 98.5 it was idle although `b`
was unfinished: generating tokens that nobody reads is the cost this design refuses
to pay.

## Constraints

- `1 <= max_running <= 8`, `1 <= max_buffered <= 8`, `stall_ms > 0`. Arrival times,
  `stall_ms`, and the instants of `take` and `close` are multiples of 0.125 ms, so
  every instant is an exact float and ties are real.
- The tests record what every `take()` returned and the stream's state after every
  `take()` and `close()`, every engine step, every `engine.release`, and the final
  state of every stream. All are compared exactly: with literal values on small
  scenarios, and on seeded Poisson and bursty workloads of up to 50 requests with the
  reference and with the rules above. The engine must hold no sequence when the
  simulation ends.
- **Peak buffered tokens.** At every instant, for every stream, the tokens the engine
  has produced for it minus the tokens taken is at most `max_buffered`. One case
  reports the peak over a workload of 40 requests whose clients read fast, slowly, or
  not at all, with `max_buffered = 8`: the limit is **8 tokens** (the reference
  reaches exactly 8; a buffer that pauses one token late reaches 9, and one that never
  pauses reaches 24 before the stall limit ends the stream).
- **No head-of-line blocking.** With a slot for every request, a client that stops
  reading must not make any token of another request later than it is when every
  client reads.
- A `ProtocolViolation` from the engine fails the case even if it is caught.
- Use `asyncio` and the engine only; time is virtual.

## Why this matters

A token stream ends in a socket, and the socket is only as fast as the client behind
it: a phone on a bad network, a browser tab in the background, a proxy that buffers,
a script that opened a stream and is stuck in a debugger. If the server keeps
generating for such a client, it pays GPU time for text nobody reads and holds the
text in memory, per connection, for as long as the client likes; a few thousand
abandoned streams are an out-of-memory kill. If instead the engine loop waits for the
slow client, every other request in the batch waits with it, and one bad connection
sets the token rate of the whole server. The bounded buffer between the two is what
makes both costs finite: the engine never waits, a slow client is paused instead of
buffered, and a client that stays away is released like any canceled request.
