Add bounded per-client buffers and backpressure to a continuous-batching streaming server so that a slow, stalled, or disconnected client consumes bounded resources without delaying other requests. The starter code is a working streaming server whose buffers grow without limit. Your task is to enforce the buffer limit, pause streams whose buffers are full, release stalled or closed streams, and keep the serving loop idle when no sequence can make progress.

![This problem is the Token stream, here a bounded buffer for each client.](assets/map.svg)

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

`engine` is a [`SimEngine`](../../../python/src/learnai/kit/simengine.py) with `max_running` slots, and each `request` is a [`SimRequest`](../../../python/src/learnai/kit/workloads.py). Time is virtual and measured in **milliseconds**. A step over $P$ prompt tokens and $D$ decoded sequences takes $8 + 0.25\,P + D$ ms.

**Streams.** The test calls `submit(request)` at the request's arrival instant. `submit` appends the request to the end of the waiting list and returns a `Stream` in state `"open"`. Each token the engine produces for the request is appended to the stream's **buffer** as its 0-based index within the request (`0`, `1`, `2`, ...). The test acts as the transport. Whenever the client can receive data, it calls `take()`, which removes and returns all buffered token indices in oldest-first order as one list, or `[]` if the buffer is empty. Multiple buffered tokens coalesce into a single list, and `take()` never blocks.

**The bound.** A stream's buffer never holds more than `max_buffered` tokens. A running sequence whose buffer holds `max_buffered` tokens is **paused**. It keeps its engine slot and its place in the admission order, but it is omitted from every step until a `take()` empties its buffer.

**The loop.** `run()` never returns. Before every step, it plans work in three stages:

1. **Admit.** While a slot is free and a request is waiting, admit the oldest waiting request and prefill its full prompt in this step. A paused sequence still occupies its slot.
2. **Decode.** Include every sequence that was already running before stage 1 and is not paused, in admission order.
3. **Nothing to do.** If stages 1 and 2 leave both the prefill and decode lists empty, do not run a step. Block until a `submit`, `take`, `close`, or stall changes the state, then plan again at that instant. Never run an empty step, never sleep or poll, and never wait on a single client.

When a step completes, append each produced token to its stream's buffer. If a sequence produced its last token, set its stream's state to `"finished"` and free its slot. Then plan the next step at the same instant.

**How a stream ends.** A stream stays `"open"` from `submit` until the first of three events occurs:

| event | happens when | result |
|---|---|---|
| finishes | the engine produces its last token | state becomes `"finished"`, and buffered tokens stay available to `take()` |
| stalls | its buffer holds `max_buffered` tokens for `stall_ms` without a `take()` | state becomes `"stalled"` |
| is closed | `close()` is called while it is `"open"` | state becomes `"closed"` |

- **Stall.** The stall timer starts at the instant the token that fills the buffer is produced and stops on any `take()`. If the timer reaches `stall_ms`, the stream stalls at that instant. A stream that finishes never stalls, even if its last token filled the buffer.
- **Close.** `close()` signals that the client disconnected. Calling `close()` on a stream that is not `"open"` does nothing.
- **Immediate cleanup.** Both a stall and a `close()` take effect immediately, either at the stall instant or inside the `close()` call. If the sequence is running, drop it from the engine with `engine.release(id)` (even while a step is in flight, in which case the engine produces nothing for it from that step) and free its slot before the next step is planned. If the request is waiting, remove it from the waiting list. Discard the buffer so `take()` returns `[]` from then on.

**Events at the same instant.** The test calls `submit`, `take`, and `close` from ordinary event-loop timers. The engine completes a step after all ordinary timers scheduled for that instant have run. Consequently, a `take()` at the instant a token is due does not return that token, and the buffer it emptied counts as empty when the next step is planned. A `take()` at the exact instant a stall is due runs first and prevents the stall. A stall due at the instant a step ends frees its slot before the next step is planned. The tests never call `take()` and `close()` on the same stream at the same instant.

## Example

Suppose `max_running = 2`, `max_buffered = 2`, and `stall_ms = 30`. Requests `a` and `b` arrive at 0 ms with 8 prompt tokens each. Request `a` wants 5 output tokens and `b` wants 8. Client `a` calls `take()` every 10 ms starting at 15.5 ms. Client `b` calls `take()` at 25.5 ms and 80.5 ms and never again.

| time (ms) | event |
|---|---|
| 0 | Step 0 prefills `a` and `b` (12 ms). |
| 12 | First tokens arrive. Step 1 decodes `a` and `b` (10 ms). |
| 15.5 | `a.take()` returns `[0]`. |
| 22 | Second tokens arrive. `b` holds `[0, 1]`, so it is full and paused with a stall due at 52 ms. Step 2 decodes `a` only (9 ms). |
| 25.5 | `a.take()` returns `[1]`. `b.take()` returns `[0, 1]` and cancels its stall timer. |
| 31 | Step 3 decodes `a` and `b` (10 ms) now that `b` is unpaused. |
| 41 | `b` holds `[2]`. Step 4 decodes `a` and `b` (10 ms). |
| 51 | `a` finishes with 5 tokens. `b` holds `[2, 3]`, is full, and has a stall due at 81 ms. No sequence can run, so the loop blocks. |
| 55.5 | `a.take()` returns `[4]`, and `a.state` is `"finished"`. |
| 80.5 | `b.take()` returns `[2, 3]` half a millisecond before its stall. Step 5 starts immediately and decodes `b` (9 ms). |
| 89.5 | Step 6 decodes `b` (9 ms). |
| 98.5 | `b` holds `[4, 5]`, is full, and has a stall due at 128.5 ms. The loop blocks. |
| 128.5 | `b` stalls. The server calls `engine.release("b")` and sets `b.state` to `"stalled"` after 6 of its 8 tokens. |

The engine ran 7 steps. Between 51 ms and 80.5 ms and after 98.5 ms, it stayed idle even though `b` had not finished, avoiding wasted generation for unread tokens.

## Constraints

- `1 <= max_running <= 8`, `1 <= max_buffered <= 8`, and `stall_ms > 0`. Arrival times, `stall_ms`, and the instants of `take()` and `close()` are multiples of 0.125 ms, so every timestamp is an exact float and ties are exact.
- The tests record what every `take()` returns, the stream's state after every `take()` and `close()`, every engine step, every `engine.release` call, and the final state of every stream. All outputs are checked for exact equality against literal values on small scenarios and against the reference implementation on seeded Poisson and bursty workloads of up to 50 requests. The engine must hold no sequences when the simulation ends.
- **Peak buffered tokens.** At every instant and for every stream, the number of produced tokens minus taken tokens must be at most `max_buffered`. One test measures this peak over 40 requests with fast, slow, and stalled clients at `max_buffered = 8`. The limit is **8 tokens**. The reference reaches 8, a buffer that pauses one token late reaches 9, and an unbounded buffer reaches 24 before the stall limit ends the stream.
- **No head-of-line blocking.** When every request has its own slot, a client that stops reading must not delay any token of another request compared to a run where every client reads.
- A `ProtocolViolation` from the engine fails the test even if caught.
- Use `asyncio` and the engine only. Time is virtual.

## Why this matters

Every token stream ends at a network socket, and that socket only drains as fast as the client behind it. A phone on a poor connection, a backgrounded browser tab, a buffering proxy, or a script paused in a debugger can all stop reading. If the server keeps generating for that client, it wastes GPU cycles on unread text and accumulates memory per connection until a few thousand abandoned streams trigger an out-of-memory kill.

Waiting on the slow client inside the engine loop is even worse, because every other request in the batch stalls behind one bad connection. Placing a bounded buffer between the engine and each client caps both costs. The engine loop never waits on a client, a slow client pauses without affecting the batch, and a client that stays stalled is released just like a disconnected one.
