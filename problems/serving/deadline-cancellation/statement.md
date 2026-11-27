Add client cancellation and per-request deadlines to a continuous-batching scheduler. When a client cancels a request or virtual time reaches its deadline, the scheduler must stop serving that request immediately—whether it is waiting or mid-step—and notify the client.

![This problem is the Waiting queue and the Engine loop, where a canceled or late request has to stop.](assets/map.svg)

## Specification

```python
class Canceled(Exception): ...
class DeadlineExceeded(Exception): ...

class Scheduler:
    def __init__(self, engine, max_running: int): ...
    def submit(self, request) -> Handle: ...
    async def run(self) -> None: ...

class Handle:                       # returned by submit; the class itself is yours
    def cancel(self) -> bool: ...
    async def result(self) -> int: ...
```

`engine` is a [`SimEngine`](../../../python/src/learnai/kit/simengine.py) with `max_running` slots, and requests are [`SimRequest`](../../../python/src/learnai/kit/workloads.py) objects. Virtual time is measured in **milliseconds**. A step with $P$ prompt tokens and $D$ decoded sequences takes $8 + 0.25\,P + D$ ms.

**The loop.** `run()` runs indefinitely and follows the `continuous-batching-scheduler` loop. Before each step, each free slot goes to the oldest waiting request (admitting it and prefilling its whole prompt in this step), and every already-running sequence is decoded in admission order. When nothing is waiting and nothing is running, `run()` blocks until `submit` wakes it. The test calls `submit(request)` at the request's arrival instant from a plain callback inside the event loop. `submit` appends the request to the waiting list and returns a `Handle` immediately.

**How a request ends.** Every request ends exactly once, in whichever of three ways happens first:

| outcome | happens when | `await handle.result()` |
|---|---|---|
| finishes | the engine produces its last token | returns the number of output tokens |
| is canceled | `handle.cancel()` is called while pending | raises `Canceled` |
| expires | virtual time reaches `request.deadline_s` while pending | raises `DeadlineExceeded` |

A request is *pending* from submission until it ends (whether waiting or running). `request.deadline_s` is either `None` (no deadline) or an absolute virtual time in milliseconds.

**Immediate early termination.** Cancellation and expiration take effect immediately inside `cancel()` or at the deadline instant, not on the next iteration of `run()`:

- A **waiting** request is removed from the waiting list and is never admitted.
- A **running** request is dropped from the engine via `engine.release(id)`, even if a step is currently in flight. The engine produces no output for it from that step, though the step still takes its planned duration. No token is ever produced for a request at or after the instant it ends, and its slot is free when the next step is planned.
- The client's `await handle.result()` raises at that same instant.

`handle.cancel()` returns `True` if it ended the request. It returns `False` and changes nothing if the request had already ended in any of the three ways.

**The loop after an early end.** A sequence released during a step appears in neither `result.tokens` nor `result.finished`. If every sequence in an in-flight step is released, the step finishes with nothing produced and the loop continues normally, admitting waiting requests if any exist or going idle otherwise. The loop must never execute an empty step.

**Events at the same instant.**

- A deadline runs as an ordinary event-loop timer, and the engine completes a step after all ordinary timers scheduled for that instant. A request therefore expires *before* a token due at the exact same instant, so it finishes in time only if its last token appears strictly before `deadline_s`. Use `loop.call_at(deadline, callback)` rather than a sleeping task, because a sleeping task is woken by its timer but only resumes after the step completes.
- The test also calls `cancel()` from an ordinary timer, so a cancel at the exact instant a token is due takes effect first and suppresses that token.
- A request arriving with `deadline_s <= now` expires immediately inside `submit`. It never enters the waiting list, and `result()` raises `DeadlineExceeded` at the arrival instant.
- A request arriving at the exact instant a step ends is submitted before the loop plans the next step, as in `continuous-batching-scheduler`.
- The tests never call `cancel()` at the exact instant of the same request's deadline.

`result()` may be awaited before or after the request ends and always reports the same outcome.

## Example

Suppose `max_running = 2`. Requests are listed as `(arrival, prompt tokens, output tokens, deadline)`:

- request 0: `(0, 8, 6, None)`
- request 1: `(0, 8, 3, 30)`
- request 2: `(4, 8, 2, 25)`
- request 3: `(6, 16, 3, 70)`
- request 4: `(12, 8, 2, 56)`

The client of request 0 calls `cancel()` at time 36.

| time | event |
|---|---|
| 0 | step 0: prefill 0 and 1 (12 ms). Requests 2, 3, and 4 arrive during this step and wait. |
| 12 | step 1: decode 0 and 1 (10 ms) |
| 22 | step 2: decode 0 and 1 (until 32) |
| 25 | request 2 expires while **waiting**; its client gets `DeadlineExceeded` |
| 30 | request 1 expires while **running**, mid-step: `engine.release(1)` |
| 32 | step 2 ends with a token for request 0 only. Step 3: prefill 3, decode 0 (13 ms) |
| 36 | `cancel()` on request 0, mid-step: `engine.release(0)`, returns `True` |
| 45 | step 3 ends with the first token of 3 and nothing for 0. Step 4: prefill 4, decode 3 (11 ms) |
| 56 | request 4 expires; its first token was due at this instant and is not produced |
| 56 | step 4 ends with a token for 3. Step 5: decode 3 (9 ms) |
| 65 | request 3 finishes with 3 tokens, before its deadline of 70 |

Request 1 produced 2 of its 3 tokens and request 0 produced 3 of its 6 tokens before ending. The engine ran 6 steps and produced 8 tokens. `engine.release` was called for request 1 at 30, request 0 at 36, and request 4 at 56.

## Constraints

- `1 <= max_running <= 8`. Arrival times, deadlines, and cancel times are multiples of 0.25 ms, so every timestamp is exact in floating point and ties occur cleanly.
- For every request, the tests record when `result()` returns or raises and with what outcome, every engine step, and every `engine.release` call. All three are compared exactly on small scenarios and on seeded Poisson and bursty workloads of up to 100 requests (about half with deadlines and about a quarter canceled). The engine's total work (steps, prefill tokens, decode tokens, output tokens) must match the reference, and the engine must hold no sequences when the simulation ends.
- A `ProtocolViolation` from the engine fails the test even if caught. Releasing a sequence that already finished or was never admitted triggers a `ProtocolViolation`.
- Use `asyncio` and the engine only. Time is virtual.

## Why this matters

Clients of an LLM server disconnect constantly when users close a tab, hit stop, hit a gateway timeout, or issue a retry. If the scheduler ignores disconnects, an abandoned sequence continues holding a batch slot and KV-cache memory while decoding tokens nobody will read.

Under heavy load, this creates a feedback loop. Rising latency causes more clients to time out and retry, which fills the engine with even more abandoned work. Production servers propagate disconnects and deadlines straight into the scheduler to avoid four common bugs: honoring cancellation only in the queue, leaking slots or KV blocks on early exit, checking deadlines only at admission, and letting stale timers fire after a request has already left.
