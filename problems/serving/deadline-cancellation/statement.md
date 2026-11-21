Add cancellation and deadlines to a continuous-batching scheduler: a request that its
client gave up on, or that can no longer finish in time, must stop consuming the
engine at that very instant, whether it is waiting for a slot or in the middle of a
forward pass, and its client must be told. The starter is a working scheduler that
serves every request to the end; the work is everything that can end a request early.

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

`engine` is a [`SimEngine`](../../../python/src/learnai/kit/simengine.py) with
`max_running` slots and requests are
[`SimRequest`](../../../python/src/learnai/kit/workloads.py) objects. Time is virtual
and counted in **milliseconds**. A step over $P$ prompt tokens and $D$ decoded
sequences takes $8 + 0.25\,P + D$ ms.

**The loop.** `run()` never returns. It is the loop of `continuous-batching-scheduler`:
before every step, each free slot goes to the oldest waiting request (admit it, prefill
its whole prompt in this step), and every sequence that was already running is decoded,
in admission order; with nothing waiting and nothing running it blocks until `submit`
wakes it. `submit(request)` is called by the test at the request's arrival instant,
from a plain callback inside the event loop; it appends the request to the waiting
list and returns a handle immediately.

**How a request ends.** Every request ends exactly once, in the first of three ways:

| it | happens when | `await handle.result()` |
|---|---|---|
| finishes | the engine produces its last token | returns the number of output tokens |
| is canceled | `handle.cancel()` is called while it is pending | raises `Canceled` |
| expires | virtual time reaches `request.deadline_s` while it is pending | raises `DeadlineExceeded` |

*Pending* means waiting or running: submitted and not ended. `request.deadline_s` is
`None` (no deadline) or an absolute virtual time in milliseconds.

**Ending early takes effect at once**, inside the `cancel()` call or at the deadline
instant, not when the loop next gets around to it:

- a **waiting** request leaves the waiting list and is never admitted;
- a **running** request is dropped from the engine with `engine.release(id)`, also
  when a step is in flight. The engine then gives it nothing from that step (the step
  still takes as long as planned), so no token is ever produced for a request at or
  after the instant it ended. Its slot is free when the next step is planned;
- the client's `result()` raises at that same instant.

`handle.cancel()` returns `True` if it ended the request, and `False`, changing
nothing, if the request had already ended in any of the three ways.

**The loop after an early end.** A sequence released during a step is in neither
`result.tokens` nor `result.finished`. If every sequence of a step is released, the
step ends with nothing produced and the loop carries on: it admits waiting requests if
there are any and otherwise goes idle. It must never run an empty step.

**Events at one instant.**

- A deadline is an ordinary timer, and the engine completes a step after every
  ordinary timer of the same instant. So a request expires *before* a token that is due
  at exactly its deadline: a request finishes in time only if its last token appears
  strictly before `deadline_s`. Use `loop.call_at(deadline, callback)` for this; a
  task that sleeps until the deadline is woken by its timer but runs only after the
  step has completed.
- The test calls `cancel()` from an ordinary timer as well, so a cancel at the instant
  a token is due also comes first, and that token is not produced.
- A request that arrives with `deadline_s <= now` expires in `submit`: it never enters
  the waiting list, and `result()` raises `DeadlineExceeded` at the arrival instant.
- A request that arrives at the instant a step ends is submitted before the loop
  plans the next step, as in `continuous-batching-scheduler`.
- The tests never call `cancel()` at the exact instant of the same request's deadline.

`result()` may be awaited before or after the request ends, and always reports the
same outcome.

## Example

`max_running = 2`. Requests as (arrival, prompt tokens, output tokens, deadline):
request 0 `(0, 8, 6, none)`, request 1 `(0, 8, 3, 30)`, request 2 `(4, 8, 2, 25)`,
request 3 `(6, 16, 3, 70)`, request 4 `(12, 8, 2, 56)`. The client of request 0 calls
`cancel()` at 36.

| time | event |
|---|---|
| 0 | step 0: prefill 0 and 1 (12 ms). Requests 2, 3, 4 arrive during it and wait. |
| 12 | step 1: decode 0, 1 (10 ms) |
| 22 | step 2: decode 0, 1 (until 32) |
| 25 | request 2 expires while **waiting**; its client gets `DeadlineExceeded` |
| 30 | request 1 expires while **running**, mid-step: `engine.release(1)` |
| 32 | step 2 ends: a token for request 0 only. Step 3: prefill 3, decode 0 (13 ms) |
| 36 | `cancel()` on request 0, mid-step: `engine.release(0)`, returns `True` |
| 45 | step 3 ends: first token of 3, nothing for 0. Step 4: prefill 4, decode 3 (11 ms) |
| 56 | request 4 expires; its first token was due at this instant and is not produced |
| 56 | step 4 ends: a token for 3. Step 5: decode 3 (9 ms) |
| 65 | request 3 finishes with 3 tokens, before its deadline of 70 |

Request 1 had produced 2 of its 3 tokens and request 0 had produced 3 of 6 when they
ended. The engine ran 6 steps and produced 8 tokens; `engine.release` was called for
1 at 30, 0 at 36, and 4 at 56.

## Constraints

- `1 <= max_running <= 8`. Arrival times, deadlines, and cancel times are multiples of
  0.25 ms, so every instant is an exact float and ties are real.
- The tests record, for every request, when its client's `result()` returned or raised
  and with what, every engine step, and every `engine.release`. All three are compared
  exactly: with literal values on small scenarios, and on seeded Poisson and bursty
  workloads of up to 100 requests (about half with deadlines, about a quarter
  canceled) with the reference and with the rules above. The engine's total work
  (steps, prefill tokens, decode tokens, output tokens) must equal the reference's,
  and the engine must hold no sequence when the simulation ends.
- A `ProtocolViolation` from the engine fails the case even if it is caught: releasing
  a sequence that already finished or was never admitted is one.
- Use `asyncio` and the engine only; time is virtual.

## Why this matters

Clients of an LLM server leave all the time: a user closes the tab or presses stop, a
gateway times out after 30 seconds, a retry supersedes the original call. A sequence
whose client is gone still occupies a batch slot and KV-cache memory and still costs a
decode per step, possibly for thousands of tokens, while live requests queue behind
it. Under overload this turns into a spiral: latency rises, more clients time out and
retry, and the engine spends a growing share of its capacity on output nobody reads.
Every serious server therefore propagates disconnects and deadlines into its
scheduler, and the bugs are the ones this problem pins down: cancellation that only
works in the queue, a slot or a KV block that is never given back, a deadline checked
once at admission, and a timer that fires for a request that has already left.
