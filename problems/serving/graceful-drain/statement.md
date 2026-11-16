Shut a continuous-batching server down without dropping the work it has accepted:
stop taking requests, let the ones in the house finish, cut off whatever is still
running when the grace period ends, and say "done" only when the engine is really
idle. The starter is a working server that never stops; the work is `drain`.

## Specification

```python
class Unavailable(Exception): ...


class Terminated(Exception):  # .tokens: output tokens produced before the cut
    def __init__(self, tokens: int): ...


class Server:
    ready: bool  # the readiness probe
    closed: bool  # True once a drain has completed

    def __init__(self, engine, max_running: int): ...
    def submit(self, request) -> Handle: ...
    def drain(self, grace_ms: float) -> asyncio.Future: ...
    async def run(self) -> None: ...


class Handle:  # returned by submit; the class itself is yours
    async def result(self) -> int: ...
```

`engine` is a [`SimEngine`](../../../python/src/learnai/kit/simengine.py) with
`max_running` slots and requests are
[`SimRequest`](../../../python/src/learnai/kit/workloads.py) objects. Time is virtual
and counted in **milliseconds**. A step over $P$ prompt tokens and $D$ decoded
sequences takes $8 + 0.25\,P + D$ ms.

**Serving.** Until a drain starts, the server is the loop of
`continuous-batching-scheduler`, and the starter contains it: before every step each
free slot goes to the oldest waiting request (admit it, prefill its whole prompt in
this step), and every sequence that was already running is decoded, in admission
order; with nothing waiting and nothing running the loop blocks until it is woken.
`submit(request)` is called by the test at the request's arrival instant, from a plain
callback inside the event loop; it appends the request to the waiting list and returns
a handle. `await handle.result()` returns the number of output tokens when the request
finishes. `ready` is `True` and `closed` is `False`.

**`drain(grace_ms)`** is called at most once, from a plain callback inside the event
loop, with `grace_ms >= 0`. It is a plain method, not a coroutine: everything in (1)
has happened when it returns.

1. **Stop admitting, at once.** `ready` becomes `False`, and from now on `submit`
   raises `Unavailable` without accepting the request, forever.
2. **Keep serving what was accepted.** Requests that were submitted before the drain
   are *pending* until they end, whether they are running or still waiting for a slot.
   The loop serves them exactly as before: waiting requests are still admitted as
   slots free up, and every step is the step the loop would have run without a drain.
3. **Cut off at the grace deadline.** The deadline is the instant of the call plus
   `grace_ms`. Every request still pending at that instant is *terminated* there:
   a running one is dropped from the engine with `engine.release(id)`, also when a
   step is in flight (the engine then gives it nothing from that step, which still
   takes as long as planned); a waiting one leaves the waiting list. Its client's
   `result()` raises `Terminated` at that instant, with `.tokens` equal to the number
   of output tokens the request had produced (0 for a request that never got a slot).
4. **Resolve when idle.** `drain` returns an `asyncio.Future`. It resolves at the
   first instant at or after the call at which no request is pending **and** the
   engine is not running a step. Its value is the number of requests that were
   terminated, an `int`. At that same instant `closed` becomes `True` and `run()`
   returns.

So the future resolves at the instant the last pending request finishes if that
happens before the deadline; at the instant of the call if the server was idle; and,
if requests were terminated in the middle of a step, when that step ends, which is
after the deadline. It never waits out a grace period that nobody needs.

**Events at one instant.**

- The grace deadline is an ordinary timer, and the engine completes a step after
  every ordinary timer of the same instant. So a token due at exactly the deadline is
  not produced, and a request whose last token is due then is terminated, not
  finished. Use `loop.call_at(deadline, callback)`; a task that sleeps until the
  deadline is woken by its timer but runs only after the step has completed.
- With `grace_ms = 0` the deadline is the call itself: everything pending is
  terminated inside `drain`, before it returns. (A timer set for the current instant
  would run after a step that ends at this instant has completed.)
- The test calls `submit` and `drain` from ordinary timers. A request submitted at
  the instant of the drain is accepted if its `submit` call comes before the `drain`
  call and refused if it comes after; the tests do both.
- A request that arrives at the instant a step ends is submitted before the loop
  plans the next step, as in `continuous-batching-scheduler`.

`result()` may be awaited before or after the request ends, and always reports the
same outcome.

## Example

`max_running = 2`. Requests as (arrival, prompt tokens, output tokens): request 0
`(0, 8, 4)`, request 1 `(0, 8, 2)`, request 2 `(4, 8, 4)`, request 3 `(30, 8, 2)`.
`drain(30)` is called at 20, so the grace deadline is 50.

| time | event |
|---|---|
| 0 | step 0: prefill 0 and 1 (12 ms). Request 2 arrives at 4 and waits. |
| 12 | step 1: decode 0, 1 (10 ms) |
| 20 | `drain(30)`: `ready` is `False`; requests 0, 1, 2 are pending |
| 22 | request 1 finishes with 2 tokens. Step 2: prefill 2, decode 0 (11 ms): the waiting request is admitted during the drain |
| 30 | request 3 arrives: `submit` raises `Unavailable` |
| 33 | step 3: decode 0, 2 (10 ms) |
| 43 | request 0 finishes with 4 tokens. Step 4: decode 2 (9 ms, until 52) |
| 50 | the grace deadline, mid-step: `engine.release(2)`; its client gets `Terminated` with `tokens = 2` |
| 52 | step 4 ends with nothing produced. The future resolves to `1`; `closed` is `True`; `run()` returns |

Had request 2 needed only 3 tokens, it would have finished at 52, after the
deadline, so it is terminated at 50 all the same. Had it needed 2, it would have
finished at 43 and the future would have resolved at 43 with the value `0`.

## Constraints

- `1 <= max_running <= 8`. Arrival times, the drain instant, and `grace_ms` are
  multiples of 0.25 ms, so every instant is an exact float and ties are real.
- The tests record, for every request, whether `submit` accepted it and when its
  client's `result()` returned or raised and with what; every engine step and every
  `engine.release`; the value of `ready` right after `drain` returns; when the future
  resolved, to what, and the value of `closed` then; and when `run()` returned. All
  are compared exactly: with literal values on small scenarios, and on seeded Poisson
  and bursty workloads of up to 60 requests with the reference and with the rules
  above. The engine must hold no sequence when the simulation ends.
- A `ProtocolViolation` from the engine fails the case even if it is caught.
- Use `asyncio` and the engine only; time is virtual.

## Why this matters

Every deployment, autoscaler scale-down, and node replacement ends with a server
being told to stop while it is in the middle of streams that took seconds of GPU time
to get where they are. A server that exits at once throws that work away and shows
its users a broken connection; a server that never stops admitting never gets idle
and is killed anyway; a server that reports "drained" while a forward pass still
runs has its GPU memory pulled from under it. And the simple fix, sleeping for the
whole grace period, makes every rollout of a hundred replicas take a hundred grace
periods. The sequence in this problem (fail readiness, refuse new work, finish what
was accepted, cut off the rest at a deadline, then exit) is what a Kubernetes pod is
expected to do between `SIGTERM` and `SIGKILL`.
