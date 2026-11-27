Implement graceful shutdown for a continuous-batching inference server. When `drain` is called, stop accepting new requests, finish accepted requests within a grace period, terminate any remaining work at the deadline, and signal completion once the engine is idle.

![This problem is one replica as a whole, shutting down without dropping accepted requests.](assets/map.svg)

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

`engine` is a [`SimEngine`](../../../python/src/learnai/kit/simengine.py) with `max_running` slots, and requests are [`SimRequest`](../../../python/src/learnai/kit/workloads.py) objects. Time is virtual and measured in **milliseconds**. A step over $P$ prompt tokens and $D$ decoded sequences takes $8 + 0.25\,P + D$ ms.

### Serving before a drain

The starter implements a working continuous-batching server (as in `continuous-batching-scheduler`). Before every step, each free slot is given to the oldest waiting request, which is admitted and has its entire prompt prefilled in that step. Every sequence already running is decoded in admission order. When nothing is waiting and nothing is running, the loop blocks until woken.

The test calls `submit(request)` at the request's arrival instant from a plain callback inside the event loop. `submit` appends the request to the waiting list and returns a `Handle`. `await handle.result()` returns the number of output tokens when the request finishes. Before a drain, `ready` is `True` and `closed` is `False`.

### Draining

`drain(grace_ms)` is called at most once from a plain callback inside the event loop, with `grace_ms >= 0`. It is a synchronous method, not a coroutine, so step 1 below must take effect before `drain` returns.

1. **Stop admitting immediately.** Set `ready = False`. From this point on, every call to `submit` raises `Unavailable` without accepting the request.
2. **Keep serving accepted requests.** Any request submitted before `drain` is *pending* until it finishes or is terminated, whether it is already running or still waiting for a slot. Continue serving pending requests exactly as before: admit waiting requests as slots open, and execute the same steps the loop would have run without a drain.
3. **Cut off remaining work at the grace deadline.** The deadline is the instant `drain` is called plus `grace_ms`. Any request still pending at that instant is *terminated*:
   - Release each running sequence with `engine.release(id)`, even if a step is currently in flight. The engine then produces no token for that sequence in the in-flight step, though the step still takes its planned duration.
   - Remove every waiting request from the waiting list.
   - Make each terminated request's `result()` raise `Terminated` at that instant, with `.tokens` set to the number of output tokens already produced (`0` for a request that never received a slot).
4. **Resolve when idle.** `drain` returns an `asyncio.Future`. That future resolves at the first instant at or after the `drain` call when no request is pending **and** the engine is not running a step. Its value is the `int` count of terminated requests. At that same instant, `closed` becomes `True` and `run()` returns.

As a result, the future resolves immediately if the server was already idle, at the instant the last pending request finishes if that happens before the deadline, or when an in-flight step finishes if requests were terminated mid-step (which is after the deadline). Never wait out an unneeded grace period.

### Events at the same instant

- The grace deadline is an ordinary event-loop timer, and the engine completes a step *after* every ordinary timer scheduled for the same instant. A token due at exactly the deadline is therefore not produced, and a request whose final token is due then is terminated rather than finished. Schedule the deadline with `loop.call_at(deadline, callback)`; a task sleeping until the deadline wakes up from its timer but only runs after the step completes.
- When `grace_ms == 0`, the deadline is the `drain` call itself. Terminate all pending requests directly inside `drain` before it returns, because a timer scheduled for the current instant would run after a step ending at that instant.
- The test invokes `submit` and `drain` from ordinary timers. A request submitted at the same instant as `drain` is accepted if `submit` runs before `drain`, and refused if `submit` runs after `drain`.
- A request arriving at the instant a step ends is submitted before the loop plans the next step, matching `continuous-batching-scheduler`.
- `result()` may be awaited before or after a request ends and must always produce the same outcome.

## Example

Let `max_running = 2`. Four requests arrive as `(arrival_ms, prompt_tokens, output_tokens)`:

- Request 0: `(0, 8, 4)`
- Request 1: `(0, 8, 2)`
- Request 2: `(4, 8, 4)`
- Request 3: `(30, 8, 2)`

`drain(30)` is called at $t = 20$, setting the grace deadline to $t = 50$.

| time | event |
|---|---|
| 0 | step 0: prefill 0 and 1 (12 ms). Request 2 arrives at 4 and waits. |
| 12 | step 1: decode 0 and 1 (10 ms). |
| 20 | `drain(30)`: `ready` becomes `False`. Requests 0, 1, and 2 are pending. |
| 22 | request 1 finishes with 2 tokens. Step 2: prefill 2 and decode 0 (11 ms). Waiting request 2 is admitted during the drain. |
| 30 | request 3 arrives. `submit` raises `Unavailable`. |
| 33 | step 3: decode 0 and 2 (10 ms). |
| 43 | request 0 finishes with 4 tokens. Step 4: decode 2 (9 ms, until 52). |
| 50 | grace deadline fires mid-step. Call `engine.release(2)`; request 2's client gets `Terminated(tokens=2)`. |
| 52 | step 4 ends with no tokens produced. The drain future resolves to `1`, `closed` becomes `True`, and `run()` returns. |

If request 2 had needed only 3 tokens, its final token would have been due at 52 (after the deadline), so it would still be terminated at 50. If it had needed 2 tokens, it would have finished at 43, and the future would have resolved at 43 with value `0`.

## Constraints

- `1 <= max_running <= 8`. Arrival times, the drain instant, and `grace_ms` are multiples of 0.25 ms, so every timestamp is an exact float and simultaneous events are exact ties.
- The tests record whether `submit` accepted each request, when each client's `result()` returned or raised and with what value, every engine step and `engine.release` call, `ready` right after `drain` returns, when the drain future resolved and to what value, `closed` at resolution, and when `run()` returned.
- All recorded events are compared exactly against literal expectations on small scenarios and against the reference solution on seeded Poisson and bursty workloads of up to 60 requests.
- The engine must hold no sequences when the simulation ends.
- Any `ProtocolViolation` raised by the engine fails the test even if caught.
- Use `asyncio` and the engine only; time is virtual.

## Why this matters

Every deployment, autoscaler scale-down, and node replacement asks a running server to shut down while active streams hold seconds of GPU work. Exiting immediately drops those streams and breaks client connections. Continuing to accept new requests prevents the server from ever becoming idle under load, so the orchestrator eventually kills it anyway. Reporting "drained" while a forward pass is still executing causes GPU memory to be torn down underneath active kernels.

Sleeping for the entire grace period is also costly, because rolling out a hundred replicas then takes a hundred full grace periods. The protocol in this problem—fail readiness, refuse new work, finish accepted work, cut off stragglers at a deadline, and exit as soon as the engine is idle—is what a Kubernetes pod must execute between `SIGTERM` and `SIGKILL`.
