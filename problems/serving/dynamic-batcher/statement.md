Implement a dynamic batching policy for an inference server. The policy groups incoming requests so they can share one forward pass, while bounding how long any waiting request sits at an idle model.

![This problem is the Engine loop, here the decision of when to run a batch.](assets/map.svg)

## Specification

```python
class DynamicBatcher:
    def __init__(self, max_batch_size: int, max_wait_ms: int): ...
    def on_arrival(self, now_ms: int, request_id) -> list | None: ...
    def on_batch_done(self, now_ms: int) -> list | None: ...
    def on_timer(self, now_ms: int) -> list | None: ...
    def next_timer_ms(self) -> int | None: ...
```

The batcher is a state machine driven by a simulator. It does not sleep, start threads, or read a real clock. Every handler call passes the current virtual time `now_ms`, an integer in milliseconds that never decreases from one call to the next.

**The model.** The model runs one batch at a time and starts *idle*. It becomes *busy* as soon as a handler returns a batch and remains busy until the simulator calls `on_batch_done`. The simulator decides how long each batch runs and does not tell the batcher in advance.

**The queue.** The queue holds requests that have arrived but have not yet been dispatched, in the order of their `on_arrival` calls. Each request has a *deadline* equal to its arrival time plus `max_wait_ms`.

**The dispatch rule.** After every event, dispatch a batch if and only if all three conditions hold:

1. The model is idle.
2. The queue is not empty.
3. The queue holds at least `max_batch_size` requests, **or** `now_ms` is greater than or equal to the deadline of the oldest request in the queue.

When the rule fires, remove the oldest `min(max_batch_size, len(queue))` requests from the queue and return them oldest first. Younger requests stay in the queue and keep their original deadlines.

**The handlers.** Each handler updates state for its event, applies the dispatch rule once, and returns a list of request IDs or `None` if the rule does not fire. An empty list is treated the same as `None`.

- `on_arrival(now_ms, request_id)`: Append the request to the end of the queue. `request_id` is an opaque hashable value, unique per request.
- `on_batch_done(now_ms)`: Mark the model idle again.
- `on_timer(now_ms)`: Handle a requested wake-up at `now_ms`. No other state changes.

**The timer.** Because the simulator cannot infer when a deadline falls, it calls `next_timer_ms()` after every handler call. Return the integer virtual time at which you want `on_timer` called, or `None` if no wake-up is needed. There is only one timer, so each answer replaces the previous one. Any returned time that is less than or equal to the current time is ignored. When the clock reaches a pending timer value `T`, the simulator calls `on_timer(T)`, even if the model is busy at that moment. An extra wake-up when the rule does not fire is harmless, but failing to schedule a wake-up before an idle model reaches a deadline causes a missed dispatch.

**Events at the same instant.** When multiple events occur at the same `now_ms`, the simulator delivers them one at a time in a fixed order and applies the dispatch rule after each call:

1. `on_batch_done`, if the running batch finishes at `now_ms`.
2. `on_arrival` calls for that instant, in arrival order.
3. `on_timer`, if the pending timer is still scheduled for `now_ms` after the calls above.

Several edge cases follow directly from these rules:

- **Arrival exactly at the oldest request's deadline.** `on_arrival` runs before `on_timer`. Because the model is idle and the oldest deadline has been reached, `on_arrival` dispatches a batch that includes the new request.
- **Arrivals while the model is busy.** New requests join the queue. Nothing is dispatched while the model is busy, regardless of how many requests are waiting or how long they have waited.
- **Batch completion.** If at least `max_batch_size` requests are waiting, `on_batch_done` dispatches the oldest `max_batch_size` of them. Otherwise, if the oldest waiting request's deadline is at or before `now_ms`, `on_batch_done` dispatches all waiting requests. Otherwise, it returns `None`, and the waiting requests keep their original deadlines until the oldest deadline arrives or the queue fills up.
- **Arrival at the same instant as a batch completion.** `on_batch_done` runs first. If it dispatches a batch, the new arrival sees a busy model and waits for the next batch.
- **Several arrivals at one instant.** Each arrival is delivered as its own event. If the second of four simultaneous arrivals fills the batch, that call dispatches the batch, and the remaining two arrivals see a busy model.
- **`max_wait_ms = 0`.** A request arriving at an idle model reaches its deadline immediately and is dispatched in its own `on_arrival` call.

## Example

Let `max_batch_size = 3` and `max_wait_ms = 10`, with a simulated model that takes `16 + 2n` ms for a batch of `n` requests. Requests `0` through `9` arrive at times `0, 4, 25, 26, 27, 28, 29, 80, 84, 86`.

| time | call | returns | why |
|---|---|---|---|
| 0 | `on_arrival(0, 0)` | `None` | deadline of request 0 is 10; `next_timer_ms()` is 10 |
| 4 | `on_arrival(4, 1)` | `None` | two waiting, not full |
| 10 | `on_timer(10)` | `[0, 1]` | the oldest request reached its deadline; runs until 30 |
| 25 to 29 | `on_arrival` for 2, 3, 4, 5, 6 | `None` | the model is busy |
| 30 | `on_batch_done(30)` | `[2, 3, 4]` | five waiting, so the oldest three go; runs until 52 |
| 52 | `on_batch_done(52)` | `[5, 6]` | request 5's deadline (38) passed while busy; runs until 72 |
| 72 | `on_batch_done(72)` | `None` | nothing is waiting |
| 80 | `on_arrival(80, 7)` | `None` | `next_timer_ms()` is 90 |
| 84 | `on_arrival(84, 8)` | `None` | |
| 86 | `on_arrival(86, 9)` | `[7, 8, 9]` | the batch is full, 4 ms before the deadline |

The dispatches are `[0, 1]` at 10, `[2, 3, 4]` at 30, `[5, 6]` at 52, and `[7, 8, 9]` at 86. After the dispatch at 86, the queue is empty, so `next_timer_ms()` returns `None` and the timer for 90 never fires.

![The example's arrivals, a row each: a dot at the arrival, a line for the wait, a box for the batch. A tick marks a deadline passed while busy.](interactive:batch-window)

## Constraints

- `1 <= max_batch_size <= 64` and `0 <= max_wait_ms <= 10_000`. All times are integers.
- The tests record every dispatch (time and request IDs) and compare them exactly against literal schedules on small scenarios. On random bursty workloads of 150 requests, the tests check both exact agreement with the reference schedule and the policy invariants: every request is dispatched exactly once in arrival order in batches of at most `max_batch_size`, no request waits past its deadline at an idle model, a full batch never waits at an idle model, and a partial batch never leaves early.
- Returning a batch while the model is busy, returning a batch larger than `max_batch_size`, or dispatching a request that is not waiting fails the test.
- Do not use the real clock, `asyncio`, threads, or sleeping.

## Why this matters

A GPU runs a batch of eight prompts in little more time than a batch of one, so batching multiplies throughput on the same hardware. The trade-off is extra waiting latency, and production serving stacks expose these exact two knobs to control it. Triton uses `max_queue_delay_microseconds` and preferred batch sizes, TorchServe uses `maxBatchDelay` and `batchSize`, and Ray Serve uses `batch_wait_timeout_s` and `max_batch_size`.

Most batcher bugs live in the timer and boundary transitions. Keying the wait timer to the newest arrival instead of the oldest turns a steady trickle of traffic into unbounded latency for the first request. Failing to re-arm the timer after a flush strands leftover requests until the next arrival, producing rare multi-second outliers at low load that are difficult to reproduce.
