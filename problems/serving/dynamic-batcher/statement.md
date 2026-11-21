Implement the policy that decides when an inference server runs a batch: wait a little
so that more requests can share one forward pass, but never longer than a latency
budget allows. The policy runs on a simulated clock, and the difficulty is in the
edges: what exactly happens at a deadline, at a full queue, and at the moment the model
becomes free.

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

The batcher is a state machine that a simulator drives. It never sleeps, starts no
threads, and reads no clock: every call tells it the current virtual time `now_ms`, an
integer number of milliseconds that never decreases from one call to the next.

**The model** runs one batch at a time. It is *busy* from the moment a handler returns
a batch until the simulator calls `on_batch_done`, and *idle* otherwise (it starts
idle). How long a batch runs is up to the simulator; the batcher is not told in
advance.

**The queue** holds the requests that have arrived and have not been dispatched, in
arrival order (the order of the `on_arrival` calls). The *deadline* of a request is its
arrival time plus `max_wait_ms`.

**The dispatch rule.** After every event, a batch is dispatched if and only if all of
these hold:

1. the model is idle,
2. the queue is not empty, and
3. the queue holds at least `max_batch_size` requests, **or** `now_ms` is greater than
   or equal to the deadline of the oldest request in the queue.

The batch consists of the oldest `min(max_batch_size, queue length)` requests, oldest
first. They leave the queue; younger requests stay and keep their own deadlines.

**The handlers.** Each one updates the state for its event, applies the dispatch rule
once, and returns the batch as a list of request ids, or `None` if the rule does not
fire. (An empty list is treated like `None`.)

- `on_arrival(now_ms, request_id)`: a request arrived and joins the end of the queue.
  `request_id` is an opaque hashable value, unique per request.
- `on_batch_done(now_ms)`: the running batch finished; the model is idle again.
- `on_timer(now_ms)`: the wake-up the batcher asked for has come. Nothing else changed.

**The timer.** The simulator cannot guess when a deadline falls, so after every handler
call it calls `next_timer_ms()` and the batcher answers with the virtual time at which
it wants `on_timer` called, or `None`. Only the most recent answer counts: there is one
timer, and each answer replaces the previous one. An answer that is not later than the
current time is ignored (no call results from it). `on_timer(T)` is called when the
clock reaches a pending answer `T`, also when the model is busy at that moment. A
wake-up at which the rule does not fire is harmless; a deadline that passes at an idle
model with no wake-up requested is a missed dispatch.

**Events at the same instant** are delivered one at a time in a fixed order, and the
dispatch rule is applied after each of them:

1. the batch completion, if the running batch finishes at this instant;
2. the arrivals of this instant, in arrival order;
3. the timer, if it is still pending for this instant after the calls above.

The cases that follow from these rules, spelled out:

- **Arrival exactly at the oldest request's deadline.** The arrival is delivered
  before the timer, and after it the rule fires (the model is idle and the deadline
  has been reached), so `on_arrival` returns a batch that includes the new request.
- **Arrivals while the model is busy** join the queue. Nothing is dispatched, however
  many are waiting and however long they have waited.
- **Batch completion.** If at least `max_batch_size` requests are waiting,
  `on_batch_done` returns the oldest `max_batch_size` of them. Otherwise, if the oldest
  request's deadline has already been reached (it expired while the model was busy, or
  falls exactly on the completion time), `on_batch_done` returns everything that is
  waiting. Otherwise it returns `None`, and the waiting requests are dispatched when
  the oldest one's deadline comes or the queue fills up, whichever is first. Their
  wait is not restarted.
- **Arrival at the same instant as a batch completion.** The completion is delivered
  first. If `on_batch_done` dispatches, the new arrival finds the model busy and waits
  for the next batch.
- **Several arrivals at one instant** are separate events. If the second of four
  simultaneous arrivals fills the batch, the batch leaves in that call, and the other
  two find the model busy.
- **`max_wait_ms = 0`**: a request that arrives at an idle model is dispatched in its
  own `on_arrival` call.

## Example

`max_batch_size = 3`, `max_wait_ms = 10`, and the simulator's model takes `16 + 2n` ms
for a batch of `n` requests. Requests `0` to `9` arrive at times
`0, 4, 25, 26, 27, 28, 29, 80, 84, 86`.

| time | call | returns | why |
|---|---|---|---|
| 0 | `on_arrival(0, 0)` | `None` | deadline of request 0 is 10; `next_timer_ms()` is 10 |
| 4 | `on_arrival(4, 1)` | `None` | two waiting, not full |
| 10 | `on_timer(10)` | `[0, 1]` | the oldest request reached its deadline; runs until 30 |
| 25 to 29 | `on_arrival` for 2, 3, 4, 5, 6 | `None` | the model is busy |
| 30 | `on_batch_done(30)` | `[2, 3, 4]` | five waiting: the oldest three go; runs until 52 |
| 52 | `on_batch_done(52)` | `[5, 6]` | request 5's deadline (38) passed while busy; runs until 72 |
| 72 | `on_batch_done(72)` | `None` | nothing is waiting |
| 80 | `on_arrival(80, 7)` | `None` | `next_timer_ms()` is 90 |
| 84 | `on_arrival(84, 8)` | `None` | |
| 86 | `on_arrival(86, 9)` | `[7, 8, 9]` | the batch is full, 4 ms before the deadline |

The dispatches are `[0, 1]` at 10, `[2, 3, 4]` at 30, `[5, 6]` at 52, and `[7, 8, 9]`
at 86. The timer set for 90 never fires: after the dispatch at 86 there is nothing to
wait for, and `next_timer_ms()` returns `None`.

![The example's arrivals, a row each: a dot at the arrival, a line for the wait, a box for the batch. A tick marks a deadline passed while busy.](interactive:batch-window)

## Constraints

- `1 <= max_batch_size <= 64` and `0 <= max_wait_ms <= 10_000`. All times are integers.
- The tests record every dispatch (time and request ids) and compare them exactly:
  with literal schedules on small scenarios, and on random bursty workloads of 150
  requests both with the reference's schedule and with the invariants of the policy
  (every request dispatched exactly once, in arrival order, in batches of at most
  `max_batch_size`; no request waits past its deadline at an idle model; a full batch
  never waits at an idle model; a partial batch never leaves early).
- Returning a batch while the model is busy, a batch larger than `max_batch_size`, or
  a request that is not waiting fails the case.
- Do not use the real clock, `asyncio`, threads, or sleeping.

## Why this matters

A GPU runs a batch of eight prompts in little more time than a batch of one, so a
server that batches serves several times the traffic on the same hardware. The price
is the time requests spend waiting for company, and the two knobs here are how every
serving stack exposes that trade: Triton's dynamic batcher has
`max_queue_delay_microseconds` and preferred batch sizes, TorchServe has `maxBatchDelay`
and `batchSize`, and Ray Serve has `batch_wait_timeout_s` and `max_batch_size`. The
bugs are in the edges this problem pins down. A timer keyed to the newest request
turns a steady trickle of traffic into unbounded latency for the oldest one. A timer
that is not re-armed after a flush strands the leftover requests until the next
arrival, which shows up in production as rare multi-second outliers at low load, the
hardest kind of latency bug to reproduce.
