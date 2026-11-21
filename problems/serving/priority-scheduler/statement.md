Give the continuous-batching loop a waiting queue that serves urgent requests first
and still never starves anyone: a request's priority improves by one level for every
`aging_ms` it has waited. The loop is given; the work is the selection rule, whose
answer changes with time.

![This problem is the Waiting queue, ordered by a priority that improves with waiting.](assets/map.svg)

## Specification

```python
def effective_priority(request, now_ms: float, aging_ms: float) -> int: ...
def pick_next(waiting: list, now_ms: float, aging_ms: float): ...
async def run(engine, queue: asyncio.Queue, max_running: int, aging_ms: float) -> None: ...
```

Requests are [`SimRequest`](../../../python/src/learnai/kit/workloads.py) objects.
Time is virtual and counted in **milliseconds**; `request.arrival_s` is the arrival
time in that unit (the kit's field names end in `_s`, the numbers here are
milliseconds). `request.priority` is an integer, and **a smaller value is more
urgent**. Priorities may be negative.

**Effective priority.** At time `now_ms`, a request that arrived at `arrival_s` has
waited `now_ms - arrival_s` and its effective priority is

$$
\text{effective}(t) = \text{priority} - \left\lfloor \frac{t - \text{arrival}}{\text{aging\_ms}} \right\rfloor .
$$

A request gains a level at the instant it has waited exactly `aging_ms`, another at
`2 * aging_ms`, and so on, without a lower bound: the result may fall below every
priority that is ever submitted. `effective_priority` returns this value as an `int`.

**Selection.** `pick_next(waiting, now_ms, aging_ms)` removes one request from the
non-empty list `waiting` and returns it. `waiting` holds the requests that have no slot
yet, in the order they left the queue. The request chosen is the first in this order:

1. the smallest effective priority at `now_ms`;
2. among equals, the earliest `arrival_s` (not the smallest id: ids are opaque);
3. among requests that arrived at the same instant, the one earliest in `waiting`.

**The loop.** `run` is the loop of `continuous-batching-scheduler` with one change, and
the starter contains it in full. Before every step it collects the queue into
`waiting`, and then, while fewer than `max_running` sequences are running and a
request is waiting, it admits `pick_next(waiting, now_ms, aging_ms)`, where `now_ms`
is the virtual time at which the step is being planned (`loop.time()`). All admissions
of one step use the same `now_ms`. The step prefills the admitted prompts whole, in
admission order, and decodes every sequence that was already running. A step over $P$
prompt tokens and $D$ decoded sequences takes $8 + 0.25\,P + D$ ms.

What follows from this, spelled out:

- **Priorities are evaluated when a slot is handed out**, not when a request arrives
  and not when it is collected. Between two admissions the order of the waiting
  requests can change although nothing arrived.
- **Waiting time counts from the arrival.** A request that arrives during a step is
  collected when the step ends; the time it spent in the queue counts.
- **No preemption.** Priorities decide who gets a free slot. A running sequence is
  never paused or evicted, however urgent the waiting requests are.
- **Same-instant arrivals** are put into the queue in arrival order, so rule 3 is
  "first delivered, first served". A request that arrives at the exact instant a step
  ends is collected before the next step is planned.

You may restructure the given loop (a heap, a class), provided the admissions and
steps are the ones defined here.

## Example

`max_running = 1`, `aging_ms = 40`. A request with an 8-token prompt and 2 output
tokens holds the slot for 10 + 9 = 19 ms; request 0 has 3 output tokens and holds it
for 28 ms.

| request | arrival | priority |
|---|---|---|
| 0 | 0 | 1 |
| 1 | 2 | 2 |
| 2 | 4 | 0 |
| 3 | 6 | 1 |
| 4 | 6 | 1 |
| 5 | 60 | 0 |

| time | waiting: id (waited, effective priority) | admitted | why |
|---|---|---|---|
| 0 | 0 (0, 1) | 0 | the only request |
| 28 | 1 (26, 2), 2 (24, 0), 3 (22, 1), 4 (22, 1) | 2 | smallest priority |
| 47 | 1 (45, 1), 3 (41, 0), 4 (41, 0) | 3 | 3 and 4 gained a level at 46; they tie on arrival, and 3 is first in the queue |
| 66 | 1 (64, 1), 4 (60, 0), 5 (6, 0) | 4 | level 0 twice; 4 arrived earlier |
| 85 | 1 (83, 0), 5 (25, 0) | 1 | request 1 gained its second level at 82 and arrived before 5 |
| 104 | 5 (44, -1) | 5 | |

The admissions are 0, 2, 3, 4, 1, 5 at times 0, 28, 47, 66, 85, 104. Without aging,
request 1 (priority 2) would be last; with it, request 1 reaches level 0 after 80 ms of
waiting and passes request 5, a priority-0 request that arrived 58 ms after it.

## Constraints

- `1 <= max_running <= 8`. `aging_ms` is a positive number. Arrival times, `aging_ms`,
  and step durations are multiples of 0.25 ms, so `now_ms - arrival_s` and the floor
  are exact in floating point.
- Priorities are integers in $[-8, 8]$. Request ids are unique and opaque (ints or
  strings, in no particular order).
- The tests call `effective_priority` and `pick_next` directly, and compare the
  admissions (time and id) and the steps of `run` exactly: with literal schedules on
  small scenarios, and on seeded Poisson and bursty workloads of up to 100 requests
  with the selection rule above and with the reference.
- **Starvation bound.** 320 requests arrive at 21 per second for four slots, slightly
  more than the slots can serve; one in eight has priority 3 and the rest priority 0;
  `aging_ms = 200`. The longest time a priority-3 request waits for admission must be
  at most **2000 ms**. (The reference reaches 1320 ms; without aging it is 4946 ms.)
- Use `asyncio` and the engine only; time is virtual.

## Why this matters

Priority queues are how a serving stack keeps interactive traffic fast while batch
jobs, evaluations, and background summarization share the same GPUs: vLLM's
`--scheduling-policy priority` orders its waiting queue by the request's `priority`
(lower first) and then by arrival time, exactly rules 1 and 2 without the aging. The
price of strict priorities is starvation. As long as urgent requests keep arriving
faster than slots free up, a low-priority request is never the best candidate, and its
client times out after minutes of silence, usually during the traffic peak when
someone is watching. Aging is the standard fix from operating-system schedulers: it
turns "never" into a bound proportional to `aging_ms` times the priority gap, at the
cost of a selection rule that depends on the clock, which is where the bugs are.
