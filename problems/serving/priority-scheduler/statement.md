Implement priority-based admission with aging for a continuous-batching scheduler. Urgent requests get slots first, and waiting requests improve by one priority level for every `aging_ms` milliseconds they wait so low-priority requests never starve.

![This problem is the Waiting queue, ordered by a priority that improves with waiting.](assets/map.svg)

## Specification

```python
def effective_priority(request, now_ms: float, aging_ms: float) -> int: ...
def pick_next(waiting: list, now_ms: float, aging_ms: float): ...
async def run(engine, queue: asyncio.Queue, max_running: int, aging_ms: float) -> None: ...
```

Requests are [`SimRequest`](../../../python/src/learnai/kit/workloads.py) objects. Virtual time is measured in **milliseconds**. Although the field name `request.arrival_s` ends in `_s`, its value is in milliseconds in this problem. `request.priority` is an integer where a **smaller value is more urgent**. Priorities may be negative.

**Effective priority.** At virtual time $t = \text{now\_ms}$, a request that arrived at `request.arrival_s` has waited $t - \text{request.arrival\_s}$ ms. For every full `aging_ms` it has waited, its priority improves by 1:

$$
\text{effective}(t) = \text{priority} - \left\lfloor \frac{t - \text{arrival}}{\text{aging\_ms}} \right\rfloor
$$

A request gains one level at the exact instant it has waited `aging_ms`, a second level at `2 * aging_ms`, and so on. There is no lower bound, so the effective priority can drop below every base priority submitted. `effective_priority` returns this value as an `int`.

**Selection.** `pick_next(waiting, now_ms, aging_ms)` removes and returns one request from the non-empty list `waiting`. The list `waiting` holds requests awaiting a slot in the order they were drained from `queue`. Choose the first request according to these criteria:

1. Smallest effective priority at `now_ms`.
2. Earliest `arrival_s` among ties (do not break ties by `request.id`, which is opaque).
3. Earliest position in `waiting` among requests with the same `arrival_s`.

**The loop.** The starter provides a complete `run` loop adapted from `continuous-batching-scheduler`. Before each step, it drains `queue` into `waiting`. While fewer than `max_running` sequences are running and `waiting` is non-empty, it admits `pick_next(waiting, now_ms, aging_ms)` using `now_ms = loop.time()` from the start of step planning. Every admission in the same step uses that same `now_ms`. The step prefills each admitted prompt in full (in admission order) and decodes all sequences that were already running. A step with $P$ prompt tokens and $D$ decoded sequences takes $8 + 0.25\,P + D$ ms.

Several rules follow from this loop:

- **Evaluate priorities at admission time.** Do not freeze priorities when a request arrives or leaves `queue`. Between two admissions, the relative order of waiting requests can change even if nothing new arrives.
- **Count waiting time from arrival.** A request that arrives during a step is collected when the step ends, and the time it spent in `queue` counts toward aging.
- **No preemption.** Priorities decide who receives a free slot. A running sequence is never paused or evicted, regardless of how urgent a waiting request is.
- **Same-instant arrivals.** Requests arriving at the same instant enter `queue` in arrival order, so rule 3 serves them first-delivered, first-served. A request that arrives at the exact instant a step ends is collected before the next step is planned.

You may restructure `run` (for example, using a heap or a class) as long as it produces the exact admissions and steps defined here.

## Example

Suppose `max_running = 1` and `aging_ms = 40`. A request with an 8-token prompt and 2 output tokens takes $8 + 0.25(8) = 10$ ms to prefill and $8 + 1 = 9$ ms to decode, holding the slot for 19 ms. Request 0 has 3 output tokens and holds the slot for $10 + 9 + 9 = 28$ ms.

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
| 0 | 0 (0, 1) | 0 | only request |
| 28 | 1 (26, 2), 2 (24, 0), 3 (22, 1), 4 (22, 1) | 2 | smallest effective priority |
| 47 | 1 (45, 1), 3 (41, 0), 4 (41, 0) | 3 | 3 and 4 gained a level at 46 and tie on arrival; 3 is earlier in `waiting` |
| 66 | 1 (64, 1), 4 (60, 0), 5 (6, 0) | 4 | 4 and 5 both have level 0; 4 arrived earlier |
| 85 | 1 (83, 0), 5 (25, 0) | 1 | 1 gained its second level at 82 and arrived before 5 |
| 104 | 5 (44, -1) | 5 | only remaining request |

The admitted requests are 0, 2, 3, 4, 1, 5 at times 0, 28, 47, 66, 85, 104. Without aging, request 1 (priority 2) would run last. With aging, it reaches level 0 after 80 ms of waiting and passes request 5, which arrived 58 ms later at priority 0.

## Constraints

- `1 <= max_running <= 8`. `aging_ms` is a positive number. Arrival times, `aging_ms`, and step durations are multiples of 0.25 ms, so `now_ms - arrival_s` and the floor division are exact in floating point.
- Priorities are integers in $[-8, 8]$. Request ids are unique and opaque (ints or strings in no particular order).
- The tests call `effective_priority` and `pick_next` directly, and compare the admissions (time and id) and steps of `run` exactly on small scenarios and on seeded Poisson and bursty workloads of up to 100 requests.
- **Starvation bound.** In the stress test, 320 requests arrive at 21 per second across 4 slots, slightly above capacity. One in eight has priority 3 and the rest have priority 0, with `aging_ms = 200`. The longest wait for a priority-3 request must be at most **2000 ms** (the reference reaches 1320 ms, while scheduling without aging reaches 4946 ms).
- Use `asyncio` and the engine only. Time is virtual.

## Why this matters

Serving stacks use priority queues so interactive traffic stays responsive while batch jobs, evaluations, and background tasks share the same GPUs. For example, vLLM's `--scheduling-policy priority` orders its waiting queue by `priority` (lower first) and then by arrival time, matching rules 1 and 2 without aging.

Strict priority causes starvation under load. As long as urgent requests arrive faster than slots free up, a low-priority request is never picked and eventually times out. Aging, borrowed from operating-system schedulers, replaces unbounded waits with a delay proportional to `aging_ms` times the priority gap. Because effective priority changes with the clock, subtle bugs arise whenever a scheduler caches stale priorities or miscomputes aging boundaries.
