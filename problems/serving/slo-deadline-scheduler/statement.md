Implement deadline-aware admission control for a continuous-batching scheduler. Before each step, decide which waiting requests to admit, which to shed because they can no longer meet their deadline, and which to leave waiting.

![This problem is the Waiting queue, served earliest deadline first with late requests refused.](assets/map.svg)

## Specification

```python
def plan_step(waiting: list, now_ms: float, running: int, free_slots: int, step_ms): ...
async def run(engine, queue: asyncio.Queue, max_running: int, on_shed) -> None: ...
```

Requests are [`SimRequest`](../../../python/src/learnai/kit/workloads.py) objects, and virtual time is measured in **milliseconds**. Each request has a time-to-first-token (**TTFT**) deadline in `request.deadline_s`, which is the absolute time in milliseconds by which its first output token must appear. A value of `None` means no deadline ($+\infty$). A first token that arrives at the exact deadline is on time. The scheduler may read `request.id`, `request.prompt_tokens`, and `request.deadline_s`, and must not read `request.output_tokens`.

**Step cost model.** A step that prefills $P$ total prompt tokens and decodes $D$ running sequences takes `step_ms(P, D)` milliseconds. In the tests for `run`, this cost is $8 + 0.25\,P + D$. Every request admitted to a step has its entire prompt prefilled in that step and receives its first token when the step ends. Adding more prompts to a step therefore delays the first token of every request admitted in that step.

**`plan_step(waiting, now_ms, running, free_slots, step_ms)`** plans admissions for the step starting at `now_ms`, when `running` sequences are already being decoded and at most `free_slots` new requests can be admitted. `waiting` holds the requests that do not have a slot, in arrival order.

Process the requests in `waiting` **earliest deadline first**. Break ties in deadline (including multiple requests with `None`) by their order in `waiting`. Maintain two lists, `admit` and `shed`, both initially empty. For each request $r$ in that order, let $P_{\text{admit}}$ be the total `prompt_tokens` of all requests already in `admit`, and let $P_r$ be `r.prompt_tokens`. Compute the time of $r$'s first token if it joins the step after everything already admitted:

$$
f = \text{now\_ms} + \text{step\_ms}(P_{\text{admit}} + P_r,\ \text{running}).
$$

Then apply exactly one of three rules:

1. **Shed.** If $f$ is strictly after $r$'s deadline, append $r$ to `shed`. It cannot arrive on time in this step, and any later step finishes even later.
2. **Admit.** Otherwise, if `len(admit) < free_slots` and $f$ is not after the deadline of any request already in `admit`, append $r$ to `admit`.
3. **Wait.** Otherwise, leave $r$ in `waiting`.

Always scan the entire list. A request that waits does not stop later requests from being admitted or shed, and hopeless requests are shed even when `free_slots` is 0. Remove every admitted and shed request from `waiting` in place, keeping the remaining requests in their original order. Return `(admit, shed)`, with each list in the order its requests were decided.

**The loop.** The starter already provides `run` in full. It uses the same continuous-batching loop as `continuous-batching-scheduler`, replacing first-come-first-served admission with `plan_step`. Before each step, it drains `queue` into `waiting` and calls `plan_step(waiting, loop.time(), len(running), max_running - len(running), engine.cost.step_s)`. It then calls `on_shed(request)` for each shed request in order, admits the chosen requests, and runs one step that prefills them in order while decoding all currently running sequences. If nothing is admitted and nothing is running, no step runs and the loop waits for the next item from `queue`.

These rules imply four invariants:

- **No preemption.** A running sequence is never paused, evicted, or released, regardless of what is waiting. Deadlines control admissions only.
- **Step-boundary shedding.** Shedding happens only when a step is planned. A request that arrives during a step is first inspected when that step ends.
- **No late first tokens.** Every request either receives its first token by its deadline or is shed without ever being admitted.
- **Boundary arrivals.** A request that arrives at the exact instant a step ends enters the queue before the next step is planned.

## Example

Let `max_running = 2` and `step_ms(P, D) = 8 + 0.25 * P + D`. Five requests arrive, written as `(arrival, prompt_tokens, output_tokens, deadline)`:

- `A (0, 16, 3, 100)`
- `B (2, 40, 2, 60)`
- `C (4, 16, 2, 30)`
- `D (6, 80, 2, 45)`
- `E (8, 32, 2, None)`

| time (ms) | waiting (EDF order) | decisions | step |
|---|---|---|---|
| 0 | A | Admit A ($f = 0 + 8 + 4 = 12 \le 100$). | prefill A (12 ms) |
| 12 | C (30), D (45), B (60), E | Admit C ($f = 12 + 8 + 4 + 1 = 25 \le 30$). D gives $f = 12 + 8 + 24 + 1 = 45 \le 45$, which is on time, but no slot is free, so D waits. B and E wait. | prefill C, decode A (13 ms) |
| 25 | D (45), B (60), E | Shed D ($f = 25 + 8 + 20 + 2 = 55 > 45$). B gives $f = 25 + 8 + 10 + 2 = 45 \le 60$, so B waits. E waits. | decode A, C (10 ms); both finish |
| 35 | B (60), E | Admit B ($f = 35 + 8 + 10 = 53 \le 60$). E gives $f = 35 + 8 + 18 = 61$, which is fine for E but exceeds B's deadline of 60, so E waits. | prefill B (18 ms) |
| 53 | E | Admit E ($f = 53 + 8 + 8 + 1 = 70$). | prefill E, decode B (17 ms) |
| 70 | | | decode E (9 ms) |

C arrives after B and is served first. D is shed at 25 ms, which is 20 ms before its deadline and without ever holding a slot. E has no deadline, but it is held back at 35 ms because its prompt would push B past its deadline. The first tokens appear at A: 12 ms, C: 25 ms, B: 53 ms, and E: 70 ms.

## Constraints

- `1 <= max_running <= 8`, `0 <= free_slots`, and `running >= 0`.
- Arrival times, deadlines, and step durations are multiples of 0.25 ms, so every comparison of $f$ with a deadline is exact.
- Request IDs are unique and opaque (ints or strings, in arbitrary order).
- Tests call `plan_step` directly with various `step_ms` cost models. They also compare the steps of `run` and the `on_shed` calls (time, request, and order) exactly against literal schedules on small scenarios and against the reference on seeded Poisson and bursty workloads of up to 80 requests.
- **SLO attainment** is defined as in `goodput-slo`, namely the fraction of all requests whose first token arrives by their deadline. Shed requests count as misses.
  - *Bursts.* 24 bursts of 12 requests arrive 700 ms apart for 4 slots, with half of the requests having a 200 ms TTFT budget and half having a 1500 ms budget. Attainment must be at least **0.78**. The reference reaches 0.84, first-come-first-served with the same shedding rule reaches 0.72, and a queue without shedding stays below 0.50.
  - *Sustained overload.* 300 requests arrive at 40 per second for 4 slots (about twice capacity), with deadlines 150 ms or 1200 ms after arrival. Attainment must be at least **0.40** (the reference reaches 0.48), and no request may receive its first token after its deadline.
- Use `asyncio` and the engine only, since time is virtual.

## Why this matters

An interactive request that receives its first token after its deadline delivers no value, yet it still consumes a batch slot and KV memory for its entire generation. Worse, its prefill delays the first token of every other request sharing that step.

Under overload, a first-come-first-served queue serves nearly every request late. The queue grows, wait times compound, and raw throughput stays high while goodput drops toward zero. Earliest-deadline-first ordering alone fails under overload as well, because the tightest deadline belongs to a request that is already doomed. Combining earliest-deadline-first ordering with early shedding fixes both failure modes by refusing unsaveable requests immediately and devoting capacity to requests that can still finish on time.
