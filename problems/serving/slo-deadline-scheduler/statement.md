Give the continuous-batching loop a waiting queue that knows about deadlines: serve
the request whose first token is due soonest, and refuse, at once, a request whose
first token can no longer arrive in time. The loop is given; the work is one function
that decides, before every step, who is admitted, who is shed, and who waits.

## Specification

```python
def plan_step(waiting: list, now_ms: float, running: int, free_slots: int, step_ms): ...
async def run(engine, queue: asyncio.Queue, max_running: int, on_shed) -> None: ...
```

Requests are [`SimRequest`](../../../python/src/whetstone/kit/workloads.py) objects.
Time is virtual and counted in **milliseconds**. `request.deadline_s` is the request's
**TTFT deadline**: the absolute time in milliseconds by which its first output token
must appear, or `None` for a request without a deadline, which counts as a deadline
of $+\infty$. A first token at exactly the deadline is on time. The scheduler may read
`request.id`, `request.prompt_tokens`, and `request.deadline_s`, and not
`request.output_tokens`.

**The step being planned.** A step that prefills $P$ prompt tokens in total and
decodes $D$ sequences takes `step_ms(P, D)` milliseconds; in the tests of `run` this
is $8 + 0.25\,P + D$. Every request admitted to a step has its whole prompt prefilled
in it and gets its first token when the step ends. So the more prompts a step takes
in, the later all of its first tokens are.

**`plan_step(waiting, now_ms, running, free_slots, step_ms)`** plans the admissions of
the step that would start at `now_ms`, in which `running` sequences are decoded and
at most `free_slots` requests can be admitted. `waiting` holds the requests that have
no slot, in arrival order. The function goes through them **earliest deadline
first**; requests with equal deadlines (including several without one) are taken in
the order of `waiting`. It keeps a list `admit`, initially empty. For each request
$r$ in that order, let

$$
f = \text{now\_ms} + \text{step\_ms}\big(\text{prompt tokens of all of } \textit{admit} + \text{prompt tokens of } r,\ \text{running}\big),
$$

the time of $r$'s first token if it joined the step behind everything admitted so
far. Then exactly one of three things happens:

1. **Shed.** If $f$ is after $r$'s deadline, $r$ is shed. (It cannot be on time in
   this step, and a later step ends later still.)
2. **Admit.** Otherwise, if fewer than `free_slots` requests have been admitted and
   $f$ is not after the deadline of any request in `admit`, $r$ is appended to
   `admit`.
3. **Wait.** Otherwise $r$ stays in `waiting`.

The scan always goes through the whole list: a request that waits stops nothing
behind it, and requests are shed also when no slot is free. `plan_step` returns the
tuple `(admit, shed)`, two lists of requests in the order they were decided, and
removes exactly those requests from `waiting`, leaving the others in their order.

**The loop.** `run` is the loop of `continuous-batching-scheduler` with `plan_step`
in the place of first come, first served, and the starter contains it in full. Before
every step it moves the queue's contents to `waiting`, calls
`plan_step(waiting, loop.time(), len(running), max_running - len(running),
engine.cost.step_s)`, calls `on_shed(request)` for each shed request in order, admits
the others, and runs one step that prefills them whole, in order, and decodes every
sequence that was already running. If nothing is admitted and nothing is running,
there is no step, and the loop goes back to waiting for the queue.

What follows from this, spelled out:

- **No preemption.** A running sequence is never paused, evicted, or released,
  whatever is waiting. Deadlines decide admissions only.
- **Shedding happens when a step is planned**, not in between: a request that
  arrives during a step is first looked at when the step ends.
- **Nobody is served late.** Every request either gets its first token by its
  deadline or is shed without ever being admitted.
- A request that arrives at the exact instant a step ends is in the queue before the
  next step is planned.

## Example

`max_running = 2`. Requests as (arrival, prompt tokens, output tokens, deadline):
`A (0, 16, 3, 100)`, `B (2, 40, 2, 60)`, `C (4, 16, 2, 30)`, `D (6, 80, 2, 45)`,
`E (8, 32, 2, none)`.

| time | waiting, earliest deadline first | decisions | step |
|---|---|---|---|
| 0 | A | admit A: $f = 0 + 8 + 4 = 12 \le 100$ | prefill A (12 ms) |
| 12 | C (30), D (45), B (60), E | admit C: $f = 12 + 8 + 4 + 1 = 25 \le 30$. D: $f = 12 + 8 + 24 + 1 = 45 \le 45$, on time but the slots are full: waits. B, E: wait | prefill C, decode A (13 ms) |
| 25 | D (45), B (60), E | shed D: $f = 25 + 8 + 20 + 2 = 55 > 45$. B: $f = 25 + 8 + 10 + 2 = 45$: waits. E waits | decode A, C (10 ms); both finish |
| 35 | B (60), E | admit B: $f = 35 + 8 + 10 = 53 \le 60$. E: $f = 35 + 8 + 18 = 61$, fine for E but after B's deadline: waits | prefill B (18 ms) |
| 53 | E | admit E: $f = 53 + 8 + 8 + 1 = 70$ | prefill E, decode B (17 ms) |
| 70 | | | decode E (9 ms) |

C arrived after B and is served first. D is told at 25 that it will not make it,
20 ms before its deadline and without ever holding a slot. E, which has no deadline,
is kept out of the step at 35 because its prompt would have cost B its deadline.
First tokens: A 12, C 25, B 53, E 70.

## Constraints

- `1 <= max_running <= 8`; `0 <= free_slots`; `running >= 0`. Arrival times,
  deadlines, and step durations are multiples of 0.25 ms, so every comparison of $f$
  with a deadline is exact and ties are real.
- Request ids are unique and opaque (ints or strings, in no particular order).
- The tests call `plan_step` directly, also with other cost models passed as
  `step_ms`, and compare the steps of `run` and the `on_shed` calls (time, request,
  order) exactly: with literal schedules on small scenarios, and on seeded Poisson
  and bursty workloads of up to 80 requests with the reference and with the rules
  above.
- **SLO attainment** is counted as in `goodput-slo`: the requests whose first token
  came by their deadline, divided by all requests; a shed request is not good.
  - *Bursts.* 24 bursts of 12 requests, 700 ms apart, for four slots; half of the
    requests have 200 ms to their first token, half 1500 ms. Attainment must be at
    least **0.78**. (The reference reaches 0.84. First come, first served with the
    same shedding rule reaches 0.72; a queue that sheds nothing stays below one
    half.)
  - *Sustained overload.* 300 requests at 40 per second for four slots, about twice
    what they can serve; deadlines 150 ms or 1200 ms after arrival. Attainment must
    be at least **0.4** (the reference reaches 0.48), and no request may get its
    first token after its deadline.
- Use `asyncio` and the engine only; time is virtual.

## Why this matters

An interactive request that gets its first token after the user has given up is
worth nothing, and it is not free: it held a batch slot and KV memory for its whole
generation, and its prefill made every first token in its step later. Under overload
a first-come-first-served queue therefore does the worst possible thing. It serves
everybody late: the queue grows, each request waits longer than the last, and
throughput stays high while goodput goes to zero. Ordering by deadline alone does
not help once the load exceeds capacity, because the most urgent request is then
the one that is already lost. The two rules of this problem are what turns that
around: decide by deadline who goes next, and refuse early what cannot be saved,
so that a request is either answered in time or told immediately to go elsewhere,
and the capacity goes to requests that still count.
