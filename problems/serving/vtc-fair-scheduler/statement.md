Add multi-tenant fairness to a continuous-batching scheduler using the Virtual Token Counter (VTC) algorithm. Each tenant maintains a counter of the weighted tokens it has received, and each free slot goes to the queued tenant with the smallest counter. The starter code already implements a working first-come, first-served scheduler, so you only need to add the three VTC counter rules.

![This problem is the Waiting queue, where the tenant with the smallest counter goes first.](assets/map.svg)

## Specification

```python
class VTCScheduler:
    def __init__(self, engine, max_running: int, w_in: float = 1.0, w_out: float = 2.0): ...
    counters: dict            # tenant -> virtual token counter
    def submit(self, request) -> None: ...
    async def run(self) -> None: ...
```

`engine` is a [`SimEngine`](../../../python/src/learnai/kit/simengine.py) with `max_running` slots. `w_in` and `w_out` are the weights charged per input (prompt) token and per generated output token. Incoming requests are [`SimRequest`](../../../python/src/learnai/kit/workloads.py) objects, where `request.tenant` is a string identifying the sender and `request.prompt_tokens` is the prompt length. Time is virtual and measured in **milliseconds**. A step over $P$ prompt tokens and $D$ decoded sequences takes $8 + 0.25\,P + D$ ms. Do not read `request.output_tokens`, because a real server does not know output lengths in advance.

Implement Algorithm 2 from Sheng et al., "Fairness in Serving Large Language Models" (OSDI 2024). Let $Q$ be the queue of submitted, unadmitted requests in arrival order. A tenant is *queued* if it has at least one request in $Q$. Requests that are already running do not make a tenant queued. Let $c_t$ denote `counters[t]`. A tenant is added to `counters` with value `0.0` on its first `submit` call and is never removed.

**Arrival (`submit(request)`).** The test harness calls `submit` at the request's arrival instant from a plain callback inside the event loop. Let $u$ be `request.tenant`:

1. If $u$ is not currently queued, apply the **counter lift**:
   - If $Q$ is non-empty, set $c_u \leftarrow \max\bigl(c_u,\ \min\{c_i : i \text{ is queued}\}\bigr)$.
   - If $Q$ is empty, set $c_u \leftarrow \max(c_u,\ c_l)$, where $l$ is the tenant of the most recently admitted request. If no request has ever been admitted, leave $c_u$ unchanged.
2. Append `request` to $Q$ and wake the scheduling loop.

**Admission (`run()`).** The starter provides the continuous-batching loop from `continuous-batching-scheduler`. Before each step, while fewer than `max_running` sequences are running and $Q$ is non-empty, admit one request. The step then prefills all newly admitted prompts whole in admission order and decodes every previously running sequence. Choose each admitted request as follows:

- Select the queued tenant $k$ with the smallest counter $c_k$. If multiple queued tenants tie for the smallest counter, pick the one whose earliest request appears first in $Q$.
- Remove and admit the earliest request of tenant $k$ in $Q$.
- Immediately charge its prompt tokens before making the next pick in the same step:
  $$
  c_k \leftarrow c_k + w_\text{in} \cdot \text{prompt\_tokens}.
  $$

**Generation.** When `engine.step` returns, iterate through every sequence id in `result.tokens`. That list contains every sequence that produced an output token in the step, including the first token produced by a prefill step. For each sequence's tenant $t$, add $w_\text{out}$ to $c_t$:
$$
c_t \leftarrow c_t + w_\text{out}.
$$

Input tokens are charged at admission, and output tokens are charged one at a time as they are generated. Nothing else modifies a counter. Counters are never reset and never decrease.

**Events at the same instant.** `submit` runs at the arrival instant even when a step is in flight, so a counter lift observes the counters as they stand then, before the in-flight step's output charges are applied. A request arriving at the exact instant a step finishes is submitted before that step's output tokens are charged. Requests arriving at the same instant are submitted in arrival order.

## Example

Let `max_running = 1`, `w_in = 1`, and `w_out = 2`. Every request has an 8-token prompt and 2 output tokens, so each request holds the slot for 19 ms and costs its tenant $8 + 2 \cdot 2 = 12$. Tenant `a` sends requests 0, 1, and 2 at time 0, and requests 6 and 7 at time 61. Tenant `b` sends request 3 at time 5, and requests 4 and 5 at time 60.

| time | event | $c_a$ | $c_b$ |
|---|---|---|---|
| 0 | 0, 1, 2 submitted; request 0 admitted (+8) | 8 | |
| 5 | 3 submitted; `b` is new to $Q$: lifted to $\min\{c_a\} = 8$ | 8 | 8 |
| 10, 19 | two tokens of request 0 (+2 each) | 12 | 8 |
| 19 | `b` has the smallest counter: request 3 admitted (+8) | 12 | 16 |
| 38 | request 3 done (+4). `a` is smaller: request 1 admitted | 20 | 20 |
| 57 | request 1 done. Only `a` is queued: request 2 admitted | 32 | 20 |
| 60 | 4, 5 submitted. $Q$ is empty and `a` was admitted last: `b` lifted to 32 | 32 | 32 |
| 61 | 6, 7 submitted; `a` is not queued: $\max(32, \min\{c_b\}) = 32$ | 32 | 32 |
| 76 | request 2 done ($c_a = 36$). `b` is smaller: request 4 admitted | 36 | 40 |
| 95 | request 4 done. `a` is smaller: request 6 admitted | 44 | 44 |
| 114 | request 6 done ($c_a = 48$). `b` is smaller: request 5 admitted | 48 | 52 |
| 133 | request 5 done. Request 7 admitted | 56 | 56 |
| 152 | request 7 done | 60 | 56 |

![The counters of the example. Tenant `b` sends nothing from 38 to 60 ms and returns lifted from 20 to 32, the counter of tenant `a`.](assets/counters.svg)

The resulting admission order is 0, 3, 1, 2, 4, 6, 5, 7. Between 38 ms and 60 ms, tenant `b` is idle while tenant `a` continues to be served. Without the counter lift at 60 ms, tenant `b` would return at 20 against tenant `a`'s 32 and both of its requests would be admitted before request 6. With the lift, unused capacity is not saved as credit, and the two tenants alternate.

## Constraints

- `1 <= max_running <= 8`. Weights are positive and exactly representable in binary floating point (such as 0.5, 1, 2, 3, or 4), so all counters are exact floats. Seeded workloads have up to four tenants.
- Every admission is deterministic. The tests read `scheduler.counters` at chosen instants and at the end of each run, and compare admissions (time and id), engine steps, and counters exactly against literal values on small scenarios and against the reference on seeded Poisson and bursty workloads of up to 100 requests.
- **Fairness bound (Theorem 4.4 of the paper).** Define the service $W_f(t_1, t_2)$ received by tenant $f$ during $[t_1, t_2)$ as $w_\text{in}$ times the prompt tokens of its requests admitted in that interval plus $w_\text{out}$ times its output tokens generated in that interval. Let $L_\text{input}$ be the longest prompt, $L_\text{output}$ be the longest output, and $M = \text{max\_running} \cdot (L_\text{input} + L_\text{output})$ be the maximum tokens in a running batch. For any two tenants $f$ and $g$ that remain queued throughout $[t_1, t_2)$,

  $$
  |W_f(t_1, t_2) - W_g(t_1, t_2)| \le 2 \max(w_\text{in} \cdot L_\text{input},\ w_\text{out} \cdot M).
  $$

  The tests verify this bound from the engine trace on every seeded workload, including a three-tenant backlog of 270 requests.
- **Work conservation and per-tenant order.** No slot may be left free while $Q$ is non-empty, and requests from the same tenant must be admitted in arrival order.
- Use `asyncio` and the provided engine only, since time is virtual.

## Why this matters

A shared LLM endpoint is a common resource. First-come, first-served scheduling hands capacity to whoever sends the most traffic, so one tenant running a batch job fills the queue and inflates latency for everyone else. The standard alternative, a per-tenant requests-per-minute limit, is both too rigid and too coarse. It rejects traffic even when GPUs are idle, and it counts a 20-token question the same as a 4,000-token document with a long answer.

VTC brings fair queueing from packet networks to token serving. It never rejects work that fits, measures actual weighted token consumption, and updates counters as tokens are generated so it never needs to know output lengths upfront. The counter lift ensures that the constant fairness bound between backlogged tenants still holds when tenants go idle and return.
