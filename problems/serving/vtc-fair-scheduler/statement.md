Make a continuous-batching scheduler fair between tenants with the Virtual Token
Counter (VTC) algorithm: each tenant has a counter of the service it has received, in
weighted tokens, and a free slot goes to the queued tenant with the smallest counter.
The starter is a working first-come-first-served scheduler; the work is the three
counter rules, of which the least obvious one keeps an idle tenant from saving up
credit.

## Specification

```python
class VTCScheduler:
    def __init__(self, engine, max_running: int, w_in: float = 1.0, w_out: float = 2.0): ...
    counters: dict            # tenant -> virtual token counter
    def submit(self, request) -> None: ...
    async def run(self) -> None: ...
```

`engine` is a [`SimEngine`](../../../python/src/learnai/kit/simengine.py) with
`max_running` slots. Requests are
[`SimRequest`](../../../python/src/learnai/kit/workloads.py) objects;
`request.tenant` is a string naming who sent it. Time is virtual and counted in
**milliseconds**; a step over $P$ prompt tokens and $D$ decoded sequences takes
$8 + 0.25\,P + D$ ms. Do not read `request.output_tokens`: a server does not know how
long an output will be.

The algorithm is Algorithm 2 of Sheng et al., "Fairness in Serving Large Language
Models" (OSDI 2024). $Q$ is the queue of requests that have been submitted and not
admitted, in arrival order. A tenant is *queued* if it has a request in $Q$; requests
that are already running do not count. `counters[t]` is $c_t$, and a tenant enters
`counters` with `0.0` at its first `submit`.

**Arrival: `submit(request)`** is called by the test at the request's arrival instant,
from a plain callback inside the event loop. With $u$ the request's tenant:

1. If $u$ is not queued, apply the **counter lift**:
   - if $Q$ is not empty: $c_u \leftarrow \max\bigl(c_u,\ \min\{c_i : i \text{ queued}\}\bigr)$;
   - if $Q$ is empty: $c_u \leftarrow \max(c_u,\ c_l)$, where $l$ is the tenant of the
     request that left $Q$ most recently (the last one admitted). If no request has
     ever been admitted, $c_u$ is unchanged.
2. Append the request to $Q$ and wake the loop.

**Admission.** `run()` is the loop of `continuous-batching-scheduler`, given in the
starter: before every step, while fewer than `max_running` sequences are running and
$Q$ is not empty, one request is admitted; the step prefills the admitted prompts
whole, in admission order, and decodes every sequence that was already running. The
request to admit is chosen as follows:

- $k$ is the queued tenant with the smallest counter. Among tenants with equal
  counters, $k$ is the one whose earliest request comes first in $Q$.
- The request is the earliest request of $k$ in $Q$.
- Immediately, before the next choice of the same step:
  $c_k \leftarrow c_k + w_\text{in} \cdot \text{prompt\_tokens}$.

**Generation.** When a step returns, for every id in `result.tokens` (the sequences
that produced one output token in that step, the first token from the prefill step
included): $c_t \leftarrow c_t + w_\text{out}$ for the sequence's tenant $t$.

So input tokens are charged when a request is admitted and output tokens one at a
time as they are generated. Nothing else changes a counter: counters are never reset
and never decrease.

**Events at one instant.** `submit` runs at the arrival instant even when a step is in
flight, so a lift sees the counters as they are then, without the charges of the step
that is still running. A request that arrives at the exact instant a step ends is
submitted before that step's output tokens are charged. Requests that arrive at the
same instant are submitted in arrival order.

## Example

`max_running = 1`, `w_in = 1`, `w_out = 2`. Every request has an 8-token prompt and 2
output tokens: it holds the slot for 19 ms and costs its tenant 8 + 2 · 2 = 12.
Tenant `a` sends requests 0, 1, 2 at time 0 and requests 6, 7 at 61. Tenant `b` sends
request 3 at 5 and requests 4, 5 at 60.

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

The admissions are 0, 3, 1, 2, 4, 6, 5, 7. Between 38 and 60 tenant `b` sent nothing
while `a` kept being served. Without the lift at 60, `b` would return with a counter of
20 against 32 and both of its requests would be admitted before request 6; with it,
the tenants alternate. Unused share is not saved up.

## Constraints

- `1 <= max_running <= 8`; weights are positive and exactly representable (such as
  0.5, 1, 2, 3, 4), so counters are exact floats. Up to four tenants in the seeded
  workloads.
- The algorithm determines every admission. The tests read `scheduler.counters` at
  chosen instants and at the end, and compare admissions (time and id), engine steps,
  and counters exactly: with literal values on small scenarios, and with the reference
  on seeded Poisson and bursty workloads of up to 100 requests.
- **Fairness bound** (Theorem 4.4 of the paper). The service a tenant receives in
  $[t_1, t_2)$, $W(t_1, t_2)$, is $w_\text{in}$ times the prompt tokens of its requests
  admitted in the interval plus $w_\text{out}$ times its output tokens generated in
  it. For any two tenants $f$ and $g$ and any interval in which both are queued
  throughout,

  $$
  |W_f(t_1, t_2) - W_g(t_1, t_2)| \le 2 \max(w_\text{in} \cdot L_\text{input},\ w_\text{out} \cdot M),
  $$

  where $L_\text{input}$ is the longest prompt and $M$ is the number of tokens that fit
  in a running batch, here `max_running` $\cdot\ (L_\text{input} + L_\text{output})$.
  The tests compute the left side from the engine's trace on every seeded workload,
  including one with three tenants backlogged for 270 requests.
- Work conservation: no slot is left free while a request is queued. Within a tenant,
  requests are admitted in arrival order.
- Use `asyncio` and the engine only; time is virtual.

## Why this matters

A shared LLM endpoint is a commons. First come, first served hands it to whoever sends
the most: one tenant running a batch job fills the queue and everyone else's latency
goes up by the length of that queue. The usual defense, a requests-per-minute limit
per tenant, is both too strict and too loose: it rejects a tenant's traffic while GPUs
sit idle, and it treats a 20-token question and a 4,000-token document with an
essay-length answer as the same unit. VTC applies fair queueing, the idea behind
network schedulers, to tokens: it never rejects work that fits, it measures what each
tenant actually consumed, and it does so without knowing output lengths in advance,
because counters are updated as tokens are produced. The paper proves that two
backlogged tenants never drift apart by more than a constant, and the counter lift is
what makes the guarantee hold for tenants that come and go.
