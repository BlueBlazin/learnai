Build an inference gateway that sits between clients and a fleet of model replicas. You will implement five stages: per-tenant admission control, least-loaded routing, prefix-cache-aware routing, fair scheduling across tenants, and autoscaling with startup delay and graceful draining.

## Overview

```python
@dataclass(frozen=True)
class Rejected:  # provided in starter
    reason: str  # "rate_limited" or "queue_full"
    retry_after_s: float | None = None
    status: str = "rejected"


class Gateway:
    def __init__(self, cluster, cfg): ...

    counters: dict  # stage 4: tenant -> virtual token counter

    def submit(self, request) -> asyncio.Future: ...  # stages 1 to 4
    def evaluate(self) -> int: ...  # stage 5
```

![A request passes admission or is rejected at once. The dispatcher pairs a queued request with a ready replica that has a free slot.](assets/architecture.svg)

Test cases are grouped by stage. Each stage only exercises its own features and those of earlier stages. Later features are disabled in `cfg` by default, and `evaluate` is only called in stage 5.

**The cluster.** `cluster` is the kit's [`SimCluster`](../../../python/src/learnai/kit/simcluster.py). Simulation time is virtual and measured in seconds via `cluster.sim.now()`.

- `cluster.replicas` lists all non-stopped replicas in ascending `id` order. Each replica exposes an integer `id`, `state` (`"starting"`, `"ready"`, `"draining"`, or `"stopped"`), `queue_len` (requests currently waiting or running on that replica), and `outstanding_tokens` (remaining prompt tokens to prefill plus remaining output tokens to generate across all requests it holds).
- `future = replica.submit(request)` dispatches a request to a replica and returns a future that resolves to an `Outcome` (`status` is `"finished"` or `"failed"`; it never raises). Both `queue_len` and `outstanding_tokens` update immediately before `submit` returns. Submitting to a `"draining"` or `"stopped"` replica fails the test case.
- Stage 5 also uses `cluster.metrics()` and `cluster.scale_to(n)`.

Each replica models a continuous-batching server running at most 8 sequences at once in first-come, first-served order. Each decode/prefill step takes $2^{-6}\text{ s} + 2^{-12}\text{ s}$ per prefilled token $+ 2^{-9}\text{ s}$ per decoded sequence. When prefix caching is enabled (stage 3), a replica skips prefilling the longest prefix (in whole 16-token blocks) shared with any prompt it has already finished prefilling. Each replica's cache holds 1200 tokens (roughly three prompts) with LRU eviction. The gateway cannot inspect replica caches directly.

**Requests.** Requests are [`SimRequest`](../../../python/src/learnai/kit/workloads.py) objects with attributes `id` (unique), `prompt_tokens`, `tenant` (a `str`), and `tokens` (a tuple of token ids, or `None`). You may read `request.output_tokens` only **after** the request finishes, because a real gateway learns the output length only upon completion.

**Configuration (`cfg`).**

| attribute | stage | meaning | default (off) |
|---|---|---|---|
| `max_queued_requests`, `max_queued_tokens` | 1 | gateway queue limits | very large |
| `rate_tokens_per_s`, `burst_tokens` | 1 | per-tenant token bucket parameters | `rate_tokens_per_s = None` |
| `replica_slots` | 1 | max requests the gateway sends to one replica | 8 |
| `prefix_routing`, `match_threshold`, `imbalance_tokens`, `index_size` | 3 | prefix-aware routing parameters | `prefix_routing = False` |
| `fair`, `w_in`, `w_out` | 4 | fair scheduling parameters | `fair = False` |
| `target`, `tolerance`, `min_replicas`, `max_replicas`, `window_s` | 5 | autoscaling policy parameters | `evaluate` not called |

**SLO and goodput.** A request is *good* if it finishes and produces its first token within **1 second** of arriving at the gateway. Rejected, failed, and unfinished requests are not good. Goodput is the count of good requests for a workload.

## Stage 1: admission control and the gateway queue

The harness calls `submit(request)` synchronously at the request's arrival time from an event-loop callback. Return an `asyncio.Future` created with `cluster.sim.loop.create_future()`.

**Gateway queue and free slots.** The gateway buffers admitted requests in its own queue. A replica *has a free slot* when `replica.state == "ready"` and `replica.queue_len < cfg.replica_slots`. A request is considered *queued* from admission until it is handed to a replica.

**Admission rules.** Let $p = \texttt{request.prompt\_tokens}$. Check the following in order:

1. **Rate limit** (active when `cfg.rate_tokens_per_s is not None`). Each tenant has a real-valued token bucket starting at `cfg.burst_tokens` when first seen. It refills continuously at `cfg.rate_tokens_per_s` up to `cfg.burst_tokens`. After $\Delta$ seconds without a charge, the balance is $\min(\text{burst},\ \text{balance} + \text{rate} \cdot \Delta)$. If the current balance is strictly less than $p$, reject with `Rejected("rate_limited", retry_after_s)`, where `retry_after_s` is $(p - \text{balance}) / \text{rate}$, or `None` if $p > \texttt{cfg.burst\_tokens}$ (the request can never fit).
2. **Queue bounds.** Let $Q$ be the number of requests currently in the gateway queue and $T$ be their total prompt tokens. If $Q + 1 > \texttt{cfg.max\_queued\_requests}$ or $T + p > \texttt{cfg.max\_queued\_tokens}$, reject with `Rejected("queue_full")` (`retry_after_s` is `None`). Requests already running on replicas do not count toward $Q$ or $T$.
3. **Admit.** Deduct $p$ from the tenant's bucket (if rate limiting is enabled) and append the request to the end of the gateway queue.

A rejected request must not modify any bucket, counter, or queue. Its returned future must already be resolved with the `Rejected` object before `submit` returns.

**Dispatch.** While the gateway queue is non-empty and at least one ready replica has a free slot, pop the **oldest** queued request and send it via `replica.submit(request)`. Dispatch at exactly three moments: after admitting a request in `submit`, inside a request's completion callback, and at the end of `evaluate` (stage 5). As a result, no queued request ever waits while a free slot is available.

**Completion.** Handle completion in a **done callback** (`future.add_done_callback`) rather than an `await` coroutine. Inside the callback, run these steps in order: charge the tenant (stage 4), resolve the client's future with the replica's `Outcome`, and dispatch.

### Example 1

One replica with `replica_slots = 1`, `max_queued_requests = 2`, `max_queued_tokens = 100`, `rate_tokens_per_s = 64`, `burst_tokens = 128`:

| time | tenant | prompt | result | reason |
|---|---|---|---|---|
| 0 | `a` | 64 | admitted | `a` starts at 128, drops to 64, goes straight to the idle replica |
| 0 | `a` | 48 | admitted | `a` drops to 16; slot is full, so queued (1 request, 48 tokens) |
| 0 | `a` | 32 | `rate_limited`, `0.25` | 32 > 16; wait is $(32 - 16) / 64$ |
| 0 | `b` | 40 | admitted | `b` drops to 88; queue now holds 2 requests, 88 tokens |
| 0 | `b` | 40 | `queue_full` | exceeds `max_queued_requests`; `b` keeps 88 tokens |
| 0 | `b` | 8 | `queue_full` | exceeds `max_queued_requests` |
| 0.25 | `a` | 32 | `queue_full` | `a` refilled to 32, but the first request is still running |
| 0.25 | `c` | 200 | `rate_limited`, `None` | 200 exceeds `burst_tokens = 128` |
| 4 | `a` | 128 | `queue_full` | queue is empty, but 128 exceeds `max_queued_tokens = 100` |
| 4 | `a` | 1 | admitted | fits both bucket and queue |

## Stage 2: least-loaded routing

When multiple ready replicas have a free slot, send the chosen request to the one with the fewest `outstanding_tokens`. Break ties by smallest `id`. Counting tokens instead of requests prevents short prompts from queuing behind long documents.

Requests arriving at the same timestamp are submitted one by one in arrival order, so each sees the `outstanding_tokens` updated by the previous dispatch.

### Example 2

Three idle replicas and six requests at $t = 0$, listed as `(prompt_tokens, output_tokens)`: `(400, 60)`, `(20, 4)`, `(20, 4)`, `(30, 4)`, `(10, 4)`, `(10, 4)`.

| request | `outstanding_tokens` of replicas 0, 1, 2 | chosen replica |
|---|---|---|
| 0 | 0, 0, 0 | 0 (lowest id) |
| 1 | 460, 0, 0 | 1 |
| 2 | 460, 24, 0 | 2 |
| 3 | 460, 24, 24 | 1 |
| 4 | 460, 58, 24 | 2 |
| 5 | 460, 58, 38 | 2 |

A request-count balancer would send request 3 to replica 0 behind the 400-token prompt.

## Stage 3: prefix-aware routing

When `cfg.prefix_routing` is `True`, route any request whose `tokens` is not `None` using a local prefix index. If `request.tokens` is `None`, fall back to stage 2 routing and do not update the index.

**Prefix index.** For each replica id, record the last `cfg.index_size` prompts routed to it under this rule, evicting the oldest entry first (identical prompts are appended again and occupy separate entries). The **match** between a prompt and a replica is the length of the longest common prefix between `request.tokens` and any prompt recorded for that replica, or `0` if its record is empty.

**Routing rule.** Let *load* be `replica.outstanding_tokens`, let the candidates be all ready replicas with a free slot, and let `least` be the minimum load among candidates.

1. A candidate is **eligible** if `load - least <= cfg.imbalance_tokens`.
2. Let `best` be the eligible candidate with the largest match. Break ties by smaller load, then smaller `id`.
3. If `match(best) >= cfg.match_threshold * len(request.tokens)`, route to `best`.
4. Otherwise, route to the least-loaded candidate as in stage 2.
5. Append `request.tokens` to the chosen replica's record in both cases.

### Example 3

Three replicas with `match_threshold = 0.5`, `imbalance_tokens = 250`, and `index_size = 2`. Let `B` be a 40-token prompt. All requests arrive at $t = 0$:

| request | prompt (`output_tokens`) | loads of 0, 1, 2 | chosen replica | why |
|---|---|---|---|---|
| 0 | `B` (200) | 0, 0, 0 | 0 | empty index; least loaded |
| 1 | 30 tokens of `B` + 10 others (20) | 240, 0, 0 | 0 | matches 30/40 on replica 0; 240 ahead is eligible |
| 2 | 8 tokens of `B` + 32 others (20) | 300, 0, 0 | 1 | replica 0 is 300 ahead (ineligible); least loaded |
| 3 | 40 unrelated tokens (20) | 300, 60, 0 | 2 | no match; least loaded |
| 4 | 35 tokens of `B` + 5 others (20) | 300, 60, 60 | 0 | 240 ahead of `least = 60` (eligible); matches 35/40 |
| 5 | 35 tokens of `B` + 1 other (4) | 360, 60, 60 | 1 | replica 0 is 300 ahead; replica 1 matches 8/36 (< 18); least loaded, lowest id |
| 6 | 35 tokens of `B` + 1 other (4) | 360, 100, 60 | 1 | replica 1 now remembers request 5 (matches 35/36) |

Requests 5 and 6 show why breaking affinity matters: once a hot prefix overloads replica 0, it spills over to replica 1, which then serves subsequent requests from its own cache.

## Stage 4: fair scheduling

When `cfg.fair` is `True`, replace FIFO selection in the gateway queue with the Virtual Token Counter (VTC) algorithm. Store each tenant's counter $c_t$ (weighted tokens served) in `self.counters[t]`. A tenant is *queued* if it currently has at least one request in the gateway queue $Q$.

- **Counter lift at admission.** After a request from tenant $u$ passes stage 1 checks and before it is appended to $Q$, initialize `self.counters[u] = 0.0` if $u$ is new. Then, if $u$ is **not** currently queued, lift its counter:
  - If $Q$ is non-empty, set $c_u \leftarrow \max(c_u,\ \min\{c_i : i \text{ is queued}\})$.
  - If $Q$ is empty, set $c_u \leftarrow \max(c_u,\ c_l)$, where $l$ is the tenant of the request that most recently left $Q$ (unchanged if no request has left $Q$ yet).
- **Selecting the next request.** Choose the earliest queued request belonging to the queued tenant with the smallest counter. If multiple queued tenants tie for the smallest counter, pick the one whose earliest request sits closest to the front of $Q$.
- **Charging.** Immediately when a request leaves $Q$ (before the next dispatch decision), add $w_\text{in} \cdot \texttt{prompt\_tokens}$ to its tenant's counter. When a request completes with `outcome.status == "finished"`, add $w_\text{out} \cdot \texttt{output\_tokens}$ inside the completion callback before calling dispatch.

Counters never decrease, and rejected requests do not touch `counters`. Unlike `vtc-fair-scheduler`, which charges output tokens per decode step, a gateway only learns `output_tokens` when the request finishes.

### Example 4

One replica with `replica_slots = 1`, `w_in = 1`, `w_out = 2`. Tenant `a` sends requests with 64 prompt and 8 output tokens (total cost $64 + 16 = 80$, duration `0.154 s`). Tenant `b` sends requests with 32 prompt and 4 output tokens (total cost $32 + 8 = 40$, duration `0.076 s`). Tenant `a` sends requests 0, 1, 2 at $t = 0$. Tenant `b` sends 3 and 4 at $t = 0.03125$. At $t = 2$, `a` sends 5, followed by `b` sending 6 and 7.

| time | event | $c_a$ | $c_b$ |
|---|---|---|---|
| 0 | 0 admitted and sent (+64); 1 and 2 queued | 64 | |
| 0.031 | 3 and 4 admitted; `b` is not queued, lifted to $\min\{c_a\} = 64$ | 64 | 64 |
| 0.154 | 0 finishes (+16); `b` is smaller, so 3 sent (+32) | 80 | 96 |
| 0.230 | 3 finishes (+8); `a` is smaller, so 1 sent | 144 | 104 |
| 0.385 | 1 finishes; 4 sent | 160 | 136 |
| 0.461 | 4 finishes; 2 sent (finishes at 0.615) | 240 | 144 |
| 2 | 5 admitted and sent | 304 | 144 |
| 2 | 6 admitted; $Q$ is empty and `a` left last, so `b` lifts to 304 | 304 | 304 |
| 2.154 | 5 finishes; 6 sent, then 7 | 320 | 384 |

Requests leave the queue in the order `0, 3, 1, 4, 2, 5, 6, 7`. Without the lift at $t = 2$, tenant `b` would return 160 tokens behind and monopolize the replica until catching up, even though it was idle by choice.

## Stage 5: autoscaling and draining

The harness calls `evaluate()` every second starting at $t = 0$ from a timer. When `evaluate()` shares a timestamp with an arrival or step completion, `evaluate()` runs first. Apply the scaling decision and return the desired replica count as a Python `int`.

Call `cluster.metrics()`, which returns `now`, `ready`, `starting`, `queue_depth` (requests waiting on replicas), and `running`. Define `current = ready + starting` and `load = len(gateway_queue) + queue_depth + running`.

1. **Proportional rule.** If `abs(load - current * target) <= tolerance * current * target`, keep `current`. Otherwise compute $\lceil \text{load} / \text{target} \rceil$ using integer ceiling division.
2. **Bounds.** Clamp the recommendation to `[min_replicas, max_replicas]` to obtain `raw`.
3. **Stabilization history.** Record `(now, raw)`. A past entry at time `s` is *in the window* at time `t` if `t - s < window_s`.
4. **Scale up immediately.** If `raw >= current`, set `n = raw`.
5. **Scale down conservatively.** Otherwise set `n = min(current, max recommendation in the window)`.

Call `cluster.scale_to(n)`, which adds new replicas (`"starting"` for 4 seconds before becoming `"ready"`) or **drains** the newest replicas (they accept no new work, finish what they hold, and stop). Never call `cluster.stop(replica)`, which kills in-flight requests. Finally, call dispatch (so newly ready replicas pick up queued work) and return `n`.

Notice how stage 5 interacts with earlier stages: routing already skips `"starting"` and `"draining"` replicas, `current` includes `"starting"` replicas so they are not re-ordered during boot, and `load` includes the gateway queue that the cluster cannot see.

## Constraints

- Up to 40 replicas, 1500 requests, and 4 tenants per case. Arrival times are multiples of $2^{-10}$ s, and all rates, bursts, and weights are exact binary floats, so buckets and counters are exact (`retry_after_s` is checked within `1e-9`).
- **Rule verification.** The harness records each replica's `state`, `queue_len`, and `outstanding_tokens` at every `replica.submit` call, then replays your gateway's exact sequence of admissions, dispatches, completions, and `evaluate` calls against the rules. It verifies admission outcomes, queue selection, replica routing, `evaluate` decisions, `replica_slots` enforcement, work conservation (no waiting while a slot is free), no failed or duplicate requests, and that every admitted future resolves with its replica's `Outcome`. Full traces are not diffed against a single reference trace.
- **Performance limits** on seeded workloads:

  | stage | workload | requirement | reference | without the stage |
  |---|---|---|---|---|
  | 1 | 600 requests at 48/s on 1 replica (3x capacity), queue of 8 requests and 1024 tokens | $\ge 130$ good requests | 159 | 19 (unbounded queue) |
  | 3 | 600 prompts over 8 prefixes of 256–512 tokens at 30/s, 4 replicas, `imbalance_tokens = 1500` | $\ge 90{,}000$ cache-hit tokens and $\ge 570$ good requests | 127,424 and 593 | 54,144 and 433 (load only) |
  | 3 | one hot prefix at 40/s, `imbalance_tokens = 400` | chosen replica never $> 400$ tokens above least-loaded candidate | 342 | |
  | 4 | batch tenant sends 300 long requests at 60/s, chat tenant sends 60 short ones at 4/s, 2 replicas | $\ge 54$ good chat requests | 60 | 0 (FCFS) |
  | 4 | 3 backlogged tenants with different request shapes, 2 replicas | service gap $\le 2000$ | 820 | 11,651 |
  | 5 | 5/s for 20 s, 30/s for 40 s, 5/s for 40 s (1528 requests), starting from 1 replica | $\ge 1300$ good requests, $\le 650$ replica-seconds | 1415, 528 | |
  | 5 | 6 bursts of 30/s lasting 6 s, 6 s apart, on a 5/s base (1455 requests) | $\ge 1200$ good requests, $\le 700$ replica-seconds | 1330, 570 | |
  | 5 | 3 tenants, shared prefixes, rate limits, fairness, and scaling together | $\ge 700$ good requests, $\le 320$ replica-seconds | 805, 249 | |

  The *service gap* is measured while all three tenants are continuously queued. Let $W_t$ be the cumulative service of tenant $t$ ($w_\text{in}$ per prompt token at dispatch plus $w_\text{out}$ per output token as generated). The service gap is the maximum change in $W_f - W_g$ over any sub-interval and any pair of tenants $f, g$. *Replica-seconds* is the sum over all replicas of the time from creation to stop.
- Stage 5 defaults: `target = 6`, `tolerance = 0.25`, `min_replicas = 1`, `max_replicas = 40`, `window_s = 20`, with 4 s replica startup time.
- Use only `asyncio` and the `cluster` API. Do not read wall-clock time, sleep in real time, inspect `replica.cache`, or read the simulation trace.

## Why this matters

Model servers decide what runs in the next forward pass, while the inference gateway decides what reaches the fleet in the first place. Without admission control, overload builds an unbounded queue where every reply misses its latency SLO instead of rejecting excess load early. Request-count balancing traps short queries behind long documents, prefix-blind routing wastes compute re-prefilling shared system prompts, and sticky prefix routing overloads the replica holding a popular prompt. FIFO queuing lets a single batch tenant starve interactive users, while an autoscaler that ignores starting replicas or kills busy replicas turns traffic spikes into outages. Real systems such as the Kubernetes Gateway API Inference Extension and llm-d are built from these exact components.
