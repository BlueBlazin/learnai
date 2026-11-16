Build the gateway that stands between clients and a fleet of inference replicas, in
five stages: admission control with a token-rate limit per tenant, least-loaded
routing, prefix-cache-aware routing that knows when to break affinity, fair
scheduling between tenants in the gateway's queue, and autoscaling with a start-up
delay and graceful scale-down. Every stage answers one question about a request
(may it come in, where does it go, whose turn is it, how many replicas should there
be), and the gateway is the place where the answers have to agree.

## Overview

```python
@dataclass(frozen=True)
class Rejected:  # given in the starter
    reason: str  # "rate_limited" or "queue_full"
    retry_after_s: float | None = None
    status: str = "rejected"


class Gateway:
    def __init__(self, cluster, cfg): ...

    counters: dict  # stage 4: tenant -> virtual token counter

    def submit(self, request) -> asyncio.Future: ...  # stages 1 to 4
    def evaluate(self) -> int: ...  # stage 5
```

![The gateway. A request passes admission (its tenant's token bucket, then the bounds of the queue) or is rejected at once. Admitted requests wait in the gateway queue until a ready replica has a free slot; the dispatcher then chooses which request leaves the queue and which replica gets it. Completions come back to the gateway, which also evaluates the autoscaling policy.](assets/architecture.svg)

The cases are grouped by stage. Each stage's cases use only what that stage and the
earlier ones specify: later features are off in `cfg` by default, and `evaluate` is
only called in stage 5.

**The cluster** is the kit's
[`SimCluster`](../../../python/src/whetstone/kit/simcluster.py). Time is virtual, in
**seconds**: read it with `cluster.sim.now()`. What the gateway uses:

- `cluster.replicas`: the replicas that are not stopped, in id order. A replica has
  an int `id`, a `state` (`"starting"`, `"ready"`, `"draining"`, `"stopped"`),
  `queue_len` (the requests it holds, waiting or running), and `outstanding_tokens`
  (prompt tokens it still has to prefill plus output tokens it still has to produce,
  over the requests it holds).
- `future = replica.submit(request)` hands a request to a replica. The future
  resolves with an `Outcome` (`status` is `"finished"` or `"failed"`; it never
  raises). `queue_len` and `outstanding_tokens` include the request as soon as the
  call returns. Sending to a draining or stopped replica is an error that fails the
  case.
- Stage 5: `cluster.metrics()` and `cluster.scale_to(n)`, described there.

Each replica is a continuous-batching server that runs at most 8 sequences at once,
first come, first served; a step costs $2^{-6}$ s plus $2^{-12}$ s per prefilled
token plus $2^{-9}$ s per decoded sequence. With a prefix cache (stage 3) a replica
does not prefill the longest prefix, in whole blocks of 16 tokens, that a new prompt
shares with a prompt it has finished prefilling; the cache holds 1200 tokens, about
three prompts, least recently used evicted first. The gateway cannot see the caches.

**Requests** are [`SimRequest`](../../../python/src/whetstone/kit/workloads.py)
objects: `id` (unique), `prompt_tokens`, `tenant` (a string), `tokens` (the prompt
as a tuple of token ids, or `None`). `output_tokens` may be read only **after** the
request has finished: a gateway learns the length of an answer when it is complete.

**`cfg`** is an object with these attributes:

| attribute | stage | meaning | off by default as |
|---|---|---|---|
| `max_queued_requests`, `max_queued_tokens` | 1 | bounds of the gateway queue | very large |
| `rate_tokens_per_s`, `burst_tokens` | 1 | each tenant's token bucket | `rate_tokens_per_s = None` |
| `replica_slots` | 1 | requests the gateway lets one replica hold | 8 |
| `prefix_routing`, `match_threshold`, `imbalance_tokens`, `index_size` | 3 | prefix-aware routing | `prefix_routing = False` |
| `fair`, `w_in`, `w_out` | 4 | fair scheduling | `fair = False` |
| `target`, `tolerance`, `min_replicas`, `max_replicas`, `window_s` | 5 | the autoscaling policy | `evaluate` is not called |

**SLO and goodput.** A request is *good* if it finished and its first token came
within **1 second** of its arrival at the gateway. Rejected, failed, and unfinished
requests are not good. Goodput is the number of good requests; the limits below are
stated as counts for a given workload.

## Stage 1: admission control and the gateway queue

`submit(request)` is called by the test at the request's arrival instant, from a
plain callback inside the event loop. It is a plain method and returns an
`asyncio.Future` (create it with `cluster.sim.loop.create_future()`).

**The queue.** The gateway holds admitted requests in its own queue and sends one
to a replica only when a replica that is `"ready"` holds fewer than
`cfg.replica_slots` requests (`queue_len < replica_slots`): such a replica *has a
free slot*. A request is *queued* from its admission until it is handed to a
replica.

**Admission.** With $p$ = `request.prompt_tokens`, in this order:

1. **Rate limit** (when `rate_tokens_per_s` is not `None`). Every tenant has a
   token bucket: a real-valued balance that is `burst_tokens` when the tenant is
   first seen, refills continuously at `rate_tokens_per_s`, and never exceeds
   `burst_tokens`: after $\Delta$ seconds without a charge it is
   $\min(\text{burst},\ \text{balance} + \text{rate} \cdot \Delta)$. If the balance
   now is below $p$, the request is rejected with `Rejected("rate_limited",
   retry_after_s)`, where `retry_after_s` is $(p - \text{balance}) / \text{rate}$,
   or `None` if $p >$ `burst_tokens` (it can never be admitted).
2. **Queue bounds.** With $Q$ queued requests holding $T$ prompt tokens: if
   $Q + 1 >$ `max_queued_requests` or $T + p >$ `max_queued_tokens`, the request is
   rejected with `Rejected("queue_full")` (`retry_after_s` is `None`). Requests
   that are already on a replica count against neither bound.
3. Otherwise the request is **admitted**: its tenant's bucket is charged $p$, and
   the request joins the end of the queue.

A rejected request changes nothing: in particular a request rejected by (2) does
not cost its tenant tokens. Its future is already resolved, with the `Rejected`
object, when `submit` returns.

**Dispatch.** While the queue is not empty and some ready replica has a free slot:
take the **oldest** queued request and hand it to such a replica
(`replica.submit`). The gateway dispatches at exactly these moments: in `submit`
after admitting a request; when a request completes; and (stage 5) at the end of
`evaluate`. So no request ever waits while a slot it could use has been free since
the gateway's last dispatch.

**Completion.** Handle a completion in a **done callback** on the replica's future
(`future.add_done_callback`), not in a task that awaits it. In the callback, in this
order: (stage 4: charge the tenant;) resolve the future that `submit` returned with
the replica's `Outcome` object; dispatch.

### Example 1

One replica, `replica_slots = 1`, `max_queued_requests = 2`,
`max_queued_tokens = 100`, `rate_tokens_per_s = 64`, `burst_tokens = 128`.

| time | tenant | prompt | answer | why |
|---|---|---|---|---|
| 0 | a | 64 | admitted | a's bucket: 128, now 64. Goes straight to the idle replica |
| 0 | a | 48 | admitted | a: 16 left. The slot is taken: queued (1 request, 48 tokens) |
| 0 | a | 32 | `rate_limited`, 0.25 | 32 > 16; $(32 - 16) / 64$ |
| 0 | b | 40 | admitted | b: 88 left. The queue holds 2 requests, 88 tokens |
| 0 | b | 40 | `queue_full` | a third queued request. b still has 88 |
| 0 | b | 8 | `queue_full` | |
| 0.25 | a | 32 | `queue_full` | a has 32 again, but the first request is still running |
| 0.25 | c | 200 | `rate_limited`, `None` | 200 tokens never fit a bucket of 128 |
| 4 | a | 128 | `queue_full` | the queue is empty, but 128 > `max_queued_tokens` |
| 4 | a | 1 | admitted | |

## Stage 2: least-loaded routing

With several replicas, the request taken from the queue goes to the replica with
the **fewest `outstanding_tokens`** among the ready replicas that have a free slot;
among equals, the smallest `id`. Load is counted in tokens because requests differ
in size by orders of magnitude: three short questions are less work than one long
document.

Same-instant arrivals are submitted one after another, in arrival order, and each
sees the load left by the previous one.

### Example 2

Three idle replicas, six requests at time 0, as (prompt tokens, output tokens):
`(400, 60)`, `(20, 4)`, `(20, 4)`, `(30, 4)`, `(10, 4)`, `(10, 4)`.

| request | outstanding tokens of replicas 0, 1, 2 | goes to |
|---|---|---|
| 0 | 0, 0, 0 | 0 (lowest id) |
| 1 | 460, 0, 0 | 1 |
| 2 | 460, 24, 0 | 2 |
| 3 | 460, 24, 24 | 1 |
| 4 | 460, 58, 24 | 2 |
| 5 | 460, 58, 38 | 2 |

A balancer that counts requests sends request 3 to replica 0, behind the long
prompt.

## Stage 3: prefix-aware routing

With `cfg.prefix_routing`, a request that has `tokens` is routed as follows; a
request whose `tokens` is `None` is routed as in stage 2 and leaves no record.

**The index.** For every replica id the gateway remembers the last
`cfg.index_size` prompts it sent there with this rule, oldest forgotten first (an
identical prompt is remembered again and counts twice). The **match** of a prompt
with a replica is the length of the longest common prefix between `tokens` and any
prompt remembered for that replica; 0 for an empty record.

**The decision.** *Load* is `outstanding_tokens`; the candidates are the ready
replicas with a free slot; `least` is the smallest load among them.

1. A candidate is **eligible** if `load - least <= cfg.imbalance_tokens`.
2. The **best** replica is the eligible one with the largest match; among equal
   matches the smaller load; then the smaller `id`.
3. If `match(best) >= cfg.match_threshold * len(tokens)`, the request goes to
   `best`.
4. Otherwise it goes to the least loaded candidate, as in stage 2.
5. The prompt is appended to the record of the chosen replica, in both cases.

### Example 3

Three replicas, `match_threshold = 0.5`, `imbalance_tokens = 250`,
`index_size = 2`. `B` is a prompt of 40 tokens; all requests arrive at time 0.

| request | prompt (output tokens) | loads of 0, 1, 2 | goes to | why |
|---|---|---|---|---|
| 0 | `B` (200) | 0, 0, 0 | 0 | nothing is remembered: least loaded |
| 1 | 30 tokens of `B`, 10 others (20) | 240, 0, 0 | 0 | matches 30 of 40 on replica 0, which is 240 ahead: eligible |
| 2 | 8 tokens of `B`, 32 others (20) | 300, 0, 0 | 1 | replica 0 is 300 ahead: not eligible. Least loaded |
| 3 | 40 unrelated tokens (20) | 300, 60, 0 | 2 | |
| 4 | 35 tokens of `B`, 5 others (20) | 300, 60, 60 | 0 | 240 ahead of the least loaded: eligible, matches 35 |
| 5 | 35 tokens of `B`, 1 other (4) | 360, 60, 60 | 1 | replica 0 is 300 ahead. Replica 1 matches 8 of 36, below 18: least loaded, lowest id |
| 6 | 35 tokens of `B`, 1 other (4) | 360, 100, 60 | 1 | replica 1 remembers request 5: 35 of 36 |

Requests 5 and 6 show the point of breaking affinity: a prefix that is too popular
for one replica spreads to a second one, which then serves it from its own cache.

## Stage 4: fair scheduling

With `cfg.fair`, the request that leaves the queue is chosen by the Virtual Token
Counter algorithm instead of by age. `gateway.counters[t]` is the counter $c_t$ of
tenant $t$: the weighted tokens it has been served. A tenant is *queued* if it has a
request in the gateway queue $Q$.

- **At admission** of a request of tenant $u$ (after the checks of stage 1, before
  it joins $Q$): $u$ enters `counters` with `0.0` if it is new. Then, if $u$ is
  **not queued**, the **counter lift**:
  - if $Q$ is not empty: $c_u \leftarrow \max(c_u,\ \min\{c_i : i \text{ queued}\})$;
  - if $Q$ is empty: $c_u \leftarrow \max(c_u,\ c_l)$, where $l$ is the tenant of
    the request that left $Q$ most recently; unchanged if none ever has.
- **Choosing**: the request that leaves $Q$ is the earliest request of the queued
  tenant with the smallest counter; among tenants with equal counters, the one
  whose earliest request is first in $Q$.
- **Charging**: when a request leaves $Q$,
  $c \leftarrow c + w_\text{in} \cdot \text{prompt\_tokens}$, at once (before the
  next choice). When a request **finishes** (`outcome.status == "finished"`),
  $c \leftarrow c + w_\text{out} \cdot \text{output\_tokens}$, in the completion
  callback before the dispatch.

Counters never decrease. A rejected request touches no counter. (In
`vtc-fair-scheduler` output tokens are charged one by one as they are generated; a
gateway sees an answer when it is complete, and charges it then.)

### Example 4

One replica, `replica_slots = 1`, `w_in = 1`, `w_out = 2`. A request of tenant `a`
has 64 prompt and 8 output tokens (it costs 64 + 16 and takes 0.154 s); one of `b`
has 32 and 4 (32 + 8, 0.076 s). `a` sends requests 0, 1, 2 at time 0; `b` sends 3
and 4 at 0.03125; at time 2, `a` sends 5 and then `b` sends 6 and 7.

| time | event | $c_a$ | $c_b$ |
|---|---|---|---|
| 0 | 0 admitted and sent (+64); 1 and 2 queued | 64 | |
| 0.031 | 3, 4 admitted; `b` is new to $Q$: lifted to $\min\{c_a\} = 64$ | 64 | 64 |
| 0.154 | 0 finishes (+16). `b` is smaller: 3 sent (+32) | 80 | 96 |
| 0.230 | 3 finishes (+8). `a` is smaller: 1 sent | 144 | 104 |
| 0.385 | 1 finishes. 4 sent | 160 | 136 |
| 0.461 | 4 finishes. 2 sent; it finishes at 0.615 | 240 | 144 |
| 2 | 5 admitted and sent | 304 | 144 |
| 2 | 6 admitted: $Q$ is empty and `a` left it last: `b` lifted to 304 | 304 | 304 |
| 2.154 | 5 finishes. 6 sent; then 7 | 320 | 384 |

The requests leave the queue in the order 0, 3, 1, 4, 2, 5, 6, 7. Without the lift
at time 2, `b` would return 160 tokens "behind" and be served ahead of `a` until it
had caught up, although nothing was withheld from it while it was away.

## Stage 5: autoscaling and draining

`evaluate()` is one evaluation of the autoscaling policy. The test calls it every
second from $t = 0$, from a plain timer; an evaluation that shares its instant with
an arrival or the end of a replica's step happens before them. It returns the
number of replicas that should be starting or ready, a Python `int`, after applying
it.

`cluster.metrics()` returns the cluster now: `now`, `ready`, `starting`,
`queue_depth` (requests waiting on replicas), and `running`. With
`current = ready + starting` and

`load = (requests in the gateway queue) + queue_depth + running`:

1. **Proportional rule.** If `abs(load - current * target) <= tolerance * current
   * target`, the recommendation is `current`. Otherwise it is
   $\lceil \text{load} / \text{target} \rceil$, in integers.
2. **Bounds.** Raise it to `min_replicas`, cap it at `max_replicas`: `raw`.
3. **History.** Record `raw` with the time `now`. A recorded recommendation is *in
   the window* at time `t` if it was made at a time `s` with `t - s < window_s`.
4. **Scale up at once.** If `raw >= current`, `n = raw`.
5. **Scale down late.** Otherwise `n = min(current, highest recommendation in the
   window)`.

Then `cluster.scale_to(n)`: it adds replicas, which are `"starting"` for 4 seconds
and then `"ready"`, or **drains** the newest ones, which take no new work, finish
what they hold, and stop. Finally the gateway dispatches (a replica that has become
ready may take queued work), and returns `n`.

What this adds to the earlier stages: only `"ready"` replicas are candidates, so a
starting replica gets nothing and a draining one must get nothing; replicas that
are starting count as capacity, so an evaluation during a start-up does not order
them again; and the load includes the gateway's own queue, which the cluster does
not see. Scaling down must never fail a request: `cluster.stop(replica)` exists and
kills what the replica holds.

## Constraints

- Up to 40 replicas, 1500 requests, and four tenants per case. Arrival times are
  multiples of $2^{-10}$ s; rates, bursts, and weights are exactly representable,
  so buckets and counters are exact floats (`retry_after_s` is compared within
  `1e-9`).
- **Every decision is checked against the rules, with the state the gateway saw.**
  The tests record the replicas' `state`, `queue_len`, and `outstanding_tokens` at
  every `replica.submit`, and replay: each admission answer (with a token bucket
  and the real queue), the request chosen (oldest, or by the counters of stage 4,
  following the gateway's own order of admissions, dispatches, and completions),
  the replica chosen (stage 2, or stage 3 with an index rebuilt from the gateway's
  own routes), and the return value and effect of every `evaluate`. They also check
  that a replica never holds more than `replica_slots` requests, that no request
  waits while a slot has been free since the last dispatch, that no request fails
  or is sent twice, and that every admitted request's future resolves with its
  replica's `Outcome`. Whole traces are not compared with the reference.
- **Limits**, each on a seeded workload, derived from the reference with a margin:

  | stage | workload | limit | reference | without the stage |
  |---|---|---|---|---|
  | 1 | 600 requests at 48/s on one replica (three times its capacity), queue of 8 requests and 1024 tokens | at least 130 good requests | 159 | 19 (unbounded queue) |
  | 3 | 600 prompts over 8 prefixes of 256 to 512 tokens at 30/s, 4 replicas, `imbalance_tokens = 1500` | at least 90,000 cache-hit tokens and 570 good requests | 127,424 and 593 | 54,144 and 433 (load only) |
  | 3 | one hot prefix at 40/s, `imbalance_tokens = 400` | the chosen replica is never more than 400 tokens above the least loaded candidate | 342 | |
  | 4 | a batch tenant sends 300 long requests at 60/s, a chat tenant 60 short ones at 4/s, 2 replicas | at least 54 good chat requests | 60 | 0 (first come, first served) |
  | 4 | three tenants with different request shapes, backlogged, 2 replicas | service gap at most 2000 | 820 | 11,651 |
  | 5 | 5/s for 20 s, 30/s for 40 s, 5/s for 40 s (1528 requests), from one replica | at least 1300 good requests, at most 650 replica-seconds | 1415, 528 | |
  | 5 | six bursts of 30/s lasting 6 s, 6 s apart, on a base of 5/s (1455 requests) | at least 1200 good requests, at most 700 replica-seconds | 1330, 570 | |
  | 5 | three tenants, shared prefixes, rate limits, fairness, and scaling together | at least 700 good requests, at most 320 replica-seconds | 805, 249 | |

  The *service gap* is measured over the period in which all three tenants are
  queued: with $W_t$ the service of tenant $t$ ($w_\text{in}$ per prompt token when
  a request is sent to a replica, $w_\text{out}$ per output token when it is
  generated), it is the largest change of $W_f - W_g$ over any interval, for any
  two tenants. Replica-seconds are the time from a replica's addition to its stop,
  summed over all replicas.
- Stage 5 defaults: `target = 6`, `tolerance = 0.25`, `min_replicas = 1`,
  `max_replicas = 40`, `window_s = 20`; replicas take 4 s to start.
- Use `asyncio` and the cluster only. Time is virtual: no real clock, no real
  sleeping. Do not read `replica.cache` or the simulation's trace.

## Why this matters

A model server answers one question: what goes into the next forward pass. The
gateway in front of a fleet answers the ones that come before it, and each has a
failure with a name. Without admission control, overload becomes a queue that makes
every answer late instead of some answers "no". A balancer that counts requests
puts a short question behind three long documents. A router that ignores the prefix
cache prefills the same system prompt on every replica; one that follows it blindly
buries the replica that holds the popular prefix. First come, first served hands
the fleet to whoever sends the most. And an autoscaler that forgets the replicas it
has ordered, or removes one that is still answering, turns a traffic spike into an
outage. The endpoint picker of the Kubernetes Gateway API Inference Extension and the
scheduler of llm-d are built from these parts: scorers for queue length and prefix
cache in front of model servers.
