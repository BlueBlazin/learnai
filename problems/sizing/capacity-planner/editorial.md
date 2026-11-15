## The idea

A capacity plan is three steps: find what one replica can do at the objective, decide
how much of that to use, and divide.

**One replica, from the cost model.** Continuous batching keeps $b$ sequences in
flight; each needs $o$ steps (one prefill, $o - 1$ decodes). Rather than ask what a
typical step looks like, count a whole cycle of $o$ steps, in which every slot does
all of its work exactly once:

$$
S(b) = \underbrace{o \cdot \text{base}}_{\text{fixed, per step}}
     + b \cdot \underbrace{\big(\text{prefill} \cdot p + \text{decode} \cdot (o - 1)\big)}_{\text{one request's work}} .
$$

This holds whatever the phases of the slots are: whether all prompts are prefilled in
one long step or one per step, the $o$ steps contain the same work. Two numbers follow.
TPOT is $S(b) / o$: it grows linearly with the batch, because every sequence waits for
everyone's work in every step. Throughput is $b / S(b)$ requests per microsecond: it
also grows with the batch, because the fixed cost is shared, but with diminishing
returns as $b \cdot \text{per\_slot}$ comes to dominate. So latency and throughput
pull in opposite directions along one knob, and the objective picks the point: the
largest batch whose TPOT is still allowed. That batch's throughput is the replica's
capacity *at the objective*, which is smaller than its maximum throughput.

**One replica, from a measurement.** A load test gives latency at a handful of rates.
The capacity is where the curve crosses the objective, and the crossing almost never
falls on a measured rate, so a rule is needed between points (linear interpolation),
beyond them (none: a latency curve near saturation bends upward faster than any line
through its last two points, and below the first point nothing is known), and for a
curve that crosses twice (the first crossing: latency does not improve with load, so
a later pass is measurement noise, as in `goodput-slo`).

**Headroom.** The model is a mean. Real arrivals are bursty and real lengths vary, so
a replica offered exactly its capacity spends half its time above it. Planning at a
utilization below 1 leaves room for that; the rule is on capacity,

$$
R = \left\lceil \frac{\lambda}{c \cdot (1 - h)} \right\rceil ,
$$

not on the replica count: 25 percent headroom means each replica runs at 75 percent,
which takes $1 / 0.75 = 1.33$ times the replicas, not 1.25 times.

**Exactness.** $R$ is a ceiling, and a ceiling is discontinuous exactly where plans
tend to land: on round numbers. `math.ceil(125 / (4.0983... * 0.5))` is 62 where the
exact quotient is 61, because the float product is a hair below the true value. With
integer inputs every comparison can be made in integers: keep the capacity as a
fraction and cross-multiply.

## Reference solution (Python)

```python
def replicas_for(rate, capacity_num, capacity_den, headroom_percent):
    usable_num = capacity_num * (100 - headroom_percent)
    usable_den = capacity_den * 100
    return max(1, -(-rate * usable_den // usable_num))


def plan(cost, workload, tpot_slo_us, headroom_percent):
    p, o = workload["prompt_tokens"], workload["output_tokens"]
    per_slot = cost["prefill_us_per_token"] * p + cost["decode_us_per_seq"] * (o - 1)
    fixed = o * cost["base_us"]
    batch = min(cost["max_seqs"], (o * tpot_slo_us - fixed) // per_slot)
    if batch < 1:
        return None
    cycle_us = fixed + batch * per_slot
    rate = workload["requests_per_s"]
    replicas = replicas_for(rate, batch * 10**6, cycle_us, headroom_percent)
    requests_per_s = batch * 10**6 / cycle_us
    return {
        "batch": batch,
        "tpot_us": cycle_us / o,
        "requests_per_s": requests_per_s,
        "tokens_per_s": o * requests_per_s,
        "replicas": replicas,
        "utilization": rate / (replicas * requests_per_s),
    }


def plan_from_curve(curve, requests_per_s, slo_ms, headroom_percent):
    if curve[0][1] > slo_ms:
        return None
    num, den = curve[-1][0], 1
    for (r0, l0), (r1, l1) in pairwise(curve):
        if l1 > slo_ms:
            num, den = r0 * (l1 - l0) + (slo_ms - l0) * (r1 - r0), l1 - l0
            break
    return {
        "capacity": num / den,
        "replicas": replicas_for(requests_per_s, num, den, headroom_percent),
    }
```

## How good is the model?

The worked example was run on the kit's `SimCluster` (one replica, `max_seqs = 32`,
Poisson arrivals, 1,500 requests of exactly 500 prompt and 200 output tokens, the
first fifth discarded). The model says a replica's capacity at a 40 ms TPOT is 3.19
requests per second.

| offered rate (req/s) | share of capacity | mean TPOT | requests with TPOT <= 40 ms | p99 TTFT |
|---|---|---|---|---|
| 2.35 (the plan with 25% headroom) | 0.74 | 22.5 ms | 99.7% | 0.19 s |
| 2.8 | 0.88 | 30.5 ms | 90.2% | 1.6 s |
| 3.07 (the plan with no headroom) | 0.96 | 37.9 ms | 52.5% | 2.6 s |
| 3.19 (the model's capacity) | 1.00 | 42.1 ms | 25.2% | 3.3 s |

At the model's capacity the mean TPOT is close to the objective, as predicted, and
three quarters of the requests miss it: a mean at the limit means roughly half above
it, and queueing pushes more over. The plan with headroom meets the objective for
nearly every request. That is what the headroom buys, and why a plan without it is a
plan for the average minute.

## Common mistakes

- **Prefill left out of the utilization.** Charging a slot only its decode steps
  ignores that prefilling 500 tokens costs as much as 50 decode steps here. The batch
  comes out as 32 instead of 25 and the plan as 14 replicas instead of 17.
- **The batch chosen for throughput only.** Throughput rises with the batch up to
  `max_seqs`, so maximizing it always picks the cap, whatever the latency.
- **The ceiling omitted.** `round` or `int` on the quotient: 16.3 replicas is 17
  replicas. A fleet cannot run a fraction of a server.
- **Headroom multiplied.** $R = \lceil 1.25 \cdot \lambda / c \rceil$ gives 16 where
  the definition gives 17; the two differ more the larger the headroom (50 percent:
  1.5 times against 2 times).
- **A float ceiling.** Correct almost always, and off by one on exact multiples.
- **Decoding the first token.** The step that completes the prefill produces the
  first token, so a request has $o - 1$ decode steps, not $o$. With one output token
  there is no decode step at all.
- **A strict objective.** "At most 40 ms" includes 40 ms; treating it as strict loses
  a sequence per replica when the bound is met exactly.
- **Forgetting `max_seqs`.** A loose objective permits a batch the replica cannot run.
- **No interpolation.** Taking the last passing measured rate as capacity is safe and
  wasteful: 3 instead of 3.48 requests per second in the example, 18 replicas instead
  of 16.
- **The last crossing.** Scanning for the highest passing rate accepts a noisy point
  beyond the knee.
- **Extrapolating.** Extending the last segment to the objective assumes the curve
  stays straight exactly where it is about to bend.

## Complexity

$O(1)$ for `plan`; $O(n)$ for a curve of $n$ points.

## In the wild

- **Latency-bounded throughput** is how serving systems are tuned and compared: the
  DistServe paper (arXiv:2401.09670) calls the per-GPU rate at a TTFT and TPOT
  attainment goal goodput and searches configurations for it, and a batch-size or
  concurrency cap (`max_seqs` here, `--max-num-seqs` in vLLM) is the knob that trades
  one for the other.
- **Autoscalers** run this calculation continuously with the measured rate in place
  of the forecast: desired replicas is the ceiling of load over target-utilized
  capacity, which is the shape of the Kubernetes Horizontal Pod Autoscaler's rule,
  `ceil(currentReplicas * currentMetric / targetMetric)`.
- **What the model leaves out**: the distribution of lengths (a mean prompt of 500
  tokens made of 100s and 4,000s behaves worse), KV memory as a second cap on the
  batch (`memory-budget-fit`), and TTFT, which at high utilization is dominated by
  queueing rather than by the prefill. A measured curve contains all of these, which
  is why a plan from the cost model is checked against a load test before it is
  trusted.
