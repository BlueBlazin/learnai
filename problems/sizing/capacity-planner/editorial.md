## The idea

Capacity planning breaks into three steps. First, find what one replica can sustain at the latency objective. Second, reserve headroom so the replica stays below saturation. Third, divide total load by usable single-replica capacity and round up.

**One replica from the cost model.** Under continuous batching, $b$ sequences stay in flight and each needs $o$ steps (one prefill of $p$ prompt tokens and $o - 1$ decodes). Rather than modeling a single step, sum an entire cycle of $o$ steps. Over $o$ steps, every slot performs its full request work once:

$$
S(b) = \underbrace{o \cdot \text{base\_us}}_{\text{fixed, per step}}
     + b \cdot \underbrace{\big(\text{prefill\_us\_per\_token} \cdot p + \text{decode\_us\_per\_seq} \cdot (o - 1)\big)}_{\text{one request's work}} .
$$

This total holds regardless of how the slots are staggered across steps. Two metrics follow directly. Mean TPOT is $S(b) / o$, which grows linearly with $b$ because every active sequence waits for the entire batch on each step. Throughput is $b / S(b)$ requests per microsecond, which also grows with $b$ as fixed overhead is amortized, though with diminishing returns. Because latency and throughput trade off along the single knob $b$, the SLO picks the largest batch whose TPOT stays within the limit. Its throughput is the replica's capacity at the objective, which is lower than peak throughput.

**One replica from a measured curve.** A load test records latency at discrete rates, and the SLO crossing rarely lands on a tested rate. Linear interpolation estimates the crossing between the last passing point and the first failing point. Extrapolation is unsafe in both directions because latency curves bend sharply upward near saturation and nothing is known below the lowest tested rate. Stopping at the first failing point prevents a noisy dip past the knee from inflating capacity, just as in `goodput-slo`.

**Headroom.** The cost model describes steady-state means. Real arrivals are bursty and sequence lengths vary, so a replica running at 100% of mean capacity spends half its time overloaded. Let $\lambda$ be the total offered request rate, $c$ be one replica's capacity, and $h = \text{headroom\_percent} / 100$ be the reserved headroom fraction. Capping utilization at $1 - h$ gives

$$
R = \left\lceil \frac{\lambda}{c \cdot (1 - h)} \right\rceil .
$$

Headroom scales usable capacity down rather than multiplying replicas up. Leaving 25% headroom means running at 75% utilization, which requires $1 / 0.75 = 1.33\times$ as many replicas rather than $1.25\times$.

**Exact integer arithmetic.** Because $R$ is a ceiling, a tiny floating-point error can add an extra replica whenever the load is an exact multiple of usable capacity. For example, `math.ceil(125 / (4.0983... * 0.5))` evaluates to 62 instead of 61 when the float denominator lands slightly low. Since all inputs are integers, keep capacity as an exact fraction `(num, den)` and cross-multiply for every comparison and ceiling.

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

Running the worked example on the kit's `SimCluster` (one replica, `max_seqs = 32`, Poisson arrivals, 1,500 requests of 500 prompt and 200 output tokens, discarding the first 20% as warmup) shows how the model compares to simulated traffic. The analytical model predicts a single-replica capacity of 3.19 requests per second at a 40 ms TPOT.

| offered rate (req/s) | share of capacity | mean TPOT | requests with TPOT <= 40 ms | p99 TTFT |
|---|---|---|---|---|
| 2.35 (the plan with 25% headroom) | 0.74 | 22.5 ms | 99.7% | 0.19 s |
| 2.8 | 0.88 | 30.5 ms | 90.2% | 1.6 s |
| 3.07 (the plan with no headroom) | 0.96 | 37.9 ms | 52.5% | 2.6 s |
| 3.19 (the model's capacity) | 1.00 | 42.1 ms | 25.2% | 3.3 s |

At the model's predicted capacity, mean TPOT lands close to 40 ms, yet three quarters of requests miss the target because queueing and arrival variance push individual requests above the mean. The 25%-headroom plan keeps 99.7% of requests within the SLO and keeps p99 TTFT low.

## Common mistakes

- **Omitting prefill from step time.** Counting only decode steps ignores that prefilling 500 tokens costs as much as 50 decode steps in the example. That yields a batch size of 32 instead of 25 and 14 replicas instead of 17.
- **Choosing batch size for throughput alone.** Throughput rises with batch size up to `max_seqs`, so maximizing throughput always picks the cap regardless of latency.
- **Dropping the ceiling.** Using `round` or `int` turns 16.3 replicas into 16, leaving the fleet under-provisioned.
- **Multiplying by headroom instead of dividing capacity.** Computing $R = \lceil 1.25 \cdot \lambda / c \rceil$ gives 16 replicas instead of 17 in the example. The error grows with headroom (at 50% headroom, $1.5\times$ vs. $2\times$).
- **Using floating-point `math.ceil`.** Floating-point rounding errors cause off-by-one replica counts on exact multiples of usable capacity.
- **Charging $o$ decode steps instead of $o - 1$.** The prefill step also emits the first output token. A request with one output token has zero decode steps.
- **Using a strict inequality for the SLO.** Meeting a 40 ms objective includes 40 ms exactly. Using `<` loses a sequence per replica when the bound is hit exactly.
- **Forgetting to cap at `max_seqs`.** A loose SLO can otherwise select a batch size larger than the engine allows.
- **Skipping interpolation on the curve.** Taking the last passing measured rate as capacity wastes hardware (3.0 instead of 3.48 requests per second in the example, requiring 18 replicas instead of 16).
- **Using the last crossing instead of the first.** Scanning for the highest passing rate trusts a noisy measurement past the saturation knee.
- **Extrapolating beyond the curve.** Extending the final segment assumes latency grows linearly right where queueing causes it to blow up.

## Complexity

`plan` runs in $O(1)$ time, and `plan_from_curve` runs in $O(n)$ time for a curve with $n$ points.

## In the wild

- **Latency-bounded throughput** is the standard metric for tuning serving systems. DistServe (arXiv:2401.09670) defines goodput as the per-GPU request rate meeting TTFT and TPOT targets, and serving engines expose a concurrency cap (`max_seqs` here, `--max-num-seqs` in vLLM) to trade latency against throughput.
- **Autoscalers** evaluate this same ratio continuously using live load in place of a forecast. Desired replicas equal the ceiling of current load divided by target-utilized capacity, matching the Kubernetes Horizontal Pod Autoscaler formula `ceil(currentReplicas * currentMetric / targetMetric)`.
- **Limits of the analytical model.** The cost model omits request-length variance (a 500-token average made of 100-token and 4,000-token prompts behaves worse than uniform 500-token prompts), KV-cache memory limits on batch size (`memory-budget-fit`), and queueing-dominated TTFT at high utilization. A measured load-test curve captures all of these effects, which is why teams validate cost-model plans against real benchmarks.
