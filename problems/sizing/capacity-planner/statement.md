Compute how many serving replicas a workload needs to meet a latency objective. Implement `plan` to size a deployment from an analytical engine cost model, and `plan_from_curve` to size one from a measured single-replica latency curve.

## Specification

```python
def plan(cost: dict, workload: dict, tpot_slo_us: int, headroom_percent: int) -> dict | None: ...
def plan_from_curve(
    curve: list[tuple[int, int]], requests_per_s: int, slo_ms: int, headroom_percent: int
) -> dict | None: ...
```

All numeric inputs are integers. The integer outputs (`"batch"` and `"replicas"`) are exact and must not depend on floating-point rounding. Neither function may modify its arguments.

### The cost model

`cost` describes one replica in the terms of the kit's `SimEngine`:

| key | meaning |
|---|---|
| `base_us` | fixed time of one engine step, in microseconds |
| `prefill_us_per_token` | time added per prompt token prefilled in the step, in microseconds |
| `decode_us_per_seq` | time added per sequence decoded in the step, in microseconds |
| `max_seqs` | maximum number of sequences a replica runs at once |

`workload` specifies `requests_per_s` (total incoming load), `prompt_tokens` ($p$), and `output_tokens` ($o$). Every request is modeled as having length $p$ prompt tokens and $o$ output tokens.

**A replica at batch size $b$.** The replica runs continuous batching with $b$ sequences in flight at all times. Each request holds one of the $b$ slots for $o$ engine steps. The first step prefills its $p$ prompt tokens and produces its first output token, and the remaining $o - 1$ steps decode one token each. When a request finishes, the next request immediately takes its slot.

Over any cycle of $o$ consecutive steps, each of the $b$ slots contributes one prefill of $p$ tokens and $o - 1$ decode steps. Those $o$ steps take

$$
S(b) = o \cdot \text{base\_us} + b \cdot \big(\text{prefill\_us\_per\_token} \cdot p + \text{decode\_us\_per\_seq} \cdot (o - 1)\big)
\quad \text{microseconds.}
$$

- Each active sequence receives one token per step, so the mean step time $S(b) / o$ is the model's mean time per output token (**TPOT**).
- In $S(b)$ microseconds, all $b$ slots complete one request each. One replica therefore sustains $X(b) = b \cdot 10^6 / S(b)$ requests per second and $o \cdot X(b)$ output tokens per second.

**`plan(cost, workload, tpot_slo_us, headroom_percent)`**

1. **Batch size.** Find the largest integer $b$ with $1 \le b \le \text{max\_seqs}$ and $S(b) / o \le \text{tpot\_slo\_us}$. A TPOT equal to `tpot_slo_us` meets the objective. Call this batch size $b^\ast$. If no such $b$ exists, return `None`.
2. **Capacity.** One replica's capacity is $X(b^\ast)$ requests per second.
3. **Headroom.** Each replica is planned to run at no more than $(100 - \text{headroom\_percent})\%$ of its capacity. Set `replicas` to the smallest integer $R \ge 1$ satisfying

$$
\text{requests\_per\_s} \;\le\; R \cdot X(b^\ast) \cdot \frac{100 - \text{headroom\_percent}}{100}.
$$

Return a dict with exactly these keys:

| key | value |
|---|---|
| `"batch"` | $b^\ast$ (`int`) |
| `"tpot_us"` | $S(b^\ast) / o$ (`float`) |
| `"requests_per_s"` | $X(b^\ast)$ (`float`), one replica's capacity |
| `"tokens_per_s"` | $o \cdot X(b^\ast)$ (`float`) |
| `"replicas"` | $R$ (`int`) |
| `"utilization"` | $\text{requests\_per\_s} / (R \cdot X(b^\ast))$ (`float`), the fraction of capacity used |

### The measured curve

`curve` is a list of `(rate, latency_ms)` integer pairs from a load test of **one replica**. Each `rate` is the offered load in requests per second, and `latency_ms` is the measured latency in milliseconds (for example, p99 time to first token, or TTFT). Rates are strictly increasing. Between adjacent measured points, latency is defined by linear interpolation. The service-level objective is `slo_ms`, and a latency equal to `slo_ms` meets it.

Compute one replica's capacity $c$ (in requests per second) as follows:

- If the first point misses the objective (`latency_ms > slo_ms`), return `None`. Do not extrapolate below the first point.
- Otherwise, find the **first** point in increasing rate order that misses the objective, $(r_1, l_1)$, and its predecessor $(r_0, l_0)$, which meets it. Set $c$ to the rate where the line segment between them reaches `slo_ms`:

$$
c = r_0 + (\text{slo\_ms} - l_0) \cdot \frac{r_1 - r_0}{l_1 - l_0}.
$$

  Ignore all points beyond the first miss, even if a later point meets the objective.
- If no point misses the objective, set $c$ to the last measured rate. Do not extrapolate beyond the last point.

**`plan_from_curve(curve, requests_per_s, slo_ms, headroom_percent)`** returns `None` if the first point misses the objective. Otherwise, it returns `{"capacity": c, "replicas": R}`, where $c$ is a float and $R$ is the smallest integer $\ge 1$ satisfying $\text{requests\_per\_s} \le R \cdot c \cdot (100 - \text{headroom\_percent}) / 100$.

## Example

Suppose an engine step costs 8 ms plus 0.1 ms per prefilled token plus 1 ms per decoded sequence, and each replica runs at most 32 sequences. Requests have 500 prompt tokens and 200 output tokens, arrive at 40 requests per second, and must meet a mean TPOT of 40 ms with 25% headroom.

```python
cost = {"base_us": 8000, "prefill_us_per_token": 100, "decode_us_per_seq": 1000, "max_seqs": 32}
workload = {"requests_per_s": 40, "prompt_tokens": 500, "output_tokens": 200}

plan(cost, workload, 40_000, 25)
# {"batch": 25, "tpot_us": 39125.0, "requests_per_s": 3.194888178913738,
#  "tokens_per_s": 638.9776357827476, "replicas": 17, "utilization": 0.7364705882352941}
```

| step | arithmetic | value |
|---|---|---|
| one slot per cycle | $100 \cdot 500 + 1000 \cdot 199$ | 249,000 us |
| $S(b)$ | $200 \cdot 8000 + 249{,}000\,b$ | $1{,}600{,}000 + 249{,}000\,b$ |
| batch | $S(b) \le 200 \cdot 40{,}000 \implies b \le 6{,}400{,}000 / 249{,}000 = 25.7$ | 25 |
| TPOT | $S(25) / 200 = 7{,}825{,}000 / 200$ | 39,125 us |
| capacity | $25 \cdot 10^6 / 7{,}825{,}000$ | 3.195 requests/s |
| replicas | $40 / (3.195 \cdot 0.75) = 16.69$, rounded up | 17 |

With 0% headroom, the same workload needs `13` replicas ($40 / 3.195 = 12.52$). With a 100 ms TPOT objective, the batch hits `max_seqs`, giving `batch = 32`, `tpot_us = 47840.0`, and `16` replicas. With a 9 ms objective, `plan` returns `None` because even a single sequence takes $S(1) / 200 = 9245$ us per token.

Now suppose a single-replica load test measured p99 TTFT at five rates:

```python
curve = [(1, 180), (2, 210), (3, 290), (4, 520), (5, 1400)]

plan_from_curve(curve, 40, 400, 25)  # {"capacity": 3.4782608695652173, "replicas": 16}
plan_from_curve(curve, 40, 2000, 25)  # {"capacity": 5.0, "replicas": 11}
plan_from_curve(curve, 40, 150, 25)  # None
```

At a 400 ms objective, the curve crosses between 3 and 4 requests per second. Linear interpolation gives $c = 3 + (400 - 290) \cdot 1 / (520 - 290) = 3.478$, and $40 / (3.478 \cdot 0.75) = 15.33$ rounds up to 16 replicas. At 2000 ms, no point misses, so capacity is capped at the last measured rate (5.0). At 150 ms, the first point already misses, so the function returns `None`.

## Constraints

- `1 <= base_us, prefill_us_per_token, decode_us_per_seq <= 10**6` and `1 <= max_seqs <= 4096`.
- `1 <= requests_per_s <= 10**6`, `1 <= prompt_tokens <= 10**6`, `1 <= output_tokens <= 10**5`, and `1 <= tpot_slo_us, slo_ms <= 10**9`.
- `0 <= headroom_percent <= 90`.
- `curve` has between 1 and 1000 points. Rates and latencies are integers in $[1, 10^6]$.
- `"batch"` and `"replicas"` are compared exactly. Tests include loads that are exact multiples of the usable capacity, where floating-point division can land slightly above the true integer. Floats are compared with a relative tolerance of `1e-9`.

## Why this matters

Replica count determines the cost of serving a model, and teams run this calculation both before launch and on every autoscaler tick. Each rule in this specification prevents a common production mistake. Sizing for maximum throughput ignores latency and violates the SLO by design. Omitting prefill from step time under-provisions long-prompt workloads (yielding 14 replicas instead of 17 in the example).

Treating "25% headroom" as multiplying replicas by 1.25 instead of capping utilization at 75% runs the fleet hotter than planned. Rounding down or trusting a noisy measurement past the saturation knee hides shortfalls until peak traffic triggers queueing.
