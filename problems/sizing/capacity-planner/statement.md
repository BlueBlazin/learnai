Answer the question a capacity plan exists for: how many replicas does this workload
need at this latency objective? You answer it twice, once from the engine's cost
model and once from a measured latency curve of one replica. Both are a few lines of
arithmetic; the work is in the definitions, because "capacity", "headroom", and the
point where a curve "meets" an objective each have several plausible readings, and a
plan is only useful when two people computing it get the same integer.

## Specification

```python
def plan(cost: dict, workload: dict, tpot_slo_us: int, headroom_percent: int) -> dict | None: ...
def plan_from_curve(
    curve: list[tuple[int, int]], requests_per_s: int, slo_ms: int, headroom_percent: int
) -> dict | None: ...
```

Every input is an integer, and the integer results (`"batch"`, `"replicas"`) are
defined exactly: they must not depend on floating-point rounding.

### The cost model

`cost` describes one replica, in the terms of the kit's `SimEngine`:

| key | meaning |
|---|---|
| `base_us` | fixed time of one engine step, in microseconds |
| `prefill_us_per_token` | added per prompt token prefilled in the step |
| `decode_us_per_seq` | added per sequence decoded in the step |
| `max_seqs` | the most sequences a replica runs at once |

`workload` has `requests_per_s` (the total load), and `prompt_tokens` $p$ and
`output_tokens` $o$, the lengths every request is taken to have (the workload's means).

**A replica at batch $b$.** The replica runs continuous batching with $b$ sequences
in flight at all times. A request holds one of the $b$ slots for $o$ engine steps: the
step that prefills its $p$ prompt tokens, which also produces its first token, and
$o - 1$ steps that decode one token each. When it finishes, the next request takes
the slot. Over any $o$ consecutive steps, every slot therefore contributes one prefill
of $p$ tokens and $o - 1$ decodes, and those $o$ steps take

$$
S(b) = o \cdot \text{base} + b \cdot \big(\text{prefill} \cdot p + \text{decode} \cdot (o - 1)\big)
\quad \text{microseconds.}
$$

- The mean step time is $S(b) / o$. A running sequence receives one token per step,
  so this is the model's **TPOT**.
- In the time $S(b)$ each of the $b$ slots completes exactly one request, so the
  replica completes $X(b) = b \cdot 10^6 / S(b)$ requests per second and $o \cdot
  X(b)$ tokens per second.

**`plan(cost, workload, tpot_slo_us, headroom_percent)`**

1. *Batch.* $b^\ast$ is the largest integer $b$ with $1 \le b \le$ `max_seqs` and
   $S(b) / o \le$ `tpot_slo_us`. A TPOT exactly at the objective meets it. If no such
   $b$ exists the function returns `None`.
2. *Capacity.* One replica's capacity is $X(b^\ast)$ requests per second.
3. *Headroom.* A replica is planned to run at no more than
   $(100 - \text{headroom\_percent})$ percent of its capacity. `replicas` is the
   smallest integer $R \ge 1$ with

$$
\text{requests\_per\_s} \;\le\; R \cdot X(b^\ast) \cdot \frac{100 - \text{headroom\_percent}}{100}.
$$

It returns a dict with exactly these keys:

| key | value |
|---|---|
| `"batch"` | $b^\ast$, int |
| `"tpot_us"` | $S(b^\ast) / o$, float |
| `"requests_per_s"` | $X(b^\ast)$, float: one replica's capacity |
| `"tokens_per_s"` | $o \cdot X(b^\ast)$, float |
| `"replicas"` | $R$, int |
| `"utilization"` | $\text{requests\_per\_s} / (R \cdot X(b^\ast))$, float: the share of capacity the plan uses |

### The measured curve

`curve` is a list of `(rate, latency_ms)` pairs from a load test of **one replica**:
the request rate offered to it, in requests per second, and the latency measured at
that rate (a p99 TTFT, say), in milliseconds. Rates are strictly increasing. Between
measured points the curve is the straight line through its neighbors (linear
interpolation). The objective is `slo_ms`; a latency equal to it meets it.

**Capacity** $c$ of one replica, in requests per second:

- If the first point misses the objective (`latency > slo_ms`), there is no capacity:
  the function returns `None`. The curve is not extended below its first point.
- Otherwise find the **first** point, in order of rate, that misses the objective,
  $(r_1, l_1)$, and its predecessor $(r_0, l_0)$, which meets it. The capacity is
  where the line between them reaches the objective:

$$
c = r_0 + (\text{slo\_ms} - l_0) \cdot \frac{r_1 - r_0}{l_1 - l_0}.
$$

  Points beyond the first miss are ignored, even if they meet the objective.
- If no point misses the objective, $c$ is the last measured rate. The curve is not
  extended beyond its last point.

**`plan_from_curve(curve, requests_per_s, slo_ms, headroom_percent)`** returns `None`
as above, or `{"capacity": c, "replicas": R}` with $c$ as a float and $R$ the smallest
integer $\ge 1$ with $\text{requests\_per\_s} \le R \cdot c \cdot (100 -
\text{headroom\_percent}) / 100$, the same rule as in `plan`.

Neither function modifies its arguments.

## Example

An engine step costs 8 ms plus 0.1 ms per prefilled token plus 1 ms per decoded
sequence, and a replica runs at most 32 sequences. Requests have 500 prompt tokens and
200 output tokens and arrive at 40 per second; the objective is a mean TPOT of 40 ms.

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
| batch | $S(b) \le 200 \cdot 40{,}000$: $b \le 6{,}400{,}000 / 249{,}000 = 25.7$ | 25 |
| TPOT | $S(25) / 200 = 7{,}825{,}000 / 200$ | 39,125 us |
| capacity | $25 \cdot 10^6 / 7{,}825{,}000$ | 3.195 requests/s |
| replicas | $40 / (3.195 \cdot 0.75) = 16.69$, rounded up | 17 |

With no headroom the same plan needs `13` replicas ($40 / 3.195 = 12.52$). With a
TPOT objective of 100 ms the batch is capped by `max_seqs`: `batch` 32, `tpot_us`
47840.0, 16 replicas. With an objective of 9 ms the result is `None`: a single
sequence already takes $S(1) / 200 = 9245$ us per token.

A load test of one replica measured p99 TTFT at five rates:

```python
curve = [(1, 180), (2, 210), (3, 290), (4, 520), (5, 1400)]

plan_from_curve(curve, 40, 400, 25)  # {"capacity": 3.4782608695652173, "replicas": 16}
plan_from_curve(curve, 40, 2000, 25)  # {"capacity": 5.0, "replicas": 11}
plan_from_curve(curve, 40, 150, 25)  # None
```

The curve passes 400 ms between 3 and 4 requests per second:
$c = 3 + (400 - 290) \cdot 1 / (520 - 290) = 3.478$, and $40 / (3.478 \cdot 0.75) =
15.33$ gives 16 replicas. At 2000 ms no point misses, so the capacity is the last
measured rate, 5, although the line would reach 2000 ms only further out.

## Constraints

- `1 <= base_us, prefill_us_per_token, decode_us_per_seq <= 10**6`;
  `1 <= max_seqs <= 4096`.
- `1 <= requests_per_s <= 10**6`; `1 <= prompt_tokens <= 10**6`;
  `1 <= output_tokens <= 10**5`; `1 <= tpot_slo_us, slo_ms <= 10**9`.
- `0 <= headroom_percent <= 90`.
- `curve` has between 1 and 1000 points; rates and latencies are ints in
  $[1, 10^6]$.
- `"batch"` and `"replicas"` are compared exactly. The tests include loads that are
  an exact multiple of the usable capacity, where a float quotient lands just above
  the integer. Floats are compared with a relative tolerance of `1e-9`.

## Why this matters

Replicas are the unit in which inference is paid for, and the count is decided by
this calculation before anything is deployed and again by every autoscaler tick
afterward. Each definition here guards against a specific way of being wrong in
production. Plan a replica at the batch with the best throughput and the plan is
cheap and misses its latency objective by design. Leave the prefill out of the step
time and a workload of long prompts gets fewer replicas than it needs (14 instead of
17 in the example). Read
"25 percent headroom" as 1.25 times the replicas rather than 75 percent utilization
and the fleet runs hotter than agreed. Round to the nearest replica, or take the
capacity from a lucky measurement beyond the knee of the curve, and the shortfall
appears as queueing at peak traffic, which is the only time the plan mattered.
