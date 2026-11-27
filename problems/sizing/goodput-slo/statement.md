Implement `slo_attainment`, `request_goodput`, and `goodput` to evaluate serving latency against service-level objectives (SLOs). Together, they measure the share of requests that meet every objective, the rate of qualifying requests completed per second, and the maximum request rate a system can sustain under an attainment target.

## Specification

```python
def slo_attainment(requests: list[dict], slo: dict) -> float: ...
def request_goodput(requests: list[dict], slo: dict, duration_s: float) -> float: ...
def goodput(runs: dict[float, list[dict]], slo: dict, target: float) -> float | None: ...
```

**A request** is a dict recorded by a load generator with these exact keys:

| key | value |
|---|---|
| `"ok"` | `True` if the request completed, `False` if it failed (error status, rejection, or timeout) |
| `"ttft"` | time to first token in seconds |
| `"tpot"` | time per output token in seconds, excluding the first token (as in `latency-metrics`), or `None` for a one-token response |
| `"e2e"` | end-to-end latency in seconds |

When `"ok"` is `False`, the latency fields may be `None` or numbers (since a rejected request still has a response time). Ignore them.

**An SLO** is a dict containing any subset of the keys `"ttft"`, `"tpot"`, and `"e2e"`, each mapping to a latency limit in seconds. Any metric omitted from `slo` is unconstrained, and `slo` may be empty.

**A good request** satisfies two conditions:

1. It completed (`request["ok"]` is `True`).
2. It meets **every** objective in `slo`. For each metric key $m$ in `slo`, either `request[m]` is `None` or $\text{request}[m] \le \text{slo}[m]$. A latency equal to its limit passes. A `"tpot"` of `None` passes any `"tpot"` objective because a one-token response has no inter-token gap.

**`slo_attainment(requests, slo)`** returns the fraction of all requests that are good as a float (`good / len(requests)`):

$$
\text{attainment} = \frac{\#\{\text{good requests}\}}{\#\{\text{all requests}\}}.
$$

Failed requests stay in the denominator. Raise `ValueError` if `requests` is empty.

**`request_goodput(requests, slo, duration_s)`** returns the number of good requests per second over a benchmark lasting `duration_s > 0` seconds, computed as the float `good / duration_s`. If `requests` is empty, return `0.0`.

**`goodput(runs, slo, target)`** summarizes a load sweep. `runs` maps each offered request rate (in requests per second) to the non-empty list of requests measured at that rate. A rate **passes** when `slo_attainment(runs[rate], slo) >= target`, where `target` is in $(0, 1]$. Return the largest rate $r$ in `runs` such that $r$ **and every rate in `runs` below it** pass, preserving the exact key object from `runs`. If the lowest rate fails, return `None`. Do not assume the keys of `runs` are sorted.

None of the functions may modify its arguments.

> [!NOTE]
> Two distinct metrics are called goodput, and this problem implements both. vLLM's benchmark (`vllm bench serve --goodput ttft:500 tpot:50`) reports `request_goodput`, the rate of completed requests that met all objectives over a single run. The DistServe paper defines `goodput` as a system capacity metric, namely the maximum offered request rate that still meets an attainment target such as 90 percent.

## Example

Suppose the SLO requires a first token within 500 ms and at most 50 ms per output token after that, and ten requests are measured over four seconds:

```python
slo = {"ttft": 0.5, "tpot": 0.05}
requests = [
    {"ok": True, "ttft": 0.20, "tpot": 0.030, "e2e": 1.7},  # good
    {"ok": True, "ttft": 0.25, "tpot": 0.040, "e2e": 2.2},  # good
    {"ok": True, "ttft": 0.90, "tpot": 0.030, "e2e": 2.4},  # slow first token
    {"ok": True, "ttft": 0.30, "tpot": 0.050, "e2e": 2.8},  # good: exactly at the tpot limit
    {"ok": True, "ttft": 0.35, "tpot": 0.080, "e2e": 4.3},  # slow decode
    {"ok": True, "ttft": 0.50, "tpot": 0.045, "e2e": 2.7},  # good: exactly at the ttft limit
    {"ok": True, "ttft": 0.40, "tpot": None, "e2e": 0.4},  # good: one token
    {"ok": False, "ttft": None, "tpot": None, "e2e": None},  # failed
    {"ok": True, "ttft": 0.45, "tpot": 0.049, "e2e": 2.9},  # good
    {"ok": True, "ttft": 0.60, "tpot": 0.070, "e2e": 4.1},  # misses both
]

slo_attainment(requests, slo)  # 0.6
request_goodput(requests, slo, 4.0)  # 1.5
slo_attainment(requests, {**slo, "e2e": 2.5})  # 0.3
slo_attainment(requests, {})  # 0.9
```

Six of the ten requests are good. The server completed nine requests in four seconds (a raw throughput of 2.25 requests per second), giving a request goodput of 1.5 requests per second. Meanwhile, the nine completed requests have a mean TTFT of 0.439 s and a mean TPOT of 0.049 s. Both averages sit inside the SLO even though four of the ten requests failed to meet it.

Now consider a sweep over five request rates, recorded in arbitrary order, with an attainment target of 0.9:

| rate (req/s) | requests | good | attainment | passes at 0.9 |
|---|---|---|---|---|
| 1.0 | 20 | 20 | 1.00 | yes |
| 2.0 | 20 | 19 | 0.95 | yes |
| 4.0 | 40 | 34 | 0.85 | no |
| 8.0 | 40 | 36 | 0.90 | yes |
| 16.0 | 40 | 11 | 0.275 | no |

```python
goodput(runs, slo, 0.9)  # 2.0
goodput(runs, slo, 0.8)  # 8.0
goodput(runs, slo, 0.99)  # 1.0
```

At a target of `0.9`, the result is `2.0` rather than `8.0`. Even though the run at `8.0` passed, the lower rate `4.0` failed, so the pass at `8.0` reflects short-run noise rather than sustainable capacity.

## Constraints

- `0 <= len(requests) <= 100,000` per call.
- `runs` contains between 1 and 100 distinct positive rates (ints or floats), each with at least one request.
- Latencies and SLO limits are finite floats in $[0, 10^6]$, and `0 < target <= 1`.
- For every test run, the float comparison `good / n >= target` matches exact rational arithmetic.
- `slo_attainment` and `request_goodput` are compared with relative and absolute tolerance `1e-12`. `goodput` is compared exactly, including its type (the original key from `runs`, or `None`).

## Why this matters

Raw throughput can always be inflated by ignoring latency. Admitting every request and batching aggressively pushes tokens per second upward while users wait ten seconds for a first token. Capacity only means something alongside the latency objective it satisfies, and goodput captures that tradeoff for tuning serving configs and estimating cost per request.

Small definition bugs distort capacity estimates in dangerous ways. Checking an SLO against mean latency approves a system where a third of users had a poor experience. Dropping failed requests from the denominator rewards a server for shedding load. Summarizing a rate sweep by its highest passing point overestimates the knee of the latency curve, which only surfaces in production when real traffic hits the rate the benchmark claimed was safe.
