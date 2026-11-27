## The idea

An SLO applies to individual requests (such as "first token within 500 ms"), and an attainment target specifies what fraction of requests must meet it (such as 90 percent). The calculation therefore starts at the request level by testing whether each request is good, then aggregating the counts.

```python
def is_good(request, slo):
    if not request["ok"]:
        return False
    for metric in ("ttft", "tpot", "e2e"):
        if metric in slo and request[metric] is not None and request[metric] > slo[metric]:
            return False
    return True
```

All three functions build on `is_good`:

- **Attainment** is the fraction of *all* sent requests that are good. Failed requests remain in the denominator because a rejected user still received a bad outcome. Excluding failures would let a server raise its score by rejecting traffic.
- **Request goodput** is good requests divided by benchmark duration. It is throughput with SLO-violating requests removed, so it never exceeds raw throughput. This matches `vllm bench serve --goodput`, which counts completed requests whose metrics all stay within their limits and divides by the benchmark duration.
- **System goodput** is the highest offered request rate at which attainment stays at or above `target`. This matches DistServe's definition of per-GPU goodput as the maximum request rate served while adhering to an attainment goal.

**Why the sweep stops at the first failure.** As offered load increases, queues grow and true attainment is non-increasing. Measured attainment has sampling noise, so a short run above the saturation knee can occasionally pass by chance. Taking the maximum passing rate treats that noise as real capacity. Requiring every lower rate to pass finds the first crossing of the threshold.

**Why percentiles cannot replace per-request checks.** For a single metric, "at least 90 percent of requests have TTFT $\le$ 0.5 s" matches "p90 TTFT $\le$ 0.5 s" under nearest-rank percentiles. With two metrics, marginal percentiles break down. Both p90 TTFT and p90 TPOT can pass while fewer than 90 percent of requests satisfy both at once, because different requests can be slow on each metric. Goodput requires the conjunction on every individual request.

## Reference solution (Python)

```python
def slo_attainment(requests, slo):
    if not requests:
        raise ValueError("attainment of no requests is undefined")
    return sum(is_good(r, slo) for r in requests) / len(requests)


def request_goodput(requests, slo, duration_s):
    return sum(is_good(r, slo) for r in requests) / duration_s


def goodput(runs, slo, target):
    best = None
    for rate in sorted(runs):
        if slo_attainment(runs[rate], slo) < target:
            break
        best = rate
    return best
```

## Common mistakes

- **Checking the SLO on mean latency.** In the example, mean TTFT (0.439 s) and mean TPOT (0.049 s) both meet the SLO even though four of ten requests fail it. Means also fail in the opposite direction when one extreme outlier drags the average over the limit while 99 percent of requests were fast.
- **Using `any` instead of `all`.** A request with a fast first token and a slow decode is not good. Every metric present in `slo` must pass.
- **Taking the highest passing rate.** Using `max` over passing rates returns `8.0` on the example sweep. Sort the rates and stop at the first failure.
- **Trusting dict iteration order.** Benchmarks are often run from high to low rate or in an interleaved order. Iterating over `runs` without sorting makes the result depend on insertion order.
- **Using `<` instead of `<=`.** A latency equal to its limit passes, and an attainment equal to `target` passes. vLLM uses the same `slo >= measured` convention.
- **Dropping failures from the denominator.** Computing attainment over completed requests only rewards load shedding.
- **Counting failed requests as good.** A rejected request often returns in a millisecond. If you do not check `request["ok"]` first, errors look like the fastest requests on an overloaded server.
- **Failing one-token responses on TPOT.** Comparing `None <= 0.05` raises a `TypeError`, and treating `None` as a violation fails every single-token response. vLLM treats a one-token response's TPOT as 0 for SLO checks, which gives the same result.
- **Reporting raw throughput as goodput.** Dividing all completed requests by `duration_s` ignores the SLO entirely.

## Complexity

For $n$ requests in a single run, `slo_attainment` and `request_goodput` run in $O(n)$ time and $O(1)$ extra space. For a sweep of $R$ rates with $N$ total requests across all runs, `goodput` runs in $O(R \log R + N)$ time and $O(R)$ space to sort the rate keys.

## In the wild

- **vLLM (`vllm bench serve`)** accepts `--goodput KEY:VALUE` pairs for `ttft`, `tpot`, and `e2el` with limits in milliseconds, referencing DistServe in its CLI documentation (`vllm/benchmarks/serve.py`).
- **DistServe** (*DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*, arXiv:2401.09670, OSDI 2024) optimizes per-GPU goodput under joint TTFT and TPOT objectives by disaggregating prefill and decode onto separate pools.
- **Finding the knee in practice** usually uses exponential search followed by binary search rather than a fixed grid. You double the rate until a run fails, then bisect between the last passing rate and the first failing rate with long enough runs to suppress noise. Stopping at the first failure makes a grid sweep agree with bisection.
- **Open-loop load generation** is required for valid latency measurements. A closed-loop client waits for responses before sending the next request, so it slows down with the server and never sustains the offered rate needed to locate the knee.
