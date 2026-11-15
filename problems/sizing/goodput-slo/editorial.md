## The idea

An SLO is a statement about individual requests ("the first token arrives within
500 ms"), and a target says how many of them it has to hold for ("for 90 percent of
requests"). So the unit of the computation is the request: decide for each one
whether it is good, then count.

```python
def is_good(request, slo):
    if not request["ok"]:
        return False
    for metric in ("ttft", "tpot", "e2e"):
        if metric in slo and request[metric] is not None and request[metric] > slo[metric]:
            return False
    return True
```

Three quantities come out of that predicate.

- **Attainment** is the good share of *all* requests sent. A failed request is a user
  who got nothing; leaving it out of the denominator would make a server look better
  the more it rejects.
- **Goodput at a load** is good requests per second. It is throughput with the
  requests that missed their objectives taken out, and it is at most the throughput.
  This is what `vllm bench serve --goodput` prints as request goodput: its code counts
  a completed request as good when every listed metric is at most its objective, and
  divides by the benchmark duration.
- **Goodput of a system** is a rate: the highest load at which attainment still meets
  the target. DistServe defines per-GPU goodput as the maximum request rate that can
  be served adhering to the SLO attainment goal (for example 90 percent), and
  attainment as the proportion of requests that meet the SLOs.

**Why the sweep stops at the first failure.** As load rises, queues grow and
attainment falls; the true curve is non-increasing. A measured curve is noisy, and a
short run above the knee can pass by luck. Taking the highest passing rate treats that
luck as capacity. Requiring every lower rate to pass reads the curve as what it is: a
monotone function observed with noise, whose knee is at the *first* crossing.

**Why not percentiles.** "At least 90 percent of requests have TTFT at most 0.5 s" is
the same statement as "the p90 of TTFT is at most 0.5 s" (with the nearest-rank
definition), so a single objective could be checked on a percentile. Two objectives
cannot: p90 TTFT and p90 TPOT can both pass while fewer than 90 percent of requests
meet both, because the slow-TTFT requests and the slow-TPOT requests need not be the
same ones. Goodput asks for the conjunction per request, which no pair of marginal
percentiles can express.

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

- **Checking the SLO on the mean.** In the example the mean TTFT (0.439 s) and the
  mean TPOT (0.049 s) both meet the objectives while four of ten requests do not. The
  mean also fails the other way: one request that took two minutes drags the mean of
  a hundred over the limit although 99 percent were fine.
- **Any objective instead of all.** A request with a fast first token and a crawling
  decode is not good. `any` where `all` is meant counts it.
- **The highest passing rate.** `max` over the passing rates returns 8.0 on the
  example sweep. Sort, and stop at the first failure.
- **Trusting the order of the runs.** Benchmarks are run in whatever order the
  operator chose, often high to low or interleaved. Walking the dict as given makes
  the answer depend on that.
- **`<` for `<=`.** A latency exactly at its limit meets it, and an attainment exactly
  at the target passes. vLLM's check is `slo >= measured`.
- **Dropping failures from the denominator.** Attainment over completed requests only
  turns load shedding into a way to raise the score.
- **Counting failures as good.** A rejection comes back in a millisecond. If the
  status is not checked first, the fastest "requests" of an overloaded server are its
  errors.
- **Failing one-token responses on TPOT.** `None <= 0.05` raises, and treating `None`
  as a violation fails every classification-style request. vLLM regards the TPOT of
  such a request as 0 for this purpose, which is the same decision.
- **Reporting throughput as goodput.** Dividing every completed request by the
  duration ignores the SLO altogether.

## Complexity

$O(n)$ for attainment and request goodput; $O(R \log R + N)$ for a sweep of $R$ rates
with $N$ requests in total.

## In the wild

- **vLLM** (`vllm bench serve`): `--goodput` takes `KEY:VALUE` pairs with the metric
  names `ttft`, `tpot`, and `e2el` and limits in milliseconds; the help text refers to
  DistServe for the definition. Source: the vLLM CLI documentation for `vllm bench
  serve` and `vllm/benchmarks/serve.py`.
- **DistServe** ("DistServe: Disaggregating Prefill and Decoding for Goodput-optimized
  Large Language Model Serving", arXiv:2401.09670, OSDI 2024) optimizes per-GPU
  goodput under TTFT and TPOT objectives; separating the two phases is its way of
  meeting both at a higher rate.
- **Finding the knee** in practice is a search rather than a fixed grid: double the
  rate until a run fails, then bisect between the last pass and the first failure,
  with runs long enough that attainment is not noise. The rule in this problem is
  what makes a grid and a bisection agree.
- **Whose clock.** These latencies must be measured by an open-loop load generator at
  the client. A closed-loop generator slows down with the server and never offers the
  rate it claims, so its sweep has no knee to find.
