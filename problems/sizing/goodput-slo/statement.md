Compute goodput: the part of a server's throughput that was worth having, because the
requests behind it met their latency objectives. Counting is all it takes; the
substance is what counts as a good request, what goes in the denominator, and how a
sweep over request rates turns into one number that can be called capacity.

## Specification

```python
def slo_attainment(requests: list[dict], slo: dict) -> float: ...
def request_goodput(requests: list[dict], slo: dict, duration_s: float) -> float: ...
def goodput(runs: dict[float, list[dict]], slo: dict, target: float) -> float | None: ...
```

**A request** is a dict with exactly these keys, measured by a load generator:

| key | value |
|---|---|
| `"ok"` | `True` if the request completed, `False` if it failed (an error status, a rejection, a timeout) |
| `"ttft"` | time to first token in seconds |
| `"tpot"` | time per output token in seconds, first token excluded as in `latency-metrics`; `None` for a response of one token |
| `"e2e"` | end-to-end latency in seconds |

When `"ok"` is `False` the three latencies may be `None` or numbers (a rejection has a
latency too); they are not looked at.

**An SLO** (service-level objective) is a dict with any subset of the keys `"ttft"`,
`"tpot"`, and `"e2e"`, each mapping to a limit in seconds. A metric that is not a key
is unconstrained. The dict may be empty.

**A good request** is one that

1. completed (`"ok"` is `True`), and
2. meets **every** objective in the SLO: for each key $m$ of `slo`,
   $\text{request}[m] \le \text{slo}[m]$. A latency exactly equal to its limit meets
   it. A `"tpot"` of `None` meets any `"tpot"` objective: a one-token response has no
   inter-token gap that could be slow.

**`slo_attainment(requests, slo)`** returns the fraction of requests that are good:

$$
\text{attainment} = \frac{\#\{\text{good requests}\}}{\#\{\text{all requests}\}},
$$

a float computed as `good / len(requests)`. Failed requests are in the denominator. An
empty `requests` raises `ValueError`.

**`request_goodput(requests, slo, duration_s)`** returns the good requests per second
of a benchmark that lasted `duration_s > 0` seconds: `good / duration_s`, a float. An
empty list gives `0.0`.

**`goodput(runs, slo, target)`** summarizes a rate sweep. `runs` maps a request rate
in requests per second (the load the generator offered) to the non-empty list of
requests measured at that rate. A rate **passes** when
`slo_attainment(runs[rate], slo) >= target`. The function returns the largest rate $r$
in `runs` such that $r$ **and every rate in `runs` below it** pass, exactly as it
appears as a key; if the lowest rate fails it returns `None`. The order of the keys in
the dict carries no meaning. `target` is in $(0, 1]$.

None of the functions modifies its arguments.

> [!NOTE]
> Two things are called goodput, and this problem implements both. vLLM's benchmark
> (`vllm bench serve --goodput ttft:500 tpot:50`) reports the first: requests per
> second that met all the given objectives, good requests divided by the benchmark's
> duration. The DistServe paper uses the word for the second: the maximum request rate
> that can be served while the attainment stays at or above a goal such as 90 percent.
> The first is a property of one run at one load; the second is a property of the
> system.

## Example

The objectives are a first token within 500 ms and at most 50 ms per token after it.
Ten requests were measured over four seconds:

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

Six of ten requests are good. The server completed nine requests in four seconds, a
throughput of 2.25 requests per second, of which 1.5 per second were goodput. Note
what the averages say: the mean TTFT of the nine completed requests is 0.439 s and
their mean TPOT is 0.049 s, both inside the objectives that four of them missed.

A sweep over five rates, run in no particular order, with an attainment target of 90
percent:

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

At a target of 0.9 the answer is 2.0, not 8.0: the run at 8.0 passed, but a lower
rate failed, so that pass is noise from a short run and not a rate the system
sustains.

## Constraints

- `0 <= len(requests) <= 100,000` per call; `runs` has between 1 and 100 rates, all
  distinct and positive (ints or floats), each with at least one request.
- Latencies and limits are finite floats in $[0, 10^6]$; `0 < target <= 1`.
- For every run in the tests, the float comparison `good / n >= target` gives the
  same answer as the exact one.
- `slo_attainment` and `request_goodput` are compared with a relative and absolute
  tolerance of `1e-12`; the result of `goodput` is compared exactly, including its
  type (the key itself, or `None`).

## Why this matters

Throughput can always be raised by ignoring latency: admit everything, batch it all,
and tokens per second climbs while every user waits ten seconds for a first token. A
capacity number is only useful with the latency it was measured at, and goodput is
that number: it is what a serving configuration is tuned for and what a cost per
request is computed from. The details decide the answer. An SLO checked against mean
latency passes a system in which a third of the users had a bad time. A denominator
without the failed requests rewards a server for rejecting load. And a sweep
summarized by its best passing run reports the knee of the curve too high, which is
found out in production, at the worst moment, when traffic reaches the rate the
benchmark said was safe.
