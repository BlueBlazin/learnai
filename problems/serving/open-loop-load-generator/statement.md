Implement an open-loop load generator for a benchmark. It sends requests according to a Poisson arrival schedule that does not depend on server response times, and it measures each request's latency from the moment the request was scheduled to be sent.

![This problem is the Client, here a benchmark that sends on a fixed schedule.](assets/map.svg)

## Specification

```python
async def generate_load(
    send, rate: float, n: int, uniforms: list[float], max_in_flight: int | None = None
) -> list[dict]: ...
```

`send` is an async callable. Calling `await send(i)` performs request `i` (an integer from `0` to `n - 1`) and either returns when the response completes or raises an `OSError` (such as `ConnectionError` or `TimeoutError`) if the request fails.

Use only the running event loop's clock (`asyncio.get_running_loop().time()` and `asyncio.sleep`, in seconds), which is simulated under the judge. Do not read the real wall clock, and do not generate random numbers. All randomness comes from `uniforms`, which has at least `n` elements in $[0, 1)$. Only the first `n` elements are used.

**The schedule.** Let $t_0$ be the loop time when `generate_load` is called, and let $u_j = \text{uniforms}[j]$. Each gap $g_j$ is drawn from an exponential distribution with mean $1 / \text{rate}$ seconds, and request $i$ has the **intended send time** $s_i$:

$$
g_j = \frac{-\ln(1 - u_j)}{\text{rate}}, \qquad s_i = t_0 + \sum_{j=0}^{i} g_j.
$$

These gaps form a Poisson arrival process at `rate` requests per second, with request `0` due one gap after $t_0$. When $u_j = 0$, the gap $g_j$ is `0`, so two requests are due at the same instant. The schedule depends only on $t_0$, `rate`, and `uniforms`. It never depends on when responses return (an **open loop**).

**Sending.**

- When `max_in_flight` is `None`, call `send(i)` at $s_i$ regardless of how many earlier requests are still in flight.
- When `max_in_flight` is an integer $m$, at most $m$ requests may be in flight at once. A request is in flight from the moment `send(i)` is called until it returns or raises. If $m$ requests are already in flight when request $i$ is due, it waits. Waiting requests are sent in request order (first come, first served), so `send(i)` is called at the earliest instant at or after $s_i$ when fewer than $m$ requests are in flight and all earlier requests have been sent. Waiting never shifts any request's intended send time.

**The result.** `generate_load` returns at the instant the last request finishes (or immediately at $t_0$ if `n == 0`). It returns a list of `n` dicts in request order, each with these exact keys:

| key | value |
|---|---|
| `"intended_s"` | $s_i$, the scheduled send time |
| `"sent_s"` | the time when `send(i)` was called |
| `"done_s"` | the time when `send(i)` returned or raised |
| `"latency_s"` | `done_s - intended_s`, the latency measured from the intended send time |
| `"ok"` | `True` if `send(i)` returned, `False` if it raised an `OSError` |

Record failed requests with `"ok": False` and their timestamps just like successful requests. Do not propagate the `OSError`, and do not let a failure affect other requests.

## Example

```python
rate = 2.0
uniforms = [1 - math.exp(-g * rate) for g in (0.5, 0.25, 0.25, 1.0)]  # the gaps, in seconds
```

Suppose `generate_load` is called at $t_0 = 0$. The four requests are due at 0.5 s, 0.75 s, 1.0 s, and 2.0 s. Request 0 takes 1.0 s to complete, and requests 1, 2, and 3 take 0.25 s each.

With `max_in_flight=None`, every request is sent at its intended time:

| request | intended | sent | done | latency |
|---|---|---|---|---|
| 0 | 0.5 | 0.5 | 1.5 | 1.0 |
| 1 | 0.75 | 0.75 | 1.0 | 0.25 |
| 2 | 1.0 | 1.0 | 1.25 | 0.25 |
| 3 | 2.0 | 2.0 | 2.25 | 0.25 |

Requests 1 and 2 are sent and completed while request 0 is still in flight. `generate_load` returns at 2.25 s.

With `max_in_flight=1`, requests 1 and 2 must wait for request 0 to finish:

| request | intended | sent | done | latency |
|---|---|---|---|---|
| 0 | 0.5 | 0.5 | 1.5 | 1.0 |
| 1 | 0.75 | 1.5 | 1.75 | 1.0 |
| 2 | 1.0 | 1.75 | 2.0 | 1.0 |
| 3 | 2.0 | 2.0 | 2.25 | 0.25 |

![The example without a limit and with `max_in_flight=1`. With the limit, requests 1 and 2 wait for request 0, and their latency of 1.0 s includes the wait.](assets/send-times.svg)

Three of the four requests took 1.0 s from the moment their user wanted an answer. If latency were measured from `sent_s` instead, the latencies would be 1.0 s, 0.25 s, 0.25 s, and 0.25 s, reporting only one slow request instead of three. Request 3 is still due and sent at 2.0 s because earlier waits do not shift the schedule.

## Constraints

- `0 <= n <= 20,000`, `rate > 0`, and `max_in_flight` is `None` or an integer `>= 1`.
- All timestamps are checked with an absolute tolerance of `1e-9` s, including the times when `send` is called, every field of every record, and the time when `generate_load` returns. When `max_in_flight` is set, the in-flight count is checked on every `send` call.
- **Statistical check.** Over 20,000 requests at 50 requests/s against a server that takes 80 ms per response, the gaps $g$ between consecutive `send` calls are tested using the mean of $e^{-\text{rate}\cdot g}$. That mean must be within the Hoeffding band around $\tfrac12$ for a failure probability of $10^{-10}$ (about $\pm 0.024$). Exponential gaps have an expected value of $\tfrac12$, whereas uniform gaps with the same mean give 0.43 and constant gaps give 0.37.
- **Simultaneous events.** Requests due at the same instant are sent in request order. If an in-flight slot opens at the exact instant a request is due, that request is sent at that instant. No other test depends on the ordering of simultaneous events.

## Why this matters

Capacity numbers come from load tests, and a load test is only as accurate as its arrival model. Real users do not wait for other users' responses before sending their own requests. A generator that waits for each response before sending the next slows down whenever the server slows down. During a server stall it sends almost nothing, so it records one slow request where real traffic would have queued hundreds, hiding tail latency.

This failure mode is called **coordinated omission**. Avoiding it requires both an arrival schedule fixed in advance and latency measured from each request's scheduled send time, so any request delayed behind a full connection pool is charged for its wait. Tools like `vllm bench serve` take a `--request-rate` flag for this reason, and rate sweeps such as `goodput-slo` depend on it.
