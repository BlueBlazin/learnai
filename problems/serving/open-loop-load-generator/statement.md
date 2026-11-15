Write the load generator of a benchmark: it sends requests at times drawn from a
Poisson process, whatever the server does with them, and measures each latency from
the moment the request was *due*. Both halves are easy to get subtly wrong, and both
mistakes make an overloaded server look healthy.

## Specification

```python
async def generate_load(
    send, rate: float, n: int, uniforms: list[float], max_in_flight: int | None = None
) -> list[dict]: ...
```

`send` is an async callable. `await send(i)` performs request `i` (an int from 0 to
`n - 1`): it returns when the response is complete, or raises an `OSError` (a
`ConnectionError`, a `TimeoutError`) if the request failed. It takes time. Time is the running event loop's clock,
`asyncio.get_running_loop().time()` and `asyncio.sleep`, in seconds; under the judge
it is simulated. Do not use the real clock, and do not draw random numbers: the
randomness is `uniforms`.

**The schedule.** Let $t_0$ be the time at which `generate_load` is called. Request
$i$ has the **intended send time**

$$
s_i = t_0 + \sum_{j=0}^{i} g_j, \qquad g_j = \frac{-\ln(1 - u_j)}{\text{rate}},
$$

with $u_j$ = `uniforms[j]`, a number in $[0, 1)$. The gaps $g_j$ are exponential
with mean $1 / \text{rate}$ seconds, so the arrivals form a Poisson process of `rate`
requests per second; the first request is due one gap after $t_0$. A uniform of 0
gives a gap of 0: two requests due at the same instant. The schedule depends on
nothing but $t_0$, `rate`, and `uniforms`. In particular it does not depend on when
any response arrives (an **open loop**).

**Sending.**

- With `max_in_flight=None`, `send(i)` is called at $s_i$, however many earlier
  requests are still in flight.
- With `max_in_flight = m`, at most $m$ requests are in flight at once (a request is
  in flight from the call of `send` until it returns or raises). A request that is
  due while $m$ are in flight waits, and waiting requests are sent in request order,
  first come, first served: `send(i)` is called at the first instant at or after
  $s_i$ at which fewer than $m$ requests are in flight and every earlier request has
  been sent. A wait never changes any intended time.

**The result.** `generate_load` returns when every request has ended, at the instant
the last one ends (at $t_0$ if `n` is 0). It returns a list of `n` dicts, in request
order, each with exactly these keys:

| key | value |
|---|---|
| `"intended_s"` | $s_i$ |
| `"sent_s"` | the time at which `send(i)` was called |
| `"done_s"` | the time at which `send(i)` returned or raised |
| `"latency_s"` | `done_s - intended_s`: the latency **from the intended send time** |
| `"ok"` | `True` if `send(i)` returned, `False` if it raised an `OSError` |

A request that fails is recorded with `"ok": False` and its times like any other;
the exception is not propagated and the other requests are not affected.

`uniforms` has at least `n` elements; only the first `n` are used.

## Example

```python
rate = 2.0
uniforms = [1 - math.exp(-g * rate) for g in (0.5, 0.25, 0.25, 1.0)]  # the gaps, in seconds
```

`generate_load` is called at $t_0 = 0$, so the requests are due at 0.5, 0.75, 1.0,
and 2.0. Request 0 takes 1 s; the others take 0.25 s.

Without a limit, every request is sent when it is due:

| request | intended | sent | done | latency |
|---|---|---|---|---|
| 0 | 0.5 | 0.5 | 1.5 | 1.0 |
| 1 | 0.75 | 0.75 | 1.0 | 0.25 |
| 2 | 1.0 | 1.0 | 1.25 | 0.25 |
| 3 | 2.0 | 2.0 | 2.25 | 0.25 |

Requests 1 and 2 are sent and answered while request 0 is still in flight. The call
returns at 2.25.

With `max_in_flight=1`, requests 1 and 2 have to wait for request 0:

| request | intended | sent | done | latency |
|---|---|---|---|---|
| 0 | 0.5 | 0.5 | 1.5 | 1.0 |
| 1 | 0.75 | 1.5 | 1.75 | 1.0 |
| 2 | 1.0 | 1.75 | 2.0 | 1.0 |
| 3 | 2.0 | 2.0 | 2.25 | 0.25 |

Three of the four requests took a second from the moment their user wanted an
answer. Measured from `sent_s` instead, the latencies would be 1.0, 0.25, 0.25, 0.25:
one slow request instead of three. Request 3 is still due at 2.0 and is sent at 2.0;
the waiting did not move the schedule.

## Constraints

- `0 <= n <= 20,000`; `rate > 0`; `max_in_flight` is `None` or an int `>= 1`.
- Times are compared with an absolute tolerance of `1e-9` s: the times at which
  `send` is called, every field of every record, and the time at which
  `generate_load` returns. With a limit, the number of requests in flight is checked
  at every send.
- **Statistical check.** With 20,000 requests at 50 per second against a server that
  takes 80 ms per response, the gaps $g$ between consecutive `send` calls must
  satisfy: the mean of $e^{-\text{rate}\cdot g}$ is $\tfrac12$ within the Hoeffding
  band for a failure probability of $10^{-10}$ (about $\pm 0.024$). Exponential gaps
  give exactly $\tfrac12$; uniform gaps with the same mean give 0.43, constant gaps
  0.37.
- Simultaneous events: requests due at the same instant are sent in request order.
  If a slot becomes free at the very instant a request is due, the request is sent
  at that instant. Nothing else in the tests depends on the order of simultaneous
  events.

## Why this matters

A capacity number comes out of a load test, and a load test is only as honest as its
arrival process. Real users do not wait for each other's responses before sending
their own; a generator that does (one request per connection at a time, the next one
after the answer) slows down exactly when the server does. During a stall it sends
almost nothing, so it records one slow request where real traffic would have
produced hundreds, and the p99 it reports is the p99 of a load that never existed.
This is called coordinated omission. The fix has two parts and needs both: an
arrival schedule that is fixed in advance, and latency counted from the scheduled
time, so that a request stuck behind a full connection pool is charged for its wait.
`vllm bench serve` and the tools the sizing track's metrics come from take a
`--request-rate` for the same reason; `goodput-slo`'s rate sweep means nothing
without it.
