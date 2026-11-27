Implement an admission controller that sits in front of an inference server's queue. It admits a request only while the queue stays within a request-count bound and a prompt-token bound, and it returns a retry estimate whenever a request is rejected. Without this gate, overload makes the queue and request latency grow without limit.

![This problem is Admission, which rejects a request when the queue is over its bounds.](assets/map.svg)

## Specification

```python
@dataclass(frozen=True)
class Admitted: ...

@dataclass(frozen=True)
class Rejected:
    retry_after_s: float | None

class Admission:
    def __init__(self, max_queued_requests: int, max_queued_tokens: int,
                 drain_tokens_per_s: float): ...
    def admit(self, req) -> Admitted | Rejected: ...
    def start(self, req) -> None: ...
    def finish(self, req) -> None: ...
    queued_requests: int      # read-only properties
    queued_tokens: int
    running_requests: int
```

`Admitted` and `Rejected` are provided in the starter. Each request `req` has a unique hashable `id` and an integer `prompt_tokens >= 1`. The tests pass [`SimRequest`](../../../python/src/learnai/kit/workloads.py) objects from the kit.

**Request lifecycle.** When a request arrives, the server calls `admit(req)`. If it returns `Admitted()`, the request is *queued*. When the server begins executing the request, it calls `start(req)`. The request is now *running* and no longer queued. When execution completes, the server calls `finish(req)`. A rejected request is not stored, and the same request may be passed to `admit` again later.

**Queue bounds.** The bounds apply only to queued requests. Let $Q$ be the current number of queued requests and $T$ be the sum of their `prompt_tokens`. For an incoming request with $p = \text{req.prompt\_tokens}$, `admit(req)` behaves as follows:

- If $Q + 1 \le \text{max\_queued\_requests}$ and $T + p \le \text{max\_queued\_tokens}$, append the request to the end of the queue and return `Admitted()`.
- Otherwise, leave the queue unchanged and return `Rejected(retry_after_s)`.

Running requests do not count against either bound. Their queue slot and prompt tokens are released when `start(req)` is called.

**Retry estimate (`retry_after_s`).** This value estimates how long the client should wait before the same request would fit, assuming no new requests arrive, queued requests leave oldest first, and the queue drains at `drain_tokens_per_s` prompt tokens per second. Let the currently queued requests be $q_1, \dots, q_Q$ in admission order (oldest first), with prompt token counts $p_1, \dots, p_Q$.

- If $p > \text{max\_queued\_tokens}$, the request can never be admitted, so `retry_after_s` is `None`.
- Otherwise, let $j \ge 1$ be the smallest number of oldest queued requests that must leave before the incoming request fits. That is, $j$ is the smallest integer with $(Q - j) + 1 \le \text{max\_queued\_requests}$ and $(T - p_1 - \dots - p_j) + p \le \text{max\_queued\_tokens}$. Then

$$
\text{retry\_after\_s} = \frac{p_1 + \dots + p_j}{\text{drain\_tokens\_per\_s}}.
$$

Such a $j$ always exists because any request with $p \le \text{max\_queued\_tokens}$ fits into an empty queue.

**Out-of-order starts.** `start(req)` may be called for any currently queued request, not only the oldest, because a scheduler may reorder work. In the `retry_after_s` calculation, "oldest first" always refers to the requests currently in the queue at the time of `admit`, ordered by when they were admitted.

**Properties.** `queued_requests` returns $Q$, `queued_tokens` returns $T$, and `running_requests` returns the number of requests currently between `start` and `finish`.

The tests only call `start` on a queued request and `finish` on a running request, and never pass a request whose `id` is already queued or running. `Admission` does not read any clock or sleep. Every method returns immediately.

## Example

Consider `Admission(max_queued_requests=3, max_queued_tokens=1000, drain_tokens_per_s=500)`:

| call | result | queue after (tokens) | reason |
|---|---|---|---|
| `admit(a)`, 400 tokens | `Admitted()` | a (400) | |
| `admit(b)`, 300 tokens | `Admitted()` | a, b (700) | |
| `admit(c)`, 400 tokens | `Rejected(0.8)` | a, b (700) | 700 + 400 > 1000; fits after a leaves (400 / 500) |
| `admit(d)`, 200 tokens | `Admitted()` | a, b, d (900) | |
| `admit(e)`, 50 tokens | `Rejected(0.8)` | a, b, d (900) | 4th request exceeds limit; fits after a leaves (400 / 500) |
| `admit(f)`, 700 tokens | `Rejected(1.4)` | a, b, d (900) | after a, 500 + 700 > 1000; after a and b, 200 + 700 fits ((400 + 300) / 500) |
| `admit(g)`, 1200 tokens | `Rejected(None)` | a, b, d (900) | 1200 > 1000, so it can never fit |
| `start(a)` | | b, d (500) | a is now running (`running_requests` is 1) |
| `admit(c)`, 400 tokens | `Admitted()` | b, d, c (900) | |
| `finish(a)` | | b, d, c (900) | `running_requests` is 0 |

## Constraints

- `1 <= max_queued_requests <= 1000`, `1 <= max_queued_tokens <= 10^6`, and `drain_tokens_per_s > 0`, with at most a few hundred requests per test.
- Admission decisions are compared exactly. `retry_after_s` is checked within `1e-9` (the tests choose drain rates and token counts whose quotients are exact).
- **Simulation tests.** Alongside unit sequences, the tests place `Admission` in front of a single-slot simulated server that runs each request for `prompt_tokens / 1024` seconds, with `drain_tokens_per_s = 1024`. When the slot frees up, the server starts the oldest queued request (except in one case that starts the shortest prompt). At any single timestamp, a completion and the start that follows it occur before new arrivals at that timestamp. Arrival rates reach up to three times the server's sustainable rate.
- In these simulations, the tests verify that actual queued requests and queued prompt tokens never exceed the bounds, that all three properties match the true state after every call, and that every decision matches the specification. For the first-come, first-served server, the tests also verify that every admitted request finishes within `(max_queued_tokens + longest_prompt) / 1024` seconds of arrival, because the currently running request takes at most one `longest_prompt` and all queued work up to and including the new request is at most `max_queued_tokens`.

## Why this matters

An inference server's throughput is capped by its hardware. When requests arrive faster than the server can finish them, the only thing that can grow is the queue. An unbounded queue turns overload into runaway latency, where every request finishes minutes late—long after clients have timed out and retried, compounding the load.

A bounded queue turns overload into fast, explicit rejections (HTTP 429 or 503 with a `Retry-After` header) that a load balancer can route to another replica, while keeping the latency of admitted requests within a predictable bound. Bounding prompt tokens is just as critical as bounding request count, because ten queued requests can represent ten seconds of prefill or ten minutes depending on prompt length. Conversely, counting running requests against the queue bound causes the opposite failure, rejecting incoming traffic even when the queue is empty.
