Implement `call_with_retry`, an asynchronous retry loop for an inference client. When a server signals overload or a connection drops, the client backs off with full jitter, honors `Retry-After` headers, respects a call deadline, and checks a shared retry budget.

![This problem is the Client, retrying when the server is overloaded.](assets/map.svg)

## Specification

```python
@dataclass(frozen=True)
class RetryPolicy:  # given in the starter
    max_attempts: int = 4  # sends in total, the first one included; >= 1
    base_s: float = 0.5  # backoff ceiling before the first retry
    cap_s: float = 8.0  # the ceiling never exceeds this
    deadline_s: float = 30.0  # total time budget of the call
    idempotent: bool = True  # may the request be resent when its outcome is unknown?


async def call_with_retry(send, policy: RetryPolicy, uniforms: list[float], budget=None): ...
```

`send` is an async callable that takes no arguments. Each `await send()` performs one attempt. An attempt takes time and either returns a **response** object with an integer `.status` and a `dict[str, str]` `.headers`, or raises `ConnectionError` (meaning the connection broke and whether the server executed the request is unknown).

Use the running event loop's clock (`asyncio.get_running_loop().time()` and `asyncio.sleep`) for all timing. Under the judge, this clock is virtual. Do not use real-time clocks or random number generators; all jitter comes from `uniforms`.

`call_with_retry` records its start time $t_0$ and sends the request. After each attempt completes, it executes the following steps in order.

### 1. Update the retry budget

`budget` is either `None` or a shared object with three methods: `success()`, `failure()`, and `allows_retry() -> bool`. If `budget` is not `None`, report the outcome before deciding whether to retry:

- Call `budget.failure()` once if the attempt returned status 429 or 503, or raised `ConnectionError` (whether retryable or not).
- Call `budget.success()` once if the attempt returned a status below 400.
- Call neither method for any other status (such as 400, 404, or 500).

The tests use a 10-token bucket where `failure()` removes 1 token (not below 0), `success()` adds 0.5 tokens (not above 10), and `allows_retry()` returns `True` while more than 5 tokens remain.

### 2. Check retry eligibility

The call may proceed to compute a delay only if all three conditions hold:

1. **The failure is retryable.** Status **429** and **503** are retryable because the server refused the request due to overload without executing it. A `ConnectionError` is retryable only if `policy.idempotent` is `True`. Nothing else is retryable (neither successes nor other error statuses such as 400, 404, or 500).
2. **Attempts remain.** Fewer than `policy.max_attempts` attempts have been made.
3. **The budget allows it.** `budget` is `None`, or `budget.allows_retry()` returns `True`.

If any condition fails, stop immediately without computing a delay or consuming an element of `uniforms`.

### 3. Compute the delay $d$

- **`Retry-After` header.** Check whether the response contains a header whose name matches `retry-after` (case-insensitive) and whose value, stripped of surrounding whitespace, consists only of ASCII digits. If so, set $d$ to that integer number of seconds, with no jitter and no cap. Treat any other value (such as an HTTP-date, a negative or fractional number, or an empty string) as absent. A delay taken from `Retry-After` does not consume an element of `uniforms`.
- **Backoff with full jitter.** Otherwise (and always after a `ConnectionError`), compute the delay from the number of retries $k$ already made by this call (regardless of how earlier delays were chosen) and the next unused value $u$ from `uniforms`:

$$
d = u \cdot \min\!\big(\text{cap\_s},\ \text{base\_s} \cdot 2^{k}\big)
$$

  The first backoff delay uses `uniforms[0]`, the second uses `uniforms[1]`, and so on. `uniforms` is guaranteed to be long enough.

### 4. Check the deadline and wait

Let $t$ be the current loop time. Retry only if $(t - t_0) + d < \text{deadline\_s}$. An attempt in flight is never interrupted; the deadline only prevents starting new attempts.

If the deadline check passes, wait $d$ seconds with `await asyncio.sleep(d)` and send the next attempt. Whenever the loop stops, return the last response (regardless of its status) or raise the last `ConnectionError`.

## Example

```python
policy = RetryPolicy(max_attempts=4, base_s=0.5, cap_s=8.0, deadline_s=30.0)
```

Suppose the call starts at $t = 1.0$ with `uniforms = [0.5, 0.75, 0.5]`, and the server responds immediately with 503, 503, and 200.

| time | event |
|---|---|
| 1.0 | attempt 1 returns 503. $k = 0$: $d = 0.5 \cdot \min(8, 0.5) = 0.25$ |
| 1.25 | attempt 2 returns 503. $k = 1$: $d = 0.75 \cdot \min(8, 1.0) = 0.75$ |
| 2.0 | attempt 3 returns 200, which is returned |

Additional cases starting at $t = 0$ with every uniform equal to 0.5:

- **Always 503.** Sends occur at 0, 0.25, 0.75, and 1.75. The fourth 503 is returned because `max_attempts` is 4.
- **404 response.** One send occurs, and the 404 is returned immediately.
- **`Retry-After` followed by backoff.** A 429 with `Retry-After: 2`, then a 503 without the header, then a 200. Sends occur at 0, 2.0, and 2.5. The second delay uses $k = 1$ (one retry already made) and `uniforms[0]` (the `Retry-After` delay consumed none), giving $0.5 \cdot \min(8, 1.0) = 0.5$.
- **Cap reached.** With `base_s = 1`, `cap_s = 4`, `max_attempts = 7`, and always 503, the ceilings are 1, 2, 4, 4, 4, and 4. Sends occur at 0, 0.5, 1.5, 3.5, 5.5, 7.5, and 9.5.
- **Deadline reached.** With `base_s = 1`, `deadline_s = 4`, `max_attempts = 10`, and always 503, sends occur at 0, 0.5, 1.5, and 3.5. The next delay would be 4.0, and $3.5 + 4.0 \ge 4$, so the 503 is returned at 3.5 without waiting.
- **Connection drop.** The connection breaks 0.5 s after the first send. With `idempotent=True`, the retry is sent at $0.5 + 0.25 = 0.75$. With `idempotent=False`, `ConnectionError` is raised at 0.5.

## Constraints

- `1 <= max_attempts <= 20`, `0 < base_s <= cap_s`, `deadline_s > 0`, and every element of `uniforms` is in $[0, 1)$.
- Send times are compared with an absolute tolerance of `1e-9` s. The number of sends and the final outcome must match exactly.
- **Thundering herd.** 64 clients call at $t = 0$ against a server that returns 503 until $t = 3$ (`base_s = 2`, `cap_s = 16`, `max_attempts = 6`, each attempt takes 1/64 s). No 1/8 s window after $t = 0$ may contain more than **24** sends. The reference sends at most 12 per window; without jitter, all 64 retries land in the same instant.
- **Retry storm.** 40 calls share one budget, and the first 20 meet a server that always returns 503. The 40 calls may send at most **50** requests in total. The reference sends 44; without the budget, it sends 101.
- **Simultaneous events.** A retry with delay 0 is sent at the same instant as the response that triggered it. Nothing else in the tests depends on the ordering of simultaneous events.
- Do not use `time.sleep`, `time.time`, `time.monotonic`, `time.perf_counter`, `random.random`, or `random.uniform`.

## Why this matters

An overloaded server can recover only when incoming load drops. Clients that retry immediately or in lockstep keep load high, because rejecting requests with 429 or 503 still consumes server capacity and turns a brief spike into a sustained outage.

Each part of this loop prevents a specific failure mode. Exponential backoff lowers the retry rate as overload persists. Full jitter spreads out clients that failed together so they do not return in synchronized waves. `Retry-After` lets the server, which knows its own queue depth (as in `admission-control`), dictate the wait. The attempt limit and deadline bound work spent on requests the caller has already abandoned. The shared retry budget caps fleet-wide retry traffic so a full outage does not multiply load by `max_attempts`. Finally, refusing to retry 4xx errors or non-idempotent connection drops avoids wasted requests and prevents duplicate generations or double charges.
