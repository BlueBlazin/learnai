## The idea

A retry bets that a failure was temporary. After every attempt, the client must decide three things: whether resending is safe, whether another attempt is still worth making, and how long to wait.

**What is safe to resend.** Status 429 or 503 means the server refused the request before executing it, so resending cannot duplicate work. Any other 4xx status means the request itself is invalid and will fail again. A dropped connection is ambiguous because the server may have executed the request before the reply was lost. Resending after a `ConnectionError` is safe only when executing twice has the same effect as executing once (idempotency). HTTP defines `GET`, `PUT`, and `DELETE` as idempotent and `POST` as non-idempotent, though APIs often make `POST` safe to retry using an idempotency key.

**Backoff with full jitter.** Waiting a fixed interval retries at a constant rate no matter how long an outage lasts. Doubling the delay ceiling on each retry (up to `cap_s`) lowers the retry rate geometrically. However, if a thousand clients fail in the same second, deterministic backoff sends all thousand back at `base_s`, then again at `3 * base_s`, replaying the overload in waves. Full jitter breaks this synchronization by multiplying the exponential ceiling for retry count $k$ by a uniform draw $u \in [0, 1)$:

$$
d = u \cdot \min(\text{cap\_s},\ \text{base\_s} \cdot 2^{k})
$$

This spreads a synchronized burst evenly across the interval. The AWS Architecture Blog post "Exponential Backoff And Jitter" (2015) introduced the term and compared no jitter, equal jitter (half the ceiling plus a random half), and decorrelated jitter in simulation. Backoff without jitter performed worst, and full jitter did the least total work.

**Honoring `Retry-After`.** The `Retry-After` header carries the server's own estimate of when capacity will be available (RFC 9110, section 10.2.3, either as seconds or an HTTP date). A client that ignores it and retries early is rejected again, while one that obeys it arrives when capacity opens up. Because HTTP header names are case-insensitive and HTTP/2 transmits them in lowercase, the lookup must normalize the header name.

**Attempts, deadline, and budget.** `max_attempts` caps work per call. The deadline stops retries that would start after the caller has already given up, so given start time `t0` and current time `t`, the check tests `(t - t0) + delay < deadline_s` before sleeping. The shared retry budget caps total retry volume across all calls. Without a budget, `max_attempts = 4` turns a full outage into four times normal traffic at the worst possible time. Draining a shared token bucket on failures and refilling it slowly on successes halts retries once failures dominate, keeping load near 1x.

## Reference solution (Python)

```python
async def call_with_retry(send, policy, uniforms, budget=None):
    loop = asyncio.get_running_loop()
    started = loop.time()
    retries = drawn = 0
    while True:
        response, error = None, None
        try:
            response = await send()
        except ConnectionError as exc:
            error = exc

        if error is not None:
            overloaded, retryable = True, policy.idempotent
        else:
            overloaded = retryable = response.status in (429, 503)
        if budget is not None:
            if overloaded:
                budget.failure()
            elif response.status < 400:
                budget.success()

        give_up = (
            not retryable
            or retries + 1 >= policy.max_attempts
            or (budget is not None and not budget.allows_retry())
        )
        if not give_up:
            delay = None if response is None else retry_after_s(response.headers)
            if delay is None:
                delay = uniforms[drawn] * min(policy.cap_s, policy.base_s * 2**retries)
                drawn += 1
            give_up = (loop.time() - started) + delay >= policy.deadline_s
        if give_up:
            if error is not None:
                raise error
            return response
        await asyncio.sleep(delay)
        retries += 1
```

`retry_after_s` returns `float(int(value))` for the first header whose lowercased name is `"retry-after"` and whose stripped value consists solely of ASCII digits, and `None` otherwise.

## Common mistakes

- **Retrying client errors.** Checking `status >= 400` resends 400 and 404 responses `max_attempts` times, adding load for requests that cannot succeed.
- **No jitter.** Using `min(cap, base * 2**k)` directly causes all 64 clients in the thundering-herd test to retry at the exact same instant on every round.
- **Equal jitter.** Using `ceiling / 2 + u * ceiling / 2` enforces a minimum wait and halves the spread. It is a valid strategy in general, but it does not match full jitter.
- **Applying the cap after jitter.** Computing `min(cap, u * base * 2**k)` clips most draws to `cap` once the exponential ceiling grows well past `cap`, re-synchronizing clients.
- **Off-by-one exponent.** The first retry uses exponent 0 (ceiling `base_s`), not 1 (`2 * base_s`). The exponent is the number of retries already made.
- **Ignoring `Retry-After` or checking only one casing.** Looking up `"Retry-After"` literally misses `"retry-after"` and other valid spellings.
- **Consuming a uniform for `Retry-After`.** Advancing the index into `uniforms` when the delay came from `Retry-After` shifts every subsequent backoff delay.
- **One attempt too many.** `max_attempts` counts total sends, including the first attempt, not retries.
- **Checking elapsed time instead of start time against the deadline.** Checking `elapsed < deadline` before sleeping allows a retry to start past the deadline (for example, at 7.5 s under a 4 s deadline).
- **Retrying non-idempotent connection errors.** If `policy.idempotent` is `False`, a `ConnectionError` must be raised immediately because the server may have executed the request.
- **Ignoring the budget or skipping `success()`.** Omitting `budget.success()` prevents the bucket from refilling after a burst of failures, so calls never retry again.
- **Using `random.random()`.** Drawing from a private RNG breaks determinism and violates the forbidden-API constraint.

## Complexity

$O(1)$ work per attempt, aside from scanning the response headers for `Retry-After`.

## In the wild

- **gRPC** implements the shared retry budget as "retry throttling" (gRFC A6). A per-server token count starts at `maxTokens`, subtracts 1 on failure, adds `tokenRatio` on success, and permits retries only while the count exceeds `maxTokens / 2`. This problem's test budget uses that exact scheme with `maxTokens = 10` and `tokenRatio = 0.5`.
- **Server-side admission control** (`admission-control`) supplies the other half of this protocol by rejecting excess load with 429 or 503 and a queue-derived `Retry-After` header.
- **Production clients** typically add a per-attempt timeout (so a hung request cannot outlive the call deadline), an upper bound on honored `Retry-After` values, extra jitter on top of `Retry-After` when many clients receive the same value, and a rule against transparently retrying streaming responses once tokens have already been delivered.
