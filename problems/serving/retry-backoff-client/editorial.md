## The idea

A retry is a bet that the failure was temporary. The loop has to decide three things
after each attempt: whether the bet is allowed at all, whether it is still worth
making, and when to make it.

**Whether: what is safe to resend.** A 429 or a 503 is the server saying "not now":
the request was refused before any work was done, so resending it cannot duplicate
anything. A 4xx other than 429 is the server saying "not ever": the request is wrong
and will be wrong again. A broken connection says nothing: the request may have been
executed and only the reply lost. Resending is then safe exactly when executing
twice is the same as executing once, which is what idempotent means. HTTP defines
GET, PUT, and DELETE as idempotent and POST as not; APIs make POST safe to retry
with an idempotency key that lets the server recognize the duplicate.

**When: backoff with full jitter.** Waiting a fixed time retries at a constant rate
however long the outage lasts. Doubling the wait each time (capped, so it does not
grow without bound) reduces the rate geometrically. But if a thousand clients were
rejected in the same second, a deterministic schedule sends all thousand back at
`base`, then again at `3 * base`, and so on: the overload is replayed in bursts.
Jitter breaks the synchronization. "Full jitter" draws the whole delay uniformly
between 0 and the exponential ceiling,

$$
d = u \cdot \min(\text{cap}, \text{base} \cdot 2^{k}),
$$

which spreads the retries of a burst evenly over the interval. The AWS Architecture
Blog post "Exponential Backoff And Jitter" (2015) introduced the name and compared
it in simulation with no jitter, "equal jitter" (half the ceiling plus a random
half), and "decorrelated jitter": backoff without jitter was the clear loser, and
full jitter did the least total work.

**When, if the server says so.** `Retry-After` carries the server's own estimate of
when capacity will be free (RFC 9110, section 10.2.3: either a number of seconds or
an HTTP date). A client that retries earlier is rejected again; one that honors it
arrives when there is room. Header names are case-insensitive, and HTTP/2 transmits
them in lower case, so the lookup must not depend on the spelling.

**Whether it is still worth it: attempts, deadline, budget.** The attempt limit
bounds the work per call. The deadline bounds the time: a caller that has given up
after 30 seconds gains nothing from a retry at second 31, so the check is on the
moment the retry would *start*. The budget bounds what neither can, the total across
calls. With `max_attempts = 4` and a server that is down, every client sends four
times as many requests as usual, at the worst moment. A shared token bucket that
failures drain and successes slowly refill stops that: once failures dominate,
retries stop altogether, and each call costs one request.

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

`retry_after_s` returns `float(int(value))` for the first header whose lower-cased
name is `retry-after` and whose stripped value is all ASCII digits, and `None`
otherwise.

## Common mistakes

- **Retrying client errors.** `status >= 400` as the test for "failed" resends a 400
  or a 404 `max_attempts` times. The request is the problem; repeating it only adds
  load.
- **No jitter.** `min(cap, base * 2**k)` alone. In the herd case all 64 clients
  return in the same instant, at every retry.
- **Equal jitter.** `ceiling / 2 + u * ceiling / 2` keeps a minimum wait and halves
  the spread. It is a legitimate scheme, and it is not full jitter.
- **The cap applied after the jitter.** `min(cap, u * base * 2**k)`: once the
  ceiling is several times the cap, most draws exceed the cap and are clipped to
  exactly `cap`, so the clients synchronize again.
- **The exponent off by one.** The first retry waits up to `base`, not `2 * base`:
  the exponent is the number of retries already made.
- **Ignoring `Retry-After`,** or looking it up with one fixed spelling.
- **Spending a uniform on a `Retry-After` delay.** Harmless with a real generator and
  wrong here: the supplied uniforms are the specification of the randomness.
- **One attempt too many.** `max_attempts` counts sends, not retries.
- **No deadline, or the deadline checked against the time elapsed.** `elapsed <
  deadline` before sleeping lets a retry start long after the deadline: in the
  example the retry at 7.5 s under a 4 s deadline.
- **Retrying a non-idempotent request after a connection error.** The server may
  have run it.
- **Ignoring the budget,** or never reporting successes to it, after which no call
  retries again.
- **A private random generator.** Jitter from `random.random()` cannot be tested or
  reproduced; taking the randomness as an input is what makes this function
  deterministic.

## Complexity

$O(1)$ work per attempt, apart from scanning the response headers.

## In the wild

- **gRPC** implements the budget as "retry throttling" (gRFC A6, client retries): a
  per-server token count starts at `maxTokens`, each failed call subtracts 1, each
  success adds `tokenRatio`, and retries are attempted only while the count is above
  half of `maxTokens`. This problem's test budget is that scheme with
  `maxTokens = 10` and `tokenRatio = 0.5`.
- **Servers** produce the other half of this protocol: an admission controller that
  rejects with 429 or 503 and a `Retry-After` computed from its queue
  (`admission-control`), so that well-behaved clients arrive when there is room.
- **What a production client adds**: a per-attempt timeout (so that a hung attempt
  cannot outlive the deadline), an upper bound on how long a `Retry-After` it is
  willing to honor, jitter on top of `Retry-After` when many clients receive the same
  value, and for streaming responses the rule that a request which has already
  delivered tokens is not transparently retried.
