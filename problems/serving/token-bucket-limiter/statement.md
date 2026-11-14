Implement the rate limiter that sits in front of an LLM server: a token bucket whose
unit is the LLM token, with a non-blocking `try_acquire`, a waiting `acquire` that
serves callers strictly in arrival order, and a `refund` for tokens that were reserved
and not used. The arithmetic is two lines; the work is in when a waiting caller wakes
up, and in never letting a small request slip past a large one.

## Specification

```python
class TokenBucket:
    def __init__(self, rate_per_s: float, burst: int): ...
    def try_acquire(self, n: int) -> bool: ...
    async def acquire(self, n: int) -> None: ...
    def refund(self, n: int) -> None: ...
```

**The balance.** The bucket holds a real-valued balance of tokens, at most `burst`.
It is full (`burst`) when constructed and refills continuously at `rate_per_s` tokens
per second: if nothing else happens between times $t_0$ and $t_1$,

$$
\text{balance}(t_1) = \min\bigl(\text{burst},\ \text{balance}(t_0) + \text{rate\_per\_s} \cdot (t_1 - t_0)\bigr).
$$

Refill is not quantized: after a quarter of a second, a quarter of `rate_per_s` tokens
have accrued. The balance is never negative.

**Time** is the clock of the running asyncio event loop,
`asyncio.get_running_loop().time()`, in seconds (the starter's `now()`). The judge
runs the loop on a simulated clock, so waiting must go through asyncio
(`loop.call_later`, `asyncio.sleep`, `asyncio.wait_for`, futures, events) and never
through the real clock. The constructor and every method are called from inside the
running loop.

**The queue.** `acquire` calls that cannot be served at once wait in a queue, in the
order in which they were called. Only the call at the head of the queue can be served:
it is served at the first instant at which the balance is at least its `n`, the balance
drops by `n`, and the next call becomes the head (and is served at the same instant if
the remaining balance already covers it).

**The methods.** `n` is an int, `n >= 1` (for `refund`, `n >= 0`).

- `try_acquire(n)` never waits. If the queue is empty and the balance is at least `n`,
  it deducts `n` and returns `True`. Otherwise it returns `False` and changes nothing.
  In particular it returns `False` whenever a call is waiting, even if the balance
  covers `n`: those tokens belong to the head of the queue.
- `await acquire(n)` returns `None` once `n` tokens have been deducted for this call.
  If the queue is empty and the balance is at least `n`, that is immediately, at the
  time of the call. Otherwise the call joins the end of the queue and returns at the
  instant it is served. A call that arrives while others wait always joins the
  queue, however small it is.
- `refund(n)` gives back `n` tokens: the balance becomes
  `min(burst, balance + n)`. If waiting calls can now be served, they are served at
  this instant, in queue order.
- **`n > burst`** can never be served, because the balance never exceeds `burst`.
  `try_acquire(n)` and `acquire(n)` then raise `ValueError`, at the time of the call
  and without changing anything: `acquire` does not join the queue.

**Events at the same instant.** A call made at time $t$ acts on the bucket as it is at
time $t$: first all tokens accrued up to $t$ are credited, and every waiting call that
those tokens can serve is served; then the call takes effect. So if the head of the
queue needs 8 tokens and the eighth accrues exactly at $t$, a `refund` or
`try_acquire` made at $t$ finds that call already served and the balance at 0. Calls
made at the same instant take effect in the order in which they are made.

Waiting `acquire` calls are never canceled in the tests.

## Example

`TokenBucket(rate_per_s=64, burst=256)`, created at time 0. A server charges each
request `prompt_tokens + max_tokens` up front and refunds what the generation did not
use.

| time | call | result | balance after |
|---|---|---|---|
| 0 | `acquire(192)` (A) | returns at 0 | 64 |
| 0 | `acquire(128)` (B) | waits: 64 tokens are missing | 64 |
| 0.5 | `acquire(16)` (C) | waits behind B, although 96 tokens are there | 96 |
| 0.5 | `try_acquire(8)` | `False`: a call is waiting | 96 |
| 1.0 | | B returns: the balance reached 128 | 0 |
| 1.125 | `refund(32)` | 8 accrued + 32 = 40; C returns at 1.125 | 24 |
| 1.5 | `try_acquire(40)` | `True`: 24 + 24 accrued = 48 | 8 |
| 10 | `try_acquire(256)` | `True`: the balance was capped at 256 | 0 |
| 10 | `try_acquire(1)` | `False` | 0 |

Without the refund, C would have returned at 1.25, when 16 tokens had accrued after B.
`acquire(300)` or `try_acquire(300)` would raise `ValueError` at any time.

## Constraints

- `rate_per_s > 0` and `1 <= burst <= 100,000`. At most a few hundred calls per test.
- In the tests, `rate_per_s` is a power of two (from 0.5 to 1024) and every call
  happens at a multiple of 1/64 s, so every balance and every grant time is exactly
  representable in a float.
- The tests record the virtual time at which each `acquire` returns, the answer of
  each `try_acquire`, and each `ValueError`, and compare them with the exact values
  (times within `1e-9` s).
- Every `acquire` with `n <= burst` must return eventually; a call that is still
  waiting when the simulation ends fails the case.
- Do not use the real clock (`time.sleep`, `time.time`, `time.monotonic`,
  `time.perf_counter`).

## Why this matters

Requests to an LLM differ in cost by four orders of magnitude, so a limit in requests
per second protects nothing: one tenant's ten requests with 100,000-token prompts
outweigh another's thousand chat turns. Production limits are therefore in tokens per
minute, and the bucket is charged with an estimate, the prompt plus `max_tokens`,
before the request runs, because the true output length is unknown until the model
stops. Without the refund, every request that stops early is overcharged and the
tenant's real throughput falls well below the configured rate. The queue discipline
is the other half: if small requests may take tokens while a large one waits, a steady
stream of small requests starves the large one forever, which in production appears
as long-context requests that time out only when the system is busy.
