Implement a token-bucket rate limiter for an LLM server. The bucket tracks tokens rather than requests, supports non-blocking checks with `try_acquire`, serves waiting `acquire` calls strictly in arrival order, and accepts `refund` calls for reserved tokens that a generation did not use.

![This problem is the Rate limit, a token bucket charged in LLM tokens.](assets/map.svg)

## Specification

```python
class TokenBucket:
    def __init__(self, rate_per_s: float, burst: int): ...
    def try_acquire(self, n: int) -> bool: ...
    async def acquire(self, n: int) -> None: ...
    def refund(self, n: int) -> None: ...
```

**Balance and refill.** The bucket holds a real-valued token balance between `0` and `burst`. It starts full at `burst` when constructed and refills continuously at `rate_per_s` tokens per second. If no calls change the balance between times $t_0$ and $t_1$, the new balance is

$$
\text{balance}(t_1) = \min\bigl(\text{burst},\ \text{balance}(t_0) + \text{rate\_per\_s} \cdot (t_1 - t_0)\bigr).
$$

Refill is continuous rather than quantized. After a quarter of a second, a quarter of `rate_per_s` tokens have accrued. The balance is never negative.

**Time.** Time comes from the running `asyncio` event loop's clock, `asyncio.get_running_loop().time()`, in seconds (provided as `now()` in the starter). The judge runs the loop on a simulated clock, so all waiting must go through `asyncio` primitives (`loop.call_later`, `asyncio.sleep`, `asyncio.wait_for`, futures, or events) and never through the real clock. The constructor and every method are called from inside the running event loop.

**Queueing and fairness.** Any `acquire` call that cannot be served immediately waits in a first-in, first-out (FIFO) queue. Only the call at the head of the queue can be served. It is served at the earliest instant when the balance reaches at least its requested `n`. At that instant, the balance decreases by `n` and the next queued call becomes the head. If the remaining balance already covers the new head, it is served at the same instant.

**Methods.** The argument `n` is an integer with `n >= 1` for `try_acquire` and `acquire`, and `n >= 0` for `refund`.

- `try_acquire(n)` never waits. If the queue is empty and the balance is at least `n`, it deducts `n` and returns `True`. Otherwise it returns `False` and leaves the bucket unchanged. In particular, it returns `False` whenever any `acquire` call is waiting, even if the current balance is at least `n`, because those tokens are reserved for the head of the queue.
- `await acquire(n)` returns `None` once `n` tokens have been deducted for this call. If the queue is empty and the balance is at least `n`, it deducts `n` and returns immediately. Otherwise it joins the back of the queue and returns at the instant it is served. A call that arrives while other calls are waiting always joins the back of the queue, regardless of how small `n` is.
- `refund(n)` returns `n` unused tokens to the bucket, updating the balance to `min(burst, balance + n)`. Any waiting calls at the head of the queue that can now be served are served at that same instant, in queue order.
- **Requests larger than `burst`.** Because the balance never exceeds `burst`, a request with `n > burst` can never be served. Both `try_acquire(n)` and `acquire(n)` must raise `ValueError` immediately at call time without modifying state or joining the queue.

**Events at the same instant.** A call made at time $t$ acts on the bucket's state at time $t$. First, all tokens accrued up to $t$ are added to the balance, and every waiting call that those tokens can serve is served. Then the new call takes effect. For example, if the head of the queue needs 8 tokens and the eighth token accrues at time $t$, a `refund` or `try_acquire` called at $t$ sees that waiting call already served and the balance at 0. Calls made at the same instant take effect in the order they are called.

Waiting `acquire` calls are never canceled in the tests.

## Example

Consider `TokenBucket(rate_per_s=64, burst=256)` created at time 0. A server reserves `prompt_tokens + max_tokens` up front for each request and refunds unused tokens when generation finishes.

| time | call | result | balance after |
|---|---|---|---|
| 0 | `acquire(192)` (A) | returns at 0 | 64 |
| 0 | `acquire(128)` (B) | waits (64 tokens short) | 64 |
| 0.5 | `acquire(16)` (C) | waits behind B (even with 96 tokens available) | 96 |
| 0.5 | `try_acquire(8)` | `False` (a call is waiting) | 96 |
| 1.0 | | B returns (balance reached 128) | 0 |
| 1.125 | `refund(32)` | 8 accrued + 32 refunded = 40, so C returns at 1.125 | 24 |
| 1.5 | `try_acquire(40)` | `True` (24 + 24 accrued = 48) | 8 |
| 10 | `try_acquire(256)` | `True` (balance capped at 256) | 0 |
| 10 | `try_acquire(1)` | `False` | 0 |

![The balance of the example up to 1.5 s. B is served when the balance reaches 128, and C waits behind it although 16 tokens are there.](assets/balance.svg)

Without the refund at 1.125 s, C would have returned at 1.25 s, when 16 tokens had accrued after B. Calling `acquire(300)` or `try_acquire(300)` raises `ValueError` at any time.

## Constraints

- `rate_per_s > 0` and `1 <= burst <= 100,000`, with at most a few hundred calls per test.
- In the tests, `rate_per_s` is a power of two between 0.5 and 1024, and every call occurs at a multiple of 1/64 s. Every balance and grant time is therefore exact in floating point.
- The tests record the virtual return time of each `acquire`, the boolean result of each `try_acquire`, and each `ValueError`, comparing against exact values (times within `1e-9` s).
- Every `acquire` with `n <= burst` must eventually return. A call still waiting when the simulation ends fails the test.
- Do not use the real clock (`time.sleep`, `time.time`, `time.monotonic`, `time.perf_counter`).

## Why this matters

LLM requests vary in cost by four orders of magnitude, so a requests-per-second limit offers little protection. Ten requests with 100,000-token prompts from one tenant can dwarf a thousand short chat turns from another. Production gateways therefore enforce limits in tokens per minute and charge an upfront estimate (`prompt_tokens + max_tokens`) before the model runs, because the true output length is unknown until generation stops.

Without refunds, every request that finishes early stays overcharged and the tenant's effective throughput drops far below the configured rate. Strict FIFO queueing is equally important. If small requests can grab tokens while a large request waits, a steady stream of small requests starves the large one indefinitely. In production, that bug surfaces as long-context requests timing out whenever the service is busy.
