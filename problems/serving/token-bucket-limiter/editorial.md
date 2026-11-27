## The idea

**Refill lazily.** Because the balance is a deterministic function of elapsed time, you do not need a background loop to update it. Store the balance alongside the timestamp `_updated_at` when it was last computed, and bring it up to date whenever any method runs:

$$
\text{balance} \leftarrow \min\bigl(\text{burst},\ \text{balance} + (\text{now} - \text{updated\_at}) \cdot \text{rate\_per\_s}\bigr).
$$

No background task ticks, an idle bucket consumes zero CPU, and the balance is exact at every instant instead of stepping in coarse intervals.

**One queue, one timer.** Fairness comes from a single rule. Only the head of the queue may be served. Because the head's token count `n` is known, the exact time it will be ready is `(n - balance) / rate_per_s` seconds from now. The bucket maintains at most one timer for that deadline and re-arms it whenever one of three events occurs: a call joins an empty queue, the head is served and exposes a new head, or a `refund` raises the balance.

**Why only the head can be served.** If any queued call that fits could take tokens, small requests would drain tokens as fast as they accrue and a large request would never see the balance reach its `n`. Reserving incoming tokens for the head bounds every wait. A call waits at most the sum of `n` across the calls ahead of it, plus its own `n`, divided by `rate_per_s`. `try_acquire` must obey the same rule, or it becomes a back door that lets small requests cut in front of queued callers.

**Settle, then act.** Every public method first refills the balance and serves any waiting calls that are now covered. This makes behavior independent of the order in which the event loop dispatches a timer callback and a user call scheduled at the same instant. It is especially important in `refund`. If the head's tokens have just accrued and `refund` adds tokens before serving the head, capping at `burst` can discard tokens that the head should already have taken.

**Why `n > burst` must raise immediately.** The balance can never exceed `burst`, so a request with `n > burst` would wait forever. In a FIFO queue, every call behind it would stall forever as well. Raising `ValueError` at call time prevents that deadlock.

## Reference solution (Python)

```python
class TokenBucket:
    def __init__(self, rate_per_s, burst):
        self.rate_per_s = rate_per_s
        self.burst = burst
        self._balance = float(burst)
        self._updated_at = None
        self._waiters = deque()      # (n, future), oldest first
        self._timer = None

    def try_acquire(self, n):
        self._check(n)
        self._refill()
        if self._waiters or self._balance < n:
            return False
        self._balance -= n
        return True

    async def acquire(self, n):
        if self.try_acquire(n):
            return
        granted = asyncio.get_running_loop().create_future()
        self._waiters.append((n, granted))
        self._arm_timer()
        await granted

    def refund(self, n):
        self._serve_waiters()                 # settle first
        self._balance = min(self.burst, self._balance + n)
        self._serve_waiters()

    def _check(self, n):
        if n > self.burst:
            raise ValueError(f"cannot acquire {n} tokens from a bucket of {self.burst}")

    def _refill(self):
        t = now()
        if self._updated_at is not None:
            accrued = (t - self._updated_at) * self.rate_per_s
            self._balance = min(self.burst, self._balance + accrued)
        self._updated_at = t

    def _serve_waiters(self):
        self._refill()
        while self._waiters and self._balance >= self._waiters[0][0] - EPSILON:
            n, granted = self._waiters.popleft()
            self._balance = max(0.0, self._balance - n)
            granted.set_result(None)
        self._arm_timer()

    def _arm_timer(self):
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._waiters:
            missing = self._waiters[0][0] - self._balance
            delay = max(0.0, missing / self.rate_per_s)
            self._timer = asyncio.get_running_loop().call_later(delay, self._serve_waiters)
```

`EPSILON` (`1e-9` tokens) protects against floating-point rounding when `rate_per_s` is not a binary fraction. When the timer fires, `elapsed * rate_per_s` can fall one ULP short of `missing`. Without that small tolerance, the timer would re-arm with a delay too tiny to advance the clock. The test suite uses exact binary fractions, so `EPSILON` does not affect the test results.

## Common mistakes

- **Refilling on a periodic tick.** A background task that adds tokens every second or every 100 ms releases capacity in chunks. Requests wait until tick boundaries, traffic leaves in synchronized bursts, and an idle bucket still burns CPU.
- **Barging.** Checking only `balance >= n` in `try_acquire` or in the fast path of `acquire` lets small requests take tokens that a waiting large request is accumulating. The fast path must also require an empty queue.
- **Serving whoever fits.** Scanning the queue for any call that fits looks like higher utilization, but it causes starvation. Under a steady stream of small requests, a large request never returns.
- **Missing the `burst` cap.** Omitting `min(burst, ...)` during refill lets an idle bucket accumulate an unbounded burst overnight. The same cap is required in `refund`.
- **Queueing impossible requests.** Enqueueing a call with `n > burst` blocks that call and every caller behind it forever. Validate `n <= burst` immediately on entry.
- **Refunding without waking waiters.** Adding refunded tokens without serving the queue leaves waiting callers asleep until their old timer fires, wasting the returned capacity.
- **Forgetting to re-arm the timer.** After serving the head, the next waiter needs a timer for its own deadline. Without re-arming, the queue only advances when an unrelated call arrives, causing requests to hang during quiet periods.
- **Refunding before settling.** At the exact instant the head's tokens finish accruing, adding a refund first and capping at `burst` can clip tokens that the head was entitled to take.
- **Starting empty.** `burst` is the cold-start allowance. Initializing the balance to `0` unnecessarily throttles the first requests after every deploy.

## Complexity

Every operation runs in $O(1)$ amortized time. Each call performs one refill, and each waiting call is enqueued and dequeued once. Memory is $O(W)$ for $W$ waiting calls, with at most one active timer regardless of queue length.

## In the wild

- **API providers and gateways** publish token-per-minute limits alongside request-per-minute limits and reserve `max_tokens` when a request enters. Gateways such as LiteLLM and Envoy AI Gateway follow this exact pattern, reserving an upfront estimate and reconciling it from the `usage` field reported by the model. That reconciliation is this problem's `refund`.
- **GCRA (Generic Cell Rate Algorithm)** is the virtual-scheduling form of a leaky bucket used in Redis-backed rate limiters. It stores a single theoretical arrival timestamp instead of a balance and last-update time. It applies the same lazy-update principle and fits in one atomic Redis script for multi-replica gateways.
- **Inside the serving engine**, the same fairness question reappears at the scheduler level. vLLM defaults to first-come, first-served scheduling, while the Virtual Token Counter scheduler (Sheng et al., OSDI 2024) prioritizes the tenant with the fewest served tokens to maintain fairness without rejecting traffic.
- **Rejecting versus waiting.** `try_acquire` maps to returning HTTP 429 with a `Retry-After` header for interactive traffic, while `acquire` maps to queueing for batch workloads. Production limiters also remove a waiting caller when its client disconnects, which this problem omits.
