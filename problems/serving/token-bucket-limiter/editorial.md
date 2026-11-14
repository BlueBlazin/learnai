## The idea

**Refill lazily.** The balance is a function of time, so it does not have to be
stored at every moment. Store the balance together with the time at which it was
valid, and bring it up to date whenever someone looks:

$$
\text{balance} \leftarrow \min\bigl(\text{burst},\ \text{balance} + (\text{now} - \text{updated\_at}) \cdot \text{rate}\bigr).
$$

No background task ticks, an idle bucket costs nothing, and the balance is exact at
every instant instead of a staircase.

**One queue, one timer.** Fairness is a single rule: only the head of the queue may be
served. A waiting call must therefore be woken at one predictable moment, when the
balance reaches the head's `n`, which is `(n - balance) / rate` seconds away. The
bucket keeps one timer for that moment. Three events change the moment and re-arm the
timer: a call joins an empty queue, the head is served (the next head has a different
`n`), and a refund raises the balance.

**Why the head only.** If any waiting call that fits could be served, small requests
would consume the tokens as fast as they accrue, and a large request would never see
the balance reach its `n`. Reserving the trickle for the head bounds every wait: a
call waits at most the sum of the `n` of the calls ahead of it, plus its own, divided
by the rate. `try_acquire` has to respect the same rule, or it becomes the side door
through which small requests barge.

**Settle, then act.** Every public method first brings the bucket up to date: credit
the refill and serve whoever that covers. This makes the outcome independent of the
order in which the event loop happens to run a timer and a call that fall on the same
instant. It matters in `refund`: if the head's tokens have just accrued and the refund
is added first, the cap at `burst` can destroy tokens that the head was about to take.

**Why `n > burst` must raise.** The balance never exceeds `burst`, so such a request
would wait forever, and in a FIFO queue everything behind it would wait forever too.
The only correct moment to report it is the call.

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

`EPSILON` (`1e-9` tokens) is for rates that are not binary fractions: when the timer
fires, `elapsed * rate` can come out one rounding error short of `missing`, and
without the slack the timer would be re-armed for a delay too small to advance the
clock. The tests use exactly representable values, where it plays no role.

## Common mistakes

- **Refilling on a tick.** A background task that adds `rate` tokens every second
  (or every 100 ms) releases tokens in lumps: requests are granted at tick boundaries,
  up to a tick late, and traffic leaves in synchronized bursts. It also runs forever
  on an idle bucket.
- **Barging.** Checking only `balance >= n` in `try_acquire` or in the fast path of
  `acquire` lets a small request take tokens that a waiting large request has been
  saving for. The queue must be empty for the fast path.
- **Serving whoever fits.** Scanning the queue for any call the balance covers looks
  like better utilization and is starvation: under a steady stream of small requests,
  the large one never returns.
- **No cap, or a cap in one place only.** Without `min(burst, ...)` in the refill, a
  bucket that was idle overnight admits an unbounded burst in the morning. The same
  cap applies to `refund`.
- **Waiting for the impossible.** Queueing a request with `n > burst` blocks it and
  everyone behind it forever. Reject it at the call.
- **A refund that wakes nobody.** Adding the tokens without serving the queue leaves
  the waiters asleep until a wake-up computed from the old balance; the refunded
  capacity sits unused in the meantime.
- **Forgetting to re-arm.** After the head is served, the next waiter needs its own
  timer. Without it the queue advances only when an unrelated call arrives, which in
  production is a request that hangs until the next request comes in.
- **Refunding before settling.** At the instant the head's tokens have accrued, adding
  a refund first and capping at `burst` discards tokens the head should already have
  taken.
- **Starting empty.** `burst` is the allowance for a cold start; an empty bucket
  throttles the first requests after every deploy.

## Complexity

Every operation is $O(1)$ amortized: one refill, and each waiting call is enqueued and
dequeued once. Memory is $O(\text{waiting calls})$, and there is at most one pending
timer whatever the queue length.

## In the wild

- **API providers** publish limits in tokens per minute next to requests per minute,
  and count the request's `max_tokens` against the limit when the request is admitted.
  Gateways such as LiteLLM and Envoy AI Gateway implement token-based limits the same
  way: reserve an estimate, then correct it from the `usage` the model reports. The
  correction is this problem's `refund`.
- **GCRA**, the "virtual scheduling" form of the leaky bucket used by Redis-based
  limiters, stores a single timestamp (the theoretical arrival time) instead of a
  balance and a time. It is the same lazy idea and it fits in one atomic Redis
  script, which is what a limiter shared by several gateway replicas needs.
- **Inside the engine** the same fairness question appears one level down. vLLM's
  scheduler is first come, first served by default, and the Virtual Token Counter
  scheduler (Sheng et al., OSDI 2024) replaces "who arrived first" with "who has
  been served the fewest tokens" to keep tenants fair without rejecting anyone.
- **Rejecting versus waiting.** `try_acquire` is the path that answers HTTP 429 with a
  `Retry-After` header; `acquire` is the path for batch jobs that prefer to queue.
  Real limiters also remove a waiting call when its client disconnects, which this
  problem leaves out.
