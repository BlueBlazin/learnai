# BUG: the refill is not capped at burst, so an idle bucket accumulates an unlimited balance and then lets an arbitrarily large burst through.
import asyncio
from collections import deque

EPSILON = 1e-9  # tokens; absorbs rounding in (elapsed * rate) when a timer fires


def now() -> float:
    """The current time in seconds, read from the running event loop's clock."""
    return asyncio.get_running_loop().time()


class TokenBucket:
    def __init__(self, rate_per_s: float, burst: int):
        self.rate_per_s = rate_per_s
        self.burst = burst
        self._balance = float(burst)  # the bucket starts full
        self._updated_at = None  # when _balance was last brought up to date
        self._waiters = deque()  # (n, future) of pending acquire() calls, oldest first
        self._timer = None  # the wake-up for the oldest waiter

    def try_acquire(self, n: int) -> bool:
        self._check(n)
        self._refill()
        # No barging: while anyone waits, the tokens that trickle in are theirs.
        if self._waiters or self._balance < n:
            return False
        self._balance -= n
        return True

    async def acquire(self, n: int) -> None:
        if self.try_acquire(n):
            return
        granted = asyncio.get_running_loop().create_future()
        self._waiters.append((n, granted))
        self._arm_timer()
        await granted

    def refund(self, n: int) -> None:
        # First bring the bucket up to date: a waiter whose tokens have accrued by now
        # takes them before the refund is added, so the cap applies to what is left.
        self._serve_waiters()
        self._balance = min(self.burst, self._balance + n)
        self._serve_waiters()

    def _check(self, n: int) -> None:
        if n > self.burst:
            # The balance never exceeds burst, so this request could never be served.
            raise ValueError(f"cannot acquire {n} tokens from a bucket of {self.burst}")

    def _refill(self) -> None:
        """Credit the tokens that accrued since the last update: lazily, on demand."""
        t = now()
        if self._updated_at is not None:
            accrued = (t - self._updated_at) * self.rate_per_s
            self._balance = self._balance + accrued
        self._updated_at = t

    def _serve_waiters(self) -> None:
        """Grant from the head of the queue while the balance allows; then re-arm."""
        self._refill()
        while self._waiters and self._balance >= self._waiters[0][0] - EPSILON:
            n, granted = self._waiters.popleft()
            self._balance = max(0.0, self._balance - n)
            granted.set_result(None)
        self._arm_timer()

    def _arm_timer(self) -> None:
        """One timer, for the moment the oldest waiter's tokens will have accrued."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._waiters:
            missing = self._waiters[0][0] - self._balance
            delay = max(0.0, missing / self.rate_per_s)
            self._timer = asyncio.get_running_loop().call_later(delay, self._serve_waiters)
