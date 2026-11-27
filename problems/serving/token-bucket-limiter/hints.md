The balance changes continuously with time, but nothing needs to run continuously in the background. If you know the balance at one timestamp and how much time has elapsed, you can compute the current balance on demand. Think about the only moment the bucket actually has to wake up by itself, and which caller it needs to wake.

---

Track three pieces of state: the current balance, the timestamp when it was last updated, and a FIFO queue of waiting calls storing each call's `n` and an `asyncio.Future`. Write one helper that credits elapsed time with `min(burst, balance + elapsed * rate_per_s)` and call it at the start of every method. Write a second helper that drains the queue from the front. While the head's `n` is covered, deduct `n`, pop the head, and resolve its future. Checking only the head—and having `try_acquire` return `False` whenever the queue is non-empty—enforces the entire fairness rule.

---

The bucket needs at most one timer, scheduled for the instant the head of the queue will be covered: `(n - balance) / rate_per_s` seconds from now, where `n` is the head's requested token count.

```python
def _arm_timer(self):
    if self._timer is not None:
        self._timer.cancel()
    if self._waiters:
        missing = self._waiters[0][0] - self._balance
        self._timer = loop.call_later(missing / self.rate_per_s, self._serve_waiters)
```

Re-arm this timer whenever the head or the balance changes: when a call joins an empty queue, after serving waiters so the new head gets its deadline, and after a refund. In `refund`, serve the queue once *before* adding the refunded tokens (so a waiter already due takes its share before the `burst` cap applies) and once *after*.
