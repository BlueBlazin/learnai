The balance changes continuously, but nothing needs to run continuously. If you know
the balance at one moment and how much time has passed, you know the balance now. When
does the bucket actually have to wake up by itself, and for whom?

---

Keep three things: the balance, the time at which it was last brought up to date, and
a FIFO queue of waiting calls, each with its `n` and an `asyncio.Future`. Write one
helper that credits the elapsed time (`min(burst, balance + elapsed * rate)`) and call
it at the start of every method. Write a second helper that serves the queue: while
the head's `n` is covered, deduct it, pop it, and resolve its future. Only the head
is ever looked at; that is the whole fairness rule, together with `try_acquire`
answering `False` whenever the queue is not empty.

---

The bucket needs exactly one timer: for the moment the head of the queue will be
covered, `(head_n - balance) / rate_per_s` seconds from now.

```python
def _arm_timer(self):
    if self._timer is not None:
        self._timer.cancel()
    if self._waiters:
        missing = self._waiters[0][0] - self._balance
        self._timer = loop.call_later(missing / self.rate_per_s, self._serve_waiters)
```

Re-arm it whenever the head or the balance changes: after a call joins an empty
queue, after serving (the next head has its own deadline), and after a refund. In
`refund`, serve the queue once *before* adding the tokens (a waiter that is due right
now takes its share first) and once after.
