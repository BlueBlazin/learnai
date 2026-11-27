Think about what the queue bound protects. An admitted request waits behind everything already ahead of it. If the queue holds at most `max_queued_tokens` prompt tokens, what is the maximum wait time, and which requests still count as queued once a request starts running?

---

Store queued requests in admission order along with their prompt token counts. A Python `dict` keyed by `req.id` preserves insertion order and lets `start` remove any request in $O(1)$ time. Also track a running sum of queued tokens and a `set` of running request IDs. In `admit`, test both bounds with the new request included (`Q + 1` and `T + p`) *before* modifying the queue, and enqueue only if both pass. `start` moves the request from the queue to the running set and subtracts its tokens. `finish` removes it from the running set.

---

To compute `retry_after_s`, simulate draining queued requests from oldest to newest until the incoming request fits:

```python
if p > self.max_queued_tokens:
    return None
requests, tokens, drained = Q, T, 0
for waiting in queue_tokens_oldest_first:
    requests -= 1
    tokens -= waiting
    drained += waiting
    if requests + 1 <= max_queued_requests and tokens + p <= max_queued_tokens:
        break
return drained / self.drain_tokens_per_s
```

The loop is guaranteed to break with a fit because an empty queue accepts any prompt of at most `max_queued_tokens`.
