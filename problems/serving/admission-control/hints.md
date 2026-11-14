What does the bound protect? A request that is admitted has to wait for everything in
front of it. If the queue can hold at most `max_queued_tokens` prompt tokens, what is
the longest that wait can be? And which requests are "in front" once a request has
started running?

---

Keep the queued requests in arrival order with their prompt tokens (a dict keyed by
`id` keeps insertion order and lets `start` remove any of them), a running total of
queued tokens, and the set of running ids. In `admit`, test both bounds with the new
request included, `Q + 1` and `T + p`, *before* touching the queue, and enqueue only
on success. `start` moves the request from the queue to the running set and subtracts
its tokens; `finish` only removes it from the running set.

---

For `retry_after_s`, replay the queue draining from its head:

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

The loop always ends with a fit, because an empty queue accepts any prompt of at most
`max_queued_tokens`.
