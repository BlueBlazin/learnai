All three handlers end with the same question: given the current state and `now_ms`, should a batch be dispatched right now? Think about the smallest state needed to answer that question at any moment. You need two pieces of information for each queued request and one flag for the model.

---

Keep a FIFO queue of `(request_id, arrival_ms)` pairs and a `busy` flag. Each handler records its event (`on_arrival` appends to the queue, `on_batch_done` clears `busy`, `on_timer` updates nothing) and then calls one shared dispatch helper.

In that helper, return `None` if the model is busy or the queue is empty. Otherwise, dispatch if the queue length has reached `max_batch_size` or `now_ms` has reached the oldest request's arrival time plus `max_wait_ms`. When dispatching, pop the oldest requests from the queue and set `busy = True`. Routing every handler through this single helper handles all the edge cases automatically.

---

Avoid storing a separate deadline or a "timer armed" flag. Instead, compute the wake-up time from the front of the queue whenever `next_timer_ms()` is called so it can never go stale:

```python
def next_timer_ms(self):
    if self.busy or not self.queue:
        return None
    return self.queue[0][1] + self.max_wait_ms   # the oldest request's deadline
```

After a dispatch that leaves requests behind, `self.queue[0]` is automatically the oldest remaining request, so this returns its deadline with no extra bookkeeping. In the dispatch check, use `now_ms >= deadline` rather than `>`. At the exact deadline, the oldest request has waited `max_wait_ms` and should leave immediately.
