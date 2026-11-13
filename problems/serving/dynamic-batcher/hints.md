All three handlers end with the same question: "given what I know now, should a batch
leave?" What is the smallest state from which that question can be answered at any
moment? (You need to know two things about the queue and one thing about the model.)

---

Keep a FIFO queue of `(request_id, arrival_ms)` pairs and a `busy` flag. Each handler
first records its event (`on_arrival` appends to the queue, `on_batch_done` clears
`busy`, `on_timer` records nothing) and then calls one shared method that applies the
dispatch rule from the statement: return `None` if busy or empty; otherwise dispatch if
the queue length has reached `max_batch_size` or `now_ms` has reached the oldest
request's arrival time plus `max_wait_ms`. When that method dispatches, it removes the
oldest requests from the queue and sets `busy`. Because every handler goes through the
same method, the edge cases in the statement need no code of their own.

---

Do not store a deadline or a "timer armed" flag: derive the wake-up from the queue
every time it is asked for, so it can never be stale.

```python
def next_timer_ms(self):
    if self.busy or not self.queue:
        return None
    return self.queue[0][1] + self.max_wait_ms   # the oldest request's deadline
```

After a flush that leaves requests behind, `self.queue[0]` is the oldest of the
leftovers, and this returns its deadline with no extra bookkeeping. The comparison in
the dispatch rule is `now_ms >= deadline`, not `>`: at the deadline itself the request
has waited exactly `max_wait_ms`, and that is when it leaves.
