The gateway reacts to three events: a request arrives (`submit`), a request completes (done callback), and the autoscaler runs (`evaluate`). Write one `_dispatch` method and call it at the end of all three. Inside `_dispatch`, there are two decisions: which queued request to pick, and which replica to send it to. Stages 2 through 4 each customize one of those decisions. Start by identifying the exact condition under which `_dispatch` should stop looping.

---

- **`_dispatch`:** Loop while the gateway queue is non-empty. Find all replicas with `r.state == "ready"` and `r.queue_len < cfg.replica_slots`; return if none exist. Pick the next request (the oldest, or under `cfg.fair` the earliest request of the queued tenant with the smallest counter). Pick the target replica (smallest `(outstanding_tokens, id)`, or under `cfg.prefix_routing` the `prefix-aware-router` rule using `outstanding_tokens` over the free replicas). Call `replica.submit(request)` and attach a done callback to the returned future.
- **Admission:** Decide before mutating state. Compute `min(burst, balance + rate * (now - since))`, check the rate limit, check `max_queued_requests` and `max_queued_tokens` against your gateway queue only, and only then update the bucket and enqueue.

---

Here are the completion callback and autoscaling evaluation, where operation order matters:

```python
def _finished(self, request, done):
    outcome = done.result()
    if self.cfg.fair and outcome.status == "finished":
        self.counters[request.tenant] += self.cfg.w_out * request.output_tokens
    self._clients.pop(request.id).set_result(outcome)
    self._dispatch()


def evaluate(self):
    m = self.cluster.metrics()
    current = m.ready + m.starting
    load = len(self._queue) + m.queue_depth + m.running
    raw = (
        current
        if abs(load - current * cfg.target) <= cfg.tolerance * current * cfg.target
        else -(-load // cfg.target)
    )
    raw = max(cfg.min_replicas, min(cfg.max_replicas, raw))
    self._history = [(t, n) for t, n in self._history if m.now - t < cfg.window_s] + [(m.now, raw)]
    n = raw if raw >= current else min(current, max(n for _, n in self._history))
    self.cluster.scale_to(n)  # drains; never cluster.stop
    self._dispatch()
    return n
```

Perform the VTC counter lift in `submit` right after admission and before appending to the queue. If the tenant has no request currently in the queue, raise its counter to the minimum counter among queued tenants, or (when the queue is empty) to the counter of the tenant whose request most recently left the queue.
