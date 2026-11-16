Everything the gateway does happens at three kinds of moment: a request arrives, a
request completes, the policy is evaluated. Write one `dispatch` routine and call it
at the end of each. Inside it there are exactly two choices, which request and
which replica, and stages 2 to 4 each replace one of them. What is the condition
under which `dispatch` has nothing to do?

---

`dispatch`: while the queue is not empty, collect the ready replicas with
`queue_len < cfg.replica_slots`; stop if there are none; choose the request (the
oldest, or with `cfg.fair` the earliest request of the queued tenant with the
smallest counter); choose the replica (least `outstanding_tokens`, lowest id; or
the router of `prefix-aware-router` with `outstanding_tokens` in place of
`queue_len` and the list of free replicas in place of all replicas); call
`replica.submit(request)` and add a done callback to the future it returns.
Admission is decide first, change state second: compute the tenant's balance as
`min(burst, balance + rate * (now - since))`, check it, check the queue bounds
against the requests in *your* queue only, and only then store the reduced balance.

---

The completion callback and the policy, which are where the order matters:

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

The counter lift happens in `submit`, after the request is admitted and before it
is appended: only if its tenant has no request in the queue, raise the tenant's
counter to the smallest counter among queued tenants, or, with an empty queue, to
the counter of the tenant whose request left the queue last.
