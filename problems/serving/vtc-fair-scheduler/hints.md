Suppose the counters were simply "total weighted tokens ever served". Tenant A works
all night; tenant B first shows up in the morning with a thousand requests. Who gets
served, and for how long? What would B's counter have to be at the moment it shows up
for the two to share the engine from then on?

---

There are three places to touch in the starter, one per rule.

- `_pick`: choose by counter, not by position. If you take the minimum over the
  waiting *requests* with the key `counters[request.tenant]`, the first minimum in
  queue order is at once the right tenant (ties go to the tenant whose request comes
  first) and that tenant's earliest request.
- The admission loop: add `w_in * request.prompt_tokens` to the tenant's counter right
  after picking, inside the `while`, so that the next pick of the same step sees it.
  Remember the tenant of each live sequence, and the tenant admitted last.
- After the step: add `w_out` once for every id in `result.tokens`.

---

The lift, in `submit`, before appending the request:

```python
counter = self.counters.setdefault(tenant, 0.0)
queued = {r.tenant for r in self._waiting}
if tenant not in queued:
    if queued:
        counter = max(counter, min(self.counters[t] for t in queued))
    elif self._last is not None:              # tenant of the last admitted request
        counter = max(counter, self.counters[self._last])
    self.counters[tenant] = counter
```

"Queued" means "has a request in the waiting list": a tenant whose requests are all
running is not queued and is lifted when its next request arrives. The outer `max`
matters as much as the `min`: a lift may raise a counter, never lower it.
