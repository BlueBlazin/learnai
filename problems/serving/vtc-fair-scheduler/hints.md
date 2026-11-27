Suppose each counter simply recorded the total weighted tokens ever served to a tenant. If tenant A runs continuously all night and tenant B arrives in the morning with a thousand requests, who gets served and for how long? What value must tenant B's counter start at so the two tenants share the server fairly from that moment on?

---

You need to update three parts of the starter code, one for each rule:

- **`_pick`:** Choose by tenant counter instead of queue position. Taking `min` over the waiting *requests* with key `self.counters[request.tenant]` simultaneously picks the tenant with the smallest counter (breaking ties by earliest arrival) and that tenant's earliest request.
- **Admission loop:** Add `self.w_in * request.prompt_tokens` to the tenant's counter immediately after `_pick`, inside the `while` loop, so subsequent picks in the same step see the updated value. Record each live sequence's tenant and update the most recently admitted tenant.
- **After `engine.step`:** Add `self.w_out` to the corresponding tenant's counter once for every sequence id in `result.tokens`.

---

Apply the counter lift inside `submit`, before appending the request to `self._waiting`:

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

A tenant is "queued" only if it has a request in `self._waiting`. A tenant whose requests are all running is not queued and is lifted when its next request arrives. The outer `max` is just as important as the inner `min`, because a lift may only raise a counter and never lower it.
