A waiting request's rank is not a fixed number: two requests can swap places while
both just sit in the queue. What does that mean for a data structure that stores a
sort key when the request is inserted, and at which moment is the rank really needed?

---

Compute everything at the moment a slot is handed out. `effective_priority` is the
base priority minus the number of *whole* aging periods waited since `arrival_s`
(floor division, so 39.75 ms at 40 ms per level is still zero levels and 40 ms is
one). `pick_next` then takes the minimum of the waiting list under the key
`(effective priority now, arrival time)`. If you scan the list in order and keep the
first minimum, requests that tie on both parts of the key come out in queue order
without a third component.

---

```python
def effective_priority(request, now_ms, aging_ms):
    return request.priority - int((now_ms - request.arrival_s) // aging_ms)
```

`min(waiting, key=...)` returns the first element with the smallest key, which is
exactly rule 3, and `waiting.remove(best)` takes it out. Do not put the request id in
the key, do not replace the floor by a fraction (a request that is 90% of the way to
its next level has not reached it), and do not cache the effective priority anywhere:
the linear scan per admission is cheap next to a forward pass.
