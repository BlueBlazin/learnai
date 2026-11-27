A waiting request's effective priority changes as virtual time advances, so two requests can swap places while both sit in the queue. Think about what happens if you cache a sort key at insertion time, and when the scheduler actually needs to know which request is best.

---

Compute effective priorities at the moment a slot is assigned. `effective_priority` subtracts the number of whole `aging_ms` periods waited since `request.arrival_s` using floor division (so 39.75 ms with `aging_ms = 40` is 0 levels, while 40 ms is 1 level). In `pick_next`, take the minimum over `waiting` using the key `(effective_priority(r, now_ms, aging_ms), r.arrival_s)`. Scanning `waiting` in order and keeping the first minimum automatically breaks remaining ties by queue position.

---

```python
def effective_priority(request, now_ms, aging_ms):
    return request.priority - int((now_ms - request.arrival_s) // aging_ms)
```

Python's `min(waiting, key=...)` returns the first element with the smallest key, which satisfies rule 3, and `waiting.remove(best)` removes it in place. Avoid putting `request.id` in the key, avoid replacing floor division with floating-point division, and do not cache effective priorities between admissions.
