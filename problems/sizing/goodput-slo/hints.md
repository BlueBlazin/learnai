Start by defining when a single request is good. A request fails the check if it did not complete or if it exceeds any metric in `slo`. Be careful with the three passing edge cases: a latency exactly equal to its limit, a `"tpot"` of `None`, and any metric not present in `slo`.

---

Write a helper `is_good(request, slo)` and reuse it in all three functions. Return `False` immediately if `not request["ok"]`, then check each key present in `slo`, skipping `None` values and returning `False` as soon as a measured latency exceeds its limit. Divide the good count by `len(requests)` (including failures) for `slo_attainment`, and by `duration_s` for `request_goodput`.

---

For `goodput`, sort the rates from lowest to highest and stop at the first failure:

```python
best = None
for rate in sorted(runs):
    if slo_attainment(runs[rate], slo) < target:
        break
    best = rate
return best
```

Use `break`, not `continue`. Once a rate fails, no higher rate can count even if its run happened to pass.
