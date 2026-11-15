Before any arithmetic, write down the predicate: when is one request good? There are
three ways to be not good (it failed, it broke one objective, it broke another), and
three ways to look bad without being bad (a latency exactly at the limit, a `None`
tpot, a metric the SLO does not mention). Everything else in the problem is counting
with that predicate.

---

Write a helper `is_good(request, slo)` and use it in all three functions: return
`False` at once if the request failed; then loop over the keys that are present in
`slo`, skip a value that is `None`, and return `False` on the first value greater than
its limit. Attainment divides the count by `len(requests)`, failures included;
`request_goodput` divides the same count by the duration.

---

For the sweep, sort the rates and stop at the first one that fails:

```python
best = None
for rate in sorted(runs):
    if slo_attainment(runs[rate], slo) < target:
        break
    best = rate
return best
```

`break`, not `continue`: a rate above a failing one never counts, however well its
run went.
