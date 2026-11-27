Two questions settle `pick`. First, how can you draw a second replica that is guaranteed to differ from the first using one uniform number and no retry loop? Second, when two replicas have equal load, which tie-breaking rule treats them symmetrically—comparing their IDs, or using the random order in which they were drawn?

---

For the second draw, sample one of the `n - 1` remaining positions and map it back into the full list by shifting any index at or above the first choice up by one. For ties, keep the first choice unless the second is *strictly* less loaded. In `Balancer`, maintain a dictionary of load estimates. Think about what a fresh report already includes (every currently outstanding token, including your earlier requests) versus what it cannot include (requests you route after the report arrives).

---

```python
i = int(u1 * n)
j = int(u2 * (n - 1))
if j >= i:
    j += 1
first, second = replicas[i], replicas[j]
return second if load[second] < load[first] else first
```

In `Balancer`, `report` replaces every estimate with the reported value, and `route` calls `pick` on the current estimates before executing `estimate[chosen] += tokens`.
