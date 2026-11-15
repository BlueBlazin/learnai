Two questions settle `pick`. How do you draw a second replica that is guaranteed to
differ from the first, with one uniform number and no retry loop? And when the two
are equally loaded, which answer treats them alike: one based on their ids, or one
based on the order in which chance presented them?

---

For the second draw, pick one of the `n - 1` positions that remain after removing the
first, then map it back to an index in the full list: positions at or after the first
choice move up by one. For the tie, keep the first choice unless the second is
*strictly* less loaded. For the balancer, keep one dict of estimates. Ask what a
report already contains (everything outstanding right now, including your own earlier
requests) and what it cannot contain (the requests you route after it).

---

```python
i = int(u1 * n)
j = int(u2 * (n - 1))
if j >= i:
    j += 1
first, second = replicas[i], replicas[j]
return second if load[second] < load[first] else first
```

In `Balancer`: `report` replaces every estimate with the reported value; `route`
calls `pick` on the estimates and then does `estimate[chosen] += tokens`.
