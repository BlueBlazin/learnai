Count in whole cycles instead of in steps. One request occupies a slot for a fixed
number of engine steps; what does one slot add to the total time of those steps, and
what is added regardless of how many slots there are? Once $S(b)$ is a linear function
of $b$, "the largest batch that meets the objective" is one integer division.

---

Keep every comparison in integers by never dividing early. `S(b) / o <= slo` is
`fixed + b * per_slot <= o * slo` with `fixed = o * base_us` and
`per_slot = prefill_us_per_token * p + decode_us_per_seq * (o - 1)`, so
`batch = min(max_seqs, (o * slo - fixed) // per_slot)`, and `None` if that is below 1.
The capacity is the fraction `batch * 10**6 / S(batch)`; carry it as a numerator and a
denominator (or a `fractions.Fraction`) until the replica count is done, and convert
to float only for the values that are reported as floats. For the curve, the capacity
is also a fraction: `(r0 * (l1 - l0) + (slo - l0) * (r1 - r0)) / (l1 - l0)`.

---

Headroom divides the capacity; it does not multiply the replicas. With the capacity
as `num / den`:

```python
usable_num = num * (100 - headroom_percent)
usable_den = den * 100
replicas = max(1, -(-requests_per_s * usable_den // usable_num))  # integer ceiling
```

Use one helper for both functions. For the curve, walk the points in order and stop
at the first one whose latency exceeds the objective; if the very first does, return
`None`; if none does, the capacity is the last rate.
