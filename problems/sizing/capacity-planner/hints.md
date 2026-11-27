Count time over a full cycle of $o$ steps instead of a single step. Each request occupies one slot for $o$ steps. Separate the cost paid once per step regardless of batch size from the work each slot contributes over those $o$ steps. Once $S(b)$ is written as a linear function of $b$, finding the largest batch that meets the SLO is a single integer division.

---

Keep every comparison in exact integers by avoiding early float division. Rewrite `S(b) / o <= tpot_slo_us` as `fixed + b * per_slot <= o * tpot_slo_us`, where `fixed = o * base_us` and `per_slot = prefill_us_per_token * p + decode_us_per_seq * (o - 1)`. Then `batch = min(max_seqs, (o * tpot_slo_us - fixed) // per_slot)`, returning `None` if `batch < 1`. Carry single-replica capacity as a numerator and denominator (or a `fractions.Fraction`) until the replica count is computed, and convert to `float` only for the float fields in the returned dict. For `plan_from_curve`, the interpolated capacity is also an exact fraction: `(r0 * (l1 - l0) + (slo_ms - l0) * (r1 - r0)) / (l1 - l0)`.

---

Headroom scales usable capacity down rather than multiplying the replica count up. Given single-replica capacity `num / den`:

```python
usable_num = num * (100 - headroom_percent)
usable_den = den * 100
replicas = max(1, -(-requests_per_s * usable_den // usable_num))  # integer ceiling
```

Share this helper between both functions. In `plan_from_curve`, walk the points in order and stop at the first point whose latency exceeds `slo_ms`. Return `None` if the first point already exceeds `slo_ms`, and use the last rate if no point exceeds it.
