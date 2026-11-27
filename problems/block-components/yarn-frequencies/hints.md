Split `yarn_inv_freq` into two parts that you can verify independently. The first part
computes two plain Python floats, `low` and `high`, from `dim`, `base`,
`original_max_positions`, the two betas, and `truncate`, with no tensors involved. The
second part turns those bounds into a ramp over the pair index and blends the two
frequency vectors. For the worked example, check what `low` and `high` are before and
after rounding, and verify which pairs land at 0 and at 1 on the ramp.

---

Use `boundary(beta) = dim * log(L0 / (beta * 2 * pi)) / (2 * log(base))` with the
natural logarithm from `math`, where `L0 = original_max_positions`. Because a larger
`beta` gives a *smaller* index, `beta_fast` produces `low` and `beta_slow` produces
`high`. Order matters for the next two steps. First apply `floor(low)` and `ceil(high)`
when `truncate` is true, and then clamp with `max(low, 0)` and `min(high, dim - 1)`.

Evaluate the ramp at the pair indices `0, 1, ..., dim/2 - 1` (`arange(dim // 2)` in
float32), not at the channel indices `0, 2, 4, ...` used in the frequency exponent.

---

```python
index = arange(dim // 2)                              # float32
ramp = clip((index - low) / (high - low), 0, 1)
theta = base ** (-arange(0, dim, 2) / dim)
inv_freq = (1 - ramp) * theta + ramp * theta / factor
```

For `yarn_tables`, compute `angles = positions[:, None] * inv_freq` (with `positions`
cast to float32) to get shape `(T, P)`. Then multiply both `cos(angles)` and
`sin(angles)` by `0.1 * math.log(factor) + 1`.
