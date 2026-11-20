Split the function into two parts that can be checked separately. The first produces
two plain Python numbers, `low` and `high`, from `dim`, `base`,
`original_max_positions`, the two betas, and `truncate`; no tensors are involved. The
second turns them into a ramp over the pair index and blends two frequency vectors.
For the worked example, what are `low` and `high` before and after the rounding, and
which pairs does the ramp leave at 0 and at 1?

---

`boundary(beta) = dim * log(L0 / (beta * 2 * pi)) / (2 * log(base))` with the natural
logarithm from `math`. A larger `beta` gives a *smaller* index, so `beta_fast` gives
`low` and `beta_slow` gives `high`. The order of the next two steps matters: first
`floor(low)` and `ceil(high)` when `truncate` is true, then `max(low, 0)` and
`min(high, dim - 1)`.

The ramp is evaluated at the pair indices `0, 1, ..., dim/2 - 1` (an `arange(dim // 2)`
in float32), not at the channel indices `0, 2, 4, ...` that appear in the exponent of
the frequencies.

---

```
index = arange(dim // 2)                              # float32
ramp = clip((index - low) / (high - low), 0, 1)
theta = base ** (-arange(0, dim, 2) / dim)
inv_freq = (1 - ramp) * theta + ramp * theta / factor
```

For the tables, `angles = positions[:, None] * inv_freq` (positions converted to
float32) has shape `(T, P)`, and both `cos(angles)` and `sin(angles)` are multiplied
by `0.1 * math.log(factor) + 1`.
