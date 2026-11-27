Positions are 0-based. If a row already has `n` tokens in its cache, which positions
do they occupy, and which position does the next token receive? Check your rule for
`n = 0` (a fresh prompt) to verify it produces `0, 1, 2, ...`, just like uncached
attention.

---

`rope_positions` is a single broadcast. Add `cache_lengths` as a column of shape
`(B, 1)` to `arange(n_new)` as a row of shape `(n_new,)` to get a `(B, n_new)` table,
where each row starts at its own cache length. Do not collapse `cache_lengths` to a
single scalar, because each batch row is an independent sequence.

---

`rotate_new` rotates both input tensors using the same positions and base:

```
q_rot = ops.rope(q, positions, theta)
k_rot = ops.rope(k, positions, theta)
```

`ops.rope` accepts `positions` of shape `(B, T)` and applies row `b` across all heads
of batch row `b`. If you write the rotation directly, compute the angle table
`positions[:, :, None] * freqs` (where `freqs` holds the $D/2$ inverse frequencies)
with shape `(B, T, D/2)` and insert a head axis (`[:, None]`) so it broadcasts against
`(B, H, T, D/2)`. Always pass `theta` through rather than relying on a default base.
Cached keys are not an argument because they are already rotated and must not be
touched.
