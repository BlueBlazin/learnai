Positions are 0-based. If a row already has `n` tokens in its cache, which positions
do they occupy, and which position does the next token get? Work it out for `n = 0`
(a fresh prompt) and check that your rule gives the positions `0, 1, 2, ...` you would
use without a cache.

---

`rope_positions` is one broadcast: `cache_lengths` as a column of shape `(B, 1)` plus
`arange(n_new)` as a row of shape `(n_new,)` gives the `(B, n_new)` table, one row of
consecutive positions per batch row, each starting at that row's own length. Do not
reduce `cache_lengths` to a single number: the rows are independent sequences.

---

`rotate_new` rotates the two tensors it is given, both with the same positions and
the same base:

```
q_rot = ops.rope(q, positions, theta)
k_rot = ops.rope(k, positions, theta)
```

`ops.rope` takes positions of shape `(B, T)` and applies row `b` to batch row `b`
across all heads. If you write the rotation yourself, the angle table is
`positions[:, :, None] * freqs`, of shape `(B, T, D/2)`, and it needs a head axis
inserted (`[:, None]`) to broadcast against `(B, H, T, D/2)`. Pass `theta` on; the
default base is not always the model's. There is nothing to do for the cached keys:
they are not an argument, because they must not be touched.
