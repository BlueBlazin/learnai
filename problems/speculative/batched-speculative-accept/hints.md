Start from the single-sequence function and ask, for each line, what shape it has
once there is a batch axis in front. The acceptance tests become a `(B, K)` boolean
matrix. "The first failure" is now a per-row quantity: which reduction, along which
axis, turns that matrix into `(B,)` counts without a loop? And once every row has
its own stopping position, how do you pick row `b`'s distributions at position
`n[b]` in one indexing operation?

---

`verify_batch`: gather $p_{b,i}[x_{b,i}]$ and $q_{b,i}[x_{b,i}]$ along the
vocabulary axis, compare, and take the running product along the draft axis; its
sum over that axis is `n_accepted`. Index with a pair of index vectors,
`p[arange(B), n_accepted]`, to get a `(B, V)` matrix of each row's own position.
Appending a zero row to `q` along the position axis makes "bonus" and "rejected"
the same expression, `max(0, p[rows, n] - q_ext[rows, n])`. The inverse CDF is a
cumulative sum along the vocabulary axis compared with a `(B, 1)` threshold,
`u_resample[:, None] * cdf[:, -1:]`: each row has its own uniform and its own
total.

---

`advance_batch`: think column by column instead of row by row. With
`cols = arange(W)[None, :]`, `start = lengths[:, None]`, and
`stop = (lengths + n_accepted)[:, None]`, column `c` of row `b` is

- the old token if `c < start`,
- draft `c - start` if `start <= c < stop`,
- the next token if `c == stop`,
- `pad_id` otherwise.

Gather the old tokens and the drafts with clamped indices (so that every index is
valid even where the value will not be used) and choose with `where`. `W` is
`max(lengths + n_accepted + 1)`, and `cache_lengths` is `lengths + n_accepted`.
