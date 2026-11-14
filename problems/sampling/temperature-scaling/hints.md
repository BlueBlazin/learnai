The temperatures have shape `(B,)` and the logits `(B, V)`. What shape must the
temperatures have so that row $b$ of the logits is divided by `temperature[b]`, and
not column $v$ by `temperature[v]`? And what does `x / 0` give in float32 for a
positive `x`, a negative `x`, and `x = 0`?

---

Compute two candidate results for the whole batch and select per row. The scaled
candidate is `logits / t` with `t` of shape `(B, 1)`. The greedy candidate is a table
that is `0` at each row's first maximum and `-inf` everywhere else. A `where` on the
condition `t == 0` then picks the greedy candidate for the zero-temperature rows.
Make sure the scaled candidate does not divide by zero in the rows you are going to
discard: an `inf` or NaN that is computed and then thrown away is a trap as soon as
someone differentiates through this code or turns on anomaly checks.

---

Divide by `where(t == 0, 1, t)`, so greedy rows are divided by 1. For the greedy
candidate, `argmax` along the last axis (keeping the dimension) returns the lowest
index among tied maxima; comparing it with `arange(V)` gives a `(B, V)` boolean table
with exactly one `True` per row, and `where(is_top, 0, -inf)` turns it into logits.
Build the table with `where`, not with arithmetic on the logits: `-inf * 0` and
`-inf - (-inf)` are NaN.
