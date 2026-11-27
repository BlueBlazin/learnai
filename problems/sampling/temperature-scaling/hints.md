`temperature` has shape `(B,)` and `logits` has shape `(B, V)`. What shape does `temperature` need so row $b$ of `logits` is divided by `temperature[b]` instead of column $v$ by `temperature[v]`? Also check what `x / 0` produces in float32 when `x` is positive, negative, or `0`.

---

Compute both branches for the full batch and select per row. The scaled branch computes `logits / t` with `t` reshaped to `(B, 1)`. The greedy branch builds a table containing `0` at each row's first maximum and `-inf` everywhere else. A `where` on `t == 0` then picks the greedy branch for zero-temperature rows. Make sure the scaled branch never divides by zero, even in rows that will be discarded, because intermediate `inf` or NaN values can break gradients or trip anomaly checks.

---

Divide by `where(t == 0, 1, t)` so greedy rows safely divide by `1`. For the greedy branch, call `argmax` along the last axis with the dimension kept to get the lowest index among tied maxima. Compare that index against `arange(V)` to form a `(B, V)` boolean mask with one `True` per row, then apply `where(is_top, 0, -inf)`. Build this table with `where` rather than arithmetic on `logits`, since `-inf * 0` and `-inf - (-inf)` both produce NaN.
