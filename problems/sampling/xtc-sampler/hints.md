Break the logic into three questions for each row: which tokens are top choices, whether XTC acts on the row, and which single top choice survives. Once you have those, the output is a single `where`. Think about which of the three needs softmax and which can be answered directly from `logits`.

---

Compute `above = softmax(logits) >= threshold` to mark the top choices. XTC acts on row `b` when `above` has at least two true entries **and** `u[b] < probability`. Keep a trailing dimension of size 1 on both conditions so they broadcast across the vocabulary axis. To find the survivor without sorting, mask non-top-choice logits to `+inf` and take the row minimum.

---

```
above = softmax(logits) >= threshold
fires = (count(above) >= 2) & (u[:, None] < probability)
floor = min(where(above, logits, +inf), axis=-1, keepdims=True)
candidate = above & (logits == floor)
survivor = max(where(candidate, arange(V), -1), axis=-1, keepdims=True)
remove = fires & above & (arange(V) != survivor)
out = where(remove, -inf, logits)
```

Multiple tokens can be marked in `candidate` when logits tie, so taking the maximum index selects exactly one survivor. Compare `logits` rather than probabilities when checking for ties, because identical logits are equal bit-for-bit whereas computed probabilities may differ due to rounding.
