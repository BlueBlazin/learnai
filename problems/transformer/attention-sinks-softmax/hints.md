A softmax over $S$ scores and one more number is a softmax over $S + 1$ numbers. Can
you express the whole specification as ordinary attention with one extra column in
the score matrix, and decide afterwards what to do with that column's weight?

---

Compute the scores `(B, H, T, S)` as usual, including the division by $\sqrt{d}$ and
the mask (`-inf` where `mask` is `False`). Then build the sink column: `sinks` has
shape `(H,)`, so reshape it to `(1, H, 1, 1)` and broadcast it to `(B, H, T, 1)`.
Concatenate it to the scores along the last axis to get `(B, H, T, S + 1)`.

The order of these steps is the specification. The sink is appended **after** the
scaling, so it is not divided by $\sqrt{d}$, and **after** the masking, so it is
never masked.

---

```python
logits = concatenate((scores, sink), axis=-1)  # (B, H, T, S + 1)
weights = softmax(logits, axis=-1)
y = weights[..., :-1] @ v  # drop the sink's column; do not renormalize
```

The slice must remove the column you added: if you put the sink first, drop column 0.
Rows that are fully masked need no special case, because the sink column keeps the
softmax well defined (`[-inf, ..., -inf, sink]` becomes `[0, ..., 0, 1]`), and after
the slice the output is zero.
