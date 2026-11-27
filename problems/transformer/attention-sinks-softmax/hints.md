A softmax over $S$ key scores and one sink logit is just a softmax over $S + 1$ numbers. Try writing the specification as standard attention with one extra column appended to the score matrix, then think about what to do with that column's weight after the softmax.

---

Compute the `(B, H, T, S)` score tensor as usual, including both the division by $\sqrt{d}$ and the mask (`-inf` wherever `mask` is `False`). Next, reshape `sinks` from `(H,)` to `(1, H, 1, 1)` and broadcast it to `(B, H, T, 1)`. Concatenate this column onto the scores along the last axis to form a `(B, H, T, S + 1)` tensor.

The order of operations enforces the rules automatically. Because the sink is appended **after** scaling, it is not divided by $\sqrt{d}$. Because it is appended **after** masking, it is never masked.

---

```python
logits = concatenate((scores, sink), axis=-1)  # (B, H, T, S + 1)
weights = softmax(logits, axis=-1)
y = weights[..., :-1] @ v  # drop the sink's column; do not renormalize
```

Make sure the slice removes the exact column you added (for example, drop index `0` if you prepended the sink). Fully masked rows need no special case, because the sink column keeps the softmax well defined (`[-inf, ..., -inf, sink]` maps to `[0, ..., 0, 1]`), and slicing off the sink leaves a zero output vector.
