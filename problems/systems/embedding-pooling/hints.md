For each pooling mode, identify which token positions in a sequence are allowed to contribute. Then trace where those positions sit under right padding versus left padding. Notice that the last column of the tensor is not always the last real token.

---

For `"mean"`, broadcast `mask` to `(B, T, 1)` and multiply it with `states` to zero out padded positions. Sum along the sequence axis `T` and divide by the sum of the mask rather than by `T`.

For `"cls"` and `"last"`, compute one target index per sequence and gather from `states`. Start with `positions = arange(T)`. The last real token is the largest index where the mask is 1, and the first real token is the smallest. You can express both as a `max` or `min` over `T` after replacing masked-out positions with a sentinel value that cannot win.

---

```python
last = (positions * mask).max(axis=1)  # 0 can never beat a real position
first = where(mask != 0, positions, T).min(axis=1)  # T can never beat one either
pooled = states[arange(B), index]  # (B, D)
```

Normalize with `p / maximum(norm(p, axis=-1, keepdims=True), 1e-12)` so a zero vector does not become `0 / 0`. Implement `truncate_matryoshka` by applying the same normalization to `e[..., :dim]`.
