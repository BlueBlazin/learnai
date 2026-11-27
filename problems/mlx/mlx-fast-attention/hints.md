Compare `n_cached + T` with `S`. When they are equal, every buffer slot holds a valid token. Which built-in mask (`None` or `"causal"`) lets each new token see everything up to itself, and which lets every token see all slots? When `n_cached + T < S`, how can you express the allowed slots for new token $i$ as a comparison on slot index $j$?

---

A full buffer needs no mask array. Use `mask="causal"` when `causal` is `True` and `T > 1`, and `mask=None` otherwise. When the buffer has unused capacity, build a boolean mask that is `True` where new token $i$ may read slot $j$. Use `j <= n_cached + i` when `causal` is `True`, and `j < n_cached + T` when `causal` is `False`. A boolean mask means "may attend" and works for both `float16` and `float32` inputs without dtype promotion issues.

Pass `k` and `v` unchanged even when they have fewer heads than `q`.

---

```python
steps, slots = q.shape[2], k.shape[2]
last = n_cached + mx.arange(steps)[:, None] if causal else n_cached + steps - 1
mask = mx.arange(slots)[None, :] <= last  # (T, S) or (1, S), True = may attend
out = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5, mask=mask)
```

This boolean mask is valid for every layout. Use it when `n_cached + steps < slots`, and fall back to `"causal"` or `None` when the buffer is full.
