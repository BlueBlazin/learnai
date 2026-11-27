Write out the gather formula for the axis you want to read along, putting the output coordinates on the left. For axis 2 of a 3D tensor, that is `out[b, t, k] = input[b, t, index[b, t, k]]`. Compare that formula with each function's specification to see which axis is gathered and what shape `index` needs.

---

- `token_scores` reads along the last axis (`V`). The gather expects a 3D index while `ids` has shape `(B, T)`. Add a trailing axis of size 1 to `ids`, gather along axis 2 to get `(B, T, 1)`, and remove axis 2 by name.
- `take_columns` reads along axis 1, and `idx` already has two axes like `values`.
- `reorder_steps` reads along axis 1 (`T`) of a 3D tensor, so `out[b, t, d] = x[b, index[b, t, d], d]`. The index needs an entry for every feature `d`, holding the same step index across all features.

---

The key lines in PyTorch (`jnp.take_along_axis(x, index, axis=...)` and `mx.take_along_axis` accept the same indices):

```
torch.gather(logits, 2, ids[:, :, None]).squeeze(2)          # (B, T, 1) -> (B, T)

torch.gather(values, 1, idx)                                 # (B, K)

index = perm[:, :, None].expand(-1, -1, D)                   # (B, T) -> (B, T, 1) -> (B, T, D)
torch.gather(x, 1, index)
```

In JAX and MLX, `reorder_steps` can keep the index at shape `(B, T, 1)` because `take_along_axis` broadcasts it over the feature axis.
