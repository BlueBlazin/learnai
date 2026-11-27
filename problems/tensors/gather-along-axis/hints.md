Write the gather formula for the axis you want to read along, with the output's
coordinates on the left. For axis 2 of three: `out[b, t, k] = input[b, t, index[b, t,
k]]`. Now compare it with the formula in the specification. Which axis is being read
along in each function, and what shape must `index` have for the two formulas to say
the same thing?

---

- `token_scores` reads along the last axis (`V`). The formula wants an index with three
  axes, and you have `(B, T)`. Give `ids` a last axis of size 1; the result is then
  `(B, T, 1)`, and that axis has to be removed again (by name).
- `take_columns` reads along axis 1, and `idx` already has two axes like `values`.
- `reorder_steps` reads along axis 1 (`T`) of three axes: `out[b, t, d] = x[b,
  index[b, t, d], d]`. The index needs an entry for every `d`, and it is the same
  step number for all of them.

---

The key lines, in PyTorch (`jnp.take_along_axis(x, index, axis=...)` and
`mx.take_along_axis` take the same indices):

```
torch.gather(logits, 2, ids[:, :, None]).squeeze(2)          # (B, T, 1) -> (B, T)

torch.gather(values, 1, idx)                                 # (B, K)

index = perm[:, :, None].expand(-1, -1, D)                   # (B, T) -> (B, T, 1) -> (B, T, D)
torch.gather(x, 1, index)
```

In JAX and MLX the last index can stay `(B, T, 1)`: `take_along_axis` broadcasts it
over the feature axis.
