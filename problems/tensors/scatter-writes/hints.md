Write the scatter formula for axis 1 with the index on the left:
`out[b, index[b, j]] = src[b, j]`. For each function, decide what the tensor being
written into is before the write (what do the entries hold that no index names?),
what `index` is, and what `src` is.

---

- `one_hot_rows`: start from zeros of shape `(B, V)` and write a 1 at column `ids[b]`
  of row `b`. The index must have two axes, and `ids` has one. A comparison of `ids`
  against `arange(V)` gives the same table without any write.
- `token_counts`: start from integer zeros of shape `(B, V)` and send a 1 to column
  `ids[b, t]` for every `t`. Several `t` can name the same column, so the write has to
  add.
- `write_at`: `x`, `idx`, and `values` are already the three arguments of the
  formula. Use the form that returns a new tensor.

---

The key lines, in PyTorch:

```
torch.zeros(B, V).scatter(1, ids[:, None], 1.0)                        # one_hot_rows

torch.zeros(B, V, dtype=torch.int64).scatter_add(1, ids, torch.ones_like(ids))

x.scatter(1, idx, values)                                              # write_at
```

In JAX, with `rows = jnp.arange(B)[:, None]`: `zeros.at[rows, ids].add(1)` and
`x.at[rows, idx].set(values)`. In MLX: `mx.put_along_axis(x, idx, values, axis=1)`
for the setting writes and `zeros.at[rows, ids].add(1)` for the counts.
