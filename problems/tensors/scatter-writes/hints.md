Write out the axis-1 scatter formula with the index on the left: `out[b, index[b, j]] = src[b, j]`. For each function, decide what initial tensor you are writing into (what untouched entries should hold), what `index` is, and what `src` is.

---

- `one_hot_rows`: start from float32 zeros of shape `(B, V)` and write `1.0` at column `ids[b]` of row `b`. The index needs two axes while `ids` has one. Comparing `ids` against `arange(V)` also builds the same table without a scatter.
- `token_counts`: start from integer zeros of shape `(B, V)` and add `1` to column `ids[b, t]` for every `t`. Multiple `t` can target the same column, so you need an adding write.
- `write_at`: `x`, `idx`, and `values` already match the three arguments of the scatter formula. Use a form that returns a new tensor rather than mutating `x`.

---

The key lines in PyTorch:

```
torch.zeros(B, V).scatter(1, ids[:, None], 1.0)                        # one_hot_rows

torch.zeros(B, V, dtype=torch.int64).scatter_add(1, ids, torch.ones_like(ids))

x.scatter(1, idx, values)                                              # write_at
```

In JAX, with `rows = jnp.arange(B)[:, None]`, use `zeros.at[rows, ids].add(1)` and `x.at[rows, idx].set(values)`. In MLX, use `mx.put_along_axis(x, idx, values, axis=1)` for setting writes and `zeros.at[rows, ids].add(1)` for counts.
