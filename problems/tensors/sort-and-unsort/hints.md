Work through one row by hand with `x = [0.5, 2.0, 0.5, 3.0]`. Write out `order` (the column at each rank) and, below it, `rank` (the rank of each column). These are two different lists. `rank_descending` returns `order`, `inverse_permutation` turns `order` into `rank`, and `keep_to_original_order` moves each entry from rank `k` to column `order[k]`.

---

- **`rank_descending`**: You need a stable descending sort rather than a reversed ascending sort. PyTorch provides both `descending=True` and `stable=True`. In frameworks without a `descending` flag, transform `x` before an ascending sort so that the largest values become the smallest while equal values stay equal.
- **`inverse_permutation`**: In `out[b, perm[b, k]] = k`, the index `perm[b, k]` appears on the left-hand side, which means you should scatter `0, 1, ..., V - 1` into those positions. Alternatively, `argsort(perm)` also computes the inverse of a permutation.
- **`keep_to_original_order`**: In `out[b, order[b, k]] = keep_sorted[b, k]`, the index is again on the left-hand side, so scatter `keep_sorted` using `order`. Alternatively, gather `keep_sorted` using `inverse_permutation(order)`.

---

The key lines in PyTorch:

```python
torch.argsort(x, dim=-1, descending=True, stable=True)             # rank_descending

ranks = torch.arange(V).expand_as(perm)
torch.zeros_like(perm).scatter(1, perm, ranks)                     # inverse_permutation

torch.zeros_like(keep_sorted).scatter(1, order, keep_sorted)       # keep_to_original_order
```

In JAX and MLX, use `argsort(-x, axis=-1)` for `rank_descending`. For the writes, JAX uses `zeros.at[rows, perm].set(...)` with `rows = jnp.arange(B)[:, None]`, and MLX uses `mx.put_along_axis(zeros, perm, ..., axis=1)`.
