Take one row by hand: `x = [0.5, 2.0, 0.5, 3.0]`. Write the order (column at each
rank) and, under it, the rank of each column. These are two different lists.
`rank_descending` returns the first, `inverse_permutation` turns one into the other,
and `keep_to_original_order` needs to move an entry from rank `k` to column
`order[k]`.

---

- `rank_descending`: you need a stable sort that is descending without being a
  reversed ascending one. PyTorch has both arguments. Where there is no `descending`
  argument, sort something else in ascending order: which transformation of `x`
  turns "largest first" into "smallest first" and leaves equal values equal?
- `inverse_permutation`: the specification `out[b, perm[b, k]] = k` has the index on
  the left-hand side. That is a write of the values `0, 1, ..., V - 1`. (Sorting
  `perm` itself also answers it: the argsort of a permutation is its inverse.)
- `keep_to_original_order`: the specification again has the index on the left. Or
  read `keep_sorted` through the inverse permutation.

---

The key lines, in PyTorch:

```
torch.argsort(x, dim=-1, descending=True, stable=True)             # rank_descending

ranks = torch.arange(V).expand_as(perm)
torch.zeros_like(perm).scatter(1, perm, ranks)                     # inverse_permutation

torch.zeros_like(keep_sorted).scatter(1, order, keep_sorted)       # keep_to_original_order
```

In JAX and MLX the sort is `argsort(-x, axis=-1)`. The writes are
`zeros.at[rows, perm].set(...)` in JAX, with `rows = jnp.arange(B)[:, None]`, and
`mx.put_along_axis(zeros, perm, ..., axis=1)` in MLX.
