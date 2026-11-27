## The idea

Sorting a row involves two inverse permutations, and most bugs come from confusing them.

| | question it answers | for `x = [0.5, 2.0, 0.5, 3.0]` |
|---|---|---|
| `order[k]` | which column is at rank `k`? | `[3, 1, 0, 2]` |
| `rank[v]` | at which rank is column `v`? | `[2, 1, 3, 0]` |

`argsort` returns `order`. Gathering through `order` sorts the row, so `x[order]` is `[3.0, 2.0, 0.5, 0.5]`. The two permutations satisfy `rank[order[k]] = k`.

**`rank_descending`.** A stable sort preserves the original index order of equal values, which matches the tie-breaking rule as long as the sort runs in the descending direction. PyTorch supports this directly with `descending=True, stable=True`. In JAX and MLX, negating `x` and running a stable ascending sort turns largest-first into smallest-first while keeping equal values equal, so ties still come out lowest index first.

**`inverse_permutation`.** The equation `out[b, perm[b, k]] = k` is a scatter that writes `k` into column `perm[b, k]`. Because each row of `perm` is a permutation, every column is written exactly once.

**`keep_to_original_order`.** The equation `out[b, order[b, k]] = keep_sorted[b, k]` is the same scatter, writing the boolean decisions instead of `k`. Equivalently, you can gather `keep_sorted` through the inverse permutation `rank` via `keep_sorted[b, rank[b, v]]`.

In short, sorting is a gather through `order`, while unsorting is either a scatter through `order` or a gather through `rank`. Gathering through `order` twice does not undo the sort.

## Reference solution (PyTorch)

```python
def rank_descending(x):
    return torch.argsort(x, dim=-1, descending=True, stable=True)


def inverse_permutation(perm):
    ranks = torch.arange(perm.shape[1]).expand_as(perm)  # ranks[b, k] = k
    return torch.zeros_like(perm).scatter(1, perm, ranks)


def keep_to_original_order(keep_sorted, order):
    return torch.zeros_like(keep_sorted).scatter(1, order, keep_sorted)
```

In JAX:

```python
jnp.argsort(-x, axis=-1)
rows = jnp.arange(B)[:, None]
jnp.zeros_like(perm).at[rows, perm].set(jnp.arange(V))
jnp.zeros_like(keep_sorted).at[rows, order].set(keep_sorted)
```

In MLX:

```python
mx.argsort(-x, axis=-1)
mx.put_along_axis(mx.zeros_like(perm), perm, mx.arange(V)[None, :], axis=1)
mx.put_along_axis(mx.zeros_like(keep_sorted), order, keep_sorted, axis=1)
```

You can also compute the inverse permutation with `argsort(perm)`. Sorting a permutation places value `k` at position `k`, so the index it moved from is the column where `k` originally sat. That takes $O(V \log V)$ instead of the scatter's $O(V)$, though top-k filtering sometimes uses it for brevity.

## Common mistakes

- **Reversing an ascending sort.** `argsort(x, stable=True).flip(-1)` sorts values largest first, but reverses each tied group so ties appear highest index first. On the example row it returns `[3, 1, 2, 0]`.
- **Using an unstable sort.** In tests for this problem, `torch.argsort(x, descending=True)` preserved tie order on rows up to 23 elements and broke it on longer rows. Small unit tests pass while full vocabularies fail.
- **Breaking ties with a small offset.** Sorting `x - eps * arange(V)` fails at both ends of precision. It reorders distinct values that differ by less than `eps * V`, and near `100.0` float32 cannot even represent a `1e-6` offset.
- **Calling bare `squeeze()` on indices.** `torch.sort(...).indices.squeeze()` drops the batch axis when `B == 1` and drops the column axis when `V == 1`.
- **Using `order` where `rank` is needed.** `argsort(argsort(x))` gives the rank of each column, which is what a `rank < k` filter needs, not what a gather-to-sort needs. Small examples hide this bug because short permutations are often their own inverses.
- **Using gather instead of scatter for the inverse.** `gather(arange, 1, perm)` reads ranks through `perm` and returns `perm` unchanged. Because `perm` indexes the output position, the operation must be a scatter.
- **Unsorting with `order` again.** Indexing `keep_sorted` through `order` applies the permutation a second time in the same direction instead of inverting it.
- **Reusing row 0's order for the whole batch.** Writing `out[:, order[0]] = keep_sorted` unsorts every row with row 0's permutation. Each row has its own sort order.
- **Sorting along the wrong axis.** Using `dim=0` sorts across rows within each column while still returning shape `(B, V)`.
- **Scattering into a float tensor.** Scattering into `zeros(B, V)` produces float32 output instead of booleans. Allocate the target with `zeros_like(keep_sorted)`.

## Complexity

The sort takes $O(B\,V \log V)$ time. Both `inverse_permutation` and `keep_to_original_order` perform one write per entry, taking $O(B\,V)$ time.

## In the three frameworks

| | PyTorch | JAX | MLX |
|---|---|---|---|
| descending, ties lowest index first | `torch.argsort(x, dim=-1, descending=True, stable=True)` | `jnp.argsort(-x, axis=-1)`; `jnp.argsort(x, axis=-1, descending=True)` | `mx.argsort(-x, axis=-1)` |
| stable | only with `stable=True` | by default (`stable=True`) | always; documented in 0.32.4 |
| `descending=` argument | yes | yes | no (`TypeError`) |
| `stable=` argument | yes | yes | no (`TypeError`) |
| default axis | last | last | last |
| dtype of the indices | int64 | int32 | uint32 |
| values and indices in one call | `torch.sort` returns both | `jnp.sort` returns values only | `mx.sort` returns values only |
| NaN in an ascending sort | last | last | last |
| NaN with `descending=True` | first | first | no such argument |
| NaN when sorting `-x` | last | last | last |

Every entry above was verified on PyTorch 2.15, JAX 0.12, and MLX 0.32.4.

Stability is the main behavioral difference. PyTorch's default sort makes no guarantee on ties and only happens to keep them in order on short rows. JAX and MLX are stable, though for MLX that guarantee comes from the 0.32.4 docstring (`print(mx.argsort.__doc__)`) and should be rechecked across versions.

Index dtypes also matter downstream. MLX's `argsort` returns `uint32`, which `mx.take_along_axis` and `mx.put_along_axis` accept. By contrast, `torch.gather` only accepts `int32` and `int64` and raises `RuntimeError` on `uint32`.

NaN values break the equivalence between `descending=True` and sorting `-x`. In PyTorch and JAX, `descending=True` puts NaN first, whereas sorting `-x` leaves NaN last. This problem forbids NaN, but any production filter that can encounter NaN must choose where it belongs.
