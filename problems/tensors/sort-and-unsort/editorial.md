## The idea

Sorting a row produces two permutations, and most bugs come from using one where the
other is meant.

| | question it answers | for `x = [0.5, 2.0, 0.5, 3.0]` |
|---|---|---|
| `order[k]` | which column is at rank `k`? | `[3, 1, 0, 2]` |
| `rank[v]` | at which rank is column `v`? | `[2, 1, 3, 0]` |

`argsort` returns `order`. Reading through it sorts: `x[order]` is
`[3.0, 2.0, 0.5, 0.5]`. The two are inverses of each other, `rank[order[k]] = k`.

**`rank_descending`.** A stable sort leaves equal values in index order, which is the
tie rule asked for, provided the sort itself runs in the descending direction.
PyTorch does that with `descending=True, stable=True`. JAX and MLX sort the negated
values in ascending order: negation turns largest-first into smallest-first and
keeps equal values equal, so the stable sort still puts the lowest index first.

**`inverse_permutation`.** `out[b, perm[b, k]] = k` is a scatter: write the value `k`
at column `perm[b, k]`. Every column is written exactly once, because a row of `perm`
is a permutation.

**`keep_to_original_order`.** `out[b, order[b, k]] = keep_sorted[b, k]` is the same
scatter with the decisions as the values written. Equivalently it is a gather through
the inverse, `keep_sorted[b, rank[b, v]]`.

Sorting is a gather through `order`, and unsorting is a scatter through `order` or a
gather through `rank`. A gather through `order` twice is not the identity.

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

The inverse is also `argsort(perm)`: sorting a permutation puts the value `k` at
position `k`, and the index it came from is the column where `k` was. It costs a sort,
$O(V \log V)$ against the scatter's $O(V)$, and Top-k Filtering uses it for its
brevity.

## Common mistakes

- **An ascending sort, reversed.** `argsort(x, stable=True).flip(-1)` gives the
  values largest first and the ties highest index first. On the example it returns
  `[3, 1, 2, 0]`.
- **An unstable sort.** In the runs for this problem,
  `torch.argsort(x, descending=True)` kept ties in index order on every row of up to
  23 values and not on longer ones. Tests on small vocabularies pass, and a real
  vocabulary does not.
- **A tie-break by a small bonus.** Sorting `x - eps * arange(V)` looks like "lowest
  column first" without a stable sort. Any `eps` is wrong somewhere: it reorders
  values that differ by less than `eps * V`, and next to a value of 100 float32
  cannot represent a bonus of `1e-6` at all.
- **A bare `squeeze()` on the indices.** `torch.sort(...).indices.squeeze()` loses the
  batch axis of a single row and the column axis when `V == 1`.
- **`order` used as `rank`.** `argsort(argsort(x))` is the rank of each column. It is
  what a `rank < k` filter needs and not what a gather that sorts needs. On small
  examples the two often coincide, because short permutations are often their own
  inverses.
- **A gather where the inverse needs a write.** `gather(arange, 1, perm)` reads the
  ranks through `perm` and returns `perm` itself. The index belongs on the left-hand
  side: a scatter.
- **Unsorting with `order` again.** `keep_sorted[order]` permutes a second time in
  the same direction.
- **One row's order for the whole batch.** `out[:, order[0]] = keep_sorted` unsorts
  every row as if it were row 0. Each row was sorted by its own values and has its
  own order.
- **Sorting the wrong axis.** `dim=0` orders the rows within each column and returns
  the right shape.
- **A float mask.** Scattering into `zeros(B, V)` returns float32. Start from
  `zeros_like(keep_sorted)`, which is boolean.

## Complexity

The sort is $O(B\,V \log V)$. The inverse and the unsort are one write per entry,
$O(B\,V)$.

## In the three frameworks

| | PyTorch | JAX | MLX |
|---|---|---|---|
| descending, ties lowest index first | `torch.argsort(x, dim=-1, descending=True, stable=True)` | `jnp.argsort(-x, axis=-1)`; `jnp.argsort(x, axis=-1, descending=True)` | `mx.argsort(-x, axis=-1)` |
| stable | only with `stable=True` | by default (`stable=True`) | always; documented in 0.32.4 |
| `descending=` argument | yes | yes | no: `TypeError` |
| `stable=` argument | yes | yes | no: `TypeError` |
| default axis | last | last | last |
| dtype of the indices | int64 | int32 | uint32 |
| values and indices in one call | `torch.sort` returns both | `jnp.sort` returns values only | `mx.sort` returns values only |
| NaN in an ascending sort | last | last | last |
| NaN with `descending=True` | first | first | no such argument |
| NaN when sorting `-x` | last | last | last |

Every entry was run on PyTorch 2.15, JAX 0.12, and MLX 0.32.4.

Stability is the difference that changes results. PyTorch's default sort makes no
promise about ties and only keeps them in order on short rows. JAX and MLX are
stable, but for MLX that is a statement of the documentation of one version
(`print(mx.argsort.__doc__)`), to be read again when the version changes.

The index dtype matters when the order is passed on. MLX's `argsort` returns uint32,
which `mx.take_along_axis` and `mx.put_along_axis` accept. `torch.gather` accepts
int32 and int64 only and raised `RuntimeError` for a uint32 index.

NaN breaks the equivalence between the two descending spellings: `descending=True`
puts NaN first in PyTorch and JAX, and sorting `-x` leaves it last. This problem has
no NaN; a filter that can receive one has to decide where it goes.
