## The idea

A scatter along axis 1 is one formula:

$$
\text{out}[i,\ \text{index}[i, j]] = \text{src}[i, j] .
$$

It is the gather formula with the index moved to the left-hand side. A gather has an
index entry for every output entry, so every output entry is defined exactly once. A
scatter has an index entry for every *source* entry, and that leaves two questions a
gather never raises: what do the output entries hold that no index names, and what
happens when two index entries name the same one? The first is answered by the tensor
you scatter into. The second is answered by the form of the write: setting or adding.

**`one_hot_rows`.** Scatter into zeros. Row `b` gets one write, a 1 at column
`ids[b]`. The index needs two axes, so `ids` becomes `(B, 1)`.

**`token_counts`.** Scatter into integer zeros, with a 1 for every `(b, t)` at column
`ids[b, t]`. A token that occurs three times in a row sends three 1s to the same
entry, and the count is their sum, so this has to be the adding form.

**`write_at`.** The formula as written: scatter `values` into a copy of `x` at `idx`.
No column is named twice, so the setting form is well defined.

## Reference solution (PyTorch)

```python
def one_hot_rows(ids, V):
    out = torch.zeros(ids.shape[0], V, dtype=torch.float32)
    return out.scatter(1, ids[:, None], 1.0)


def token_counts(ids, V):
    counts = torch.zeros(ids.shape[0], V, dtype=torch.int64)
    return counts.scatter_add(1, ids, torch.ones_like(ids))


def write_at(x, idx, values):
    return x.scatter(1, idx, values)
```

In JAX the writes are paired indexing behind `.at`, with a row index that broadcasts
against the columns:

```python
rows = jnp.arange(B)[:, None]  # (B, 1)
jnp.zeros((B, V), dtype=jnp.float32).at[jnp.arange(B), ids].set(1.0)
jnp.zeros((B, V), dtype=jnp.int32).at[rows, ids].add(1)
x.at[rows, idx].set(values)
```

In MLX the setting writes are `put_along_axis` and the adding write is `.at`:

```python
mx.put_along_axis(mx.zeros((B, V)), ids[:, None], mx.array(1.0), axis=1)
mx.zeros((B, V), dtype=mx.int32).at[rows, ids].add(1)
mx.put_along_axis(x, idx, values, axis=1)
```

Two of the functions need no write at all. A comparison against `arange(V)` builds
the same tables by broadcasting, at the cost of a `(B, T, V)` intermediate for the
counts:

```python
(ids[:, None] == torch.arange(V)).float()  # one_hot_rows
(ids[:, :, None] == torch.arange(V)).sum(dim=1)  # token_counts
```

## Common mistakes

- **Counting with a setting write.** `counts.scatter(1, ids, ones)` writes a 1 for
  every occurrence, and the later writes replace the earlier ones. Every count is 0
  or 1.
- **Counting with `counts[rows, ids] += 1`.** This reads `counts[rows, ids]`, adds 1
  to that copy, and assigns it back with a setting write. A repeated entry is read
  once as 0 and written several times as 1. It is the same bug in a form that looks
  like a loop body.
- **Counting over the whole batch.** `bincount(ids.flatten())` has no row in it.
  With row offsets it is a correct solution: add `b * V` to row `b`, count to
  `B * V`, and reshape.
- **The scatter on the batch axis.** `scatter(0, idx, values)` writes
  `out[idx[b, k], k]`: the index picks a row. It raises only when an index is at
  least `B`, so a tall batch hides it.
- **The index written as the value.** `x.scatter(1, idx, idx.to(x.dtype))` runs and
  has the right shape; the result holds column numbers where `values` belong. In
  `scatter(dim, index, src)` the second argument says where and the third says what.
- **One row's columns for every row.** `out[:, idx[0]] = values` writes each row at
  the columns named by row 0. A per-row index needs a row index next to it,
  `out[rows, idx]`, or a scatter.
- **`scatter_` when the caller still needs `x`.** The trailing underscore writes into
  the argument. So does `x[rows, idx] = values` in PyTorch and MLX; copy first
  (`x.clone()`, `mx.array(x)`).
- **A float count, or an integer one-hot.** `torch.nn.functional.one_hot` returns
  int64 and `jax.nn.one_hot` returns float32, which is one reason to know how to build
  the table yourself with the dtype you want.

## Complexity

Each write touches one entry per index entry: $O(B)$ writes for the one-hot rows,
$O(B\,T)$ for the counts, and $O(B\,K)$ for `write_at`, on top of the $O(B\,V)$ result.
The comparison form of the counts costs $O(B\,T\,V)$ time and memory.

## In the three frameworks

| | PyTorch | JAX | MLX |
|---|---|---|---|
| setting write, new tensor | `x.scatter(dim, index, src)` | `x.at[rows, cols].set(values)`; `jnp.put_along_axis(x, index, values, axis, inplace=False)` | `mx.put_along_axis(x, index, values, axis)` |
| setting write, in place | `x.scatter_(dim, index, src)`; `x[rows, cols] = values` | none: assignment raises `TypeError` | `x[rows, cols] = values` |
| adding write | `x.scatter_add(dim, index, src)`; `x.index_put((rows, cols), values, accumulate=True)` | `x.at[rows, cols].add(values)` | `x.at[rows, cols].add(values)` |
| `x[rows, cols] += 1` with a repeated entry | adds 1 once | `TypeError` | adds 1 once |
| a Python number as the value | `scatter`: yes. `scatter_add`: `TypeError` | yes | `put_along_axis`: `TypeError`, pass `mx.array(1.0)`. Assignment and `.at[].add`: yes |
| index with fewer axes than `x` | `scatter`: `RuntimeError` | `put_along_axis`: `ValueError` | `put_along_axis`: `ValueError` |
| values of another dtype | `scatter`: `RuntimeError` | cast to the dtype of `x` | cast to the dtype of `x` |
| an index out of range | `scatter`, `scatter_add`: `RuntimeError` | `.at[].set`, `.at[].add`: the write is dropped, no error | `put_along_axis`, `x[rows, cols] = v`, `.at[].add`: no error; column `V + c` of row `b` is written at column `c` of row `b + 1`, and a write past the end of the array is dropped |
| a negative index | `scatter`, `scatter_add`: `RuntimeError` | counts from the end | counts from the end |
| built-in one-hot | `torch.nn.functional.one_hot`, int64 | `jax.nn.one_hot`, float32 | none |

Every entry was run on PyTorch 2.15, JAX 0.12, and MLX 0.32.4.

The frameworks differ most in the adding write. PyTorch has it as a scatter along an
axis. JAX and MLX have it only as `.at[...].add` with paired indices, so the row index
has to be built by hand; MLX's `put_along_axis` has no adding variant, and MLX's `.at`
has `add`, `subtract`, `multiply`, `divide`, `maximum`, and `minimum` but no `set`.

They also differ in what a bad index does. PyTorch's scatters raise. JAX drops an
out-of-range write silently, and MLX does not check: the write goes to the entry at
that offset in the flattened array, which is in a later row, and is dropped only
past the end of the array. (A setting write into the next row can then be
overwritten by that row's own write, which looks like a drop.) A count that comes out
one short in JAX is usually an id equal to `V`.
