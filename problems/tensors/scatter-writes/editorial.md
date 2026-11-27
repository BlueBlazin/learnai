## The idea

A scatter along axis 1 writes each source entry into the column named by the index:

$$
\text{out}[i,\ \text{index}[i, j]] = \text{src}[i, j] .
$$

This is the gather formula with the index moved to the left-hand side. A gather has one index entry per output position, so every output entry is defined exactly once. A scatter has one index entry per *source* position, which raises two extra questions: what values sit in untouched output entries, and what happens when two source entries target the same position? The initial target tensor answers the first question, and the choice between a setting write and an adding write answers the second.

**`one_hot_rows`** scatters into a `(B, V)` tensor of zeros. Row `b` receives a single `1.0` at column `ids[b]`. Because the target is 2D, `ids` needs a second axis of size 1, giving shape `(B, 1)`.

**`token_counts`** scatters into a `(B, V)` tensor of integer zeros, sending `1` to column `ids[b, t]` for every `(b, t)`. A token that appears three times sends three `1`s to the same entry, and their sum is the count, so this requires an adding write.

**`write_at`** applies the scatter formula directly by writing `values` into a copy of `x` at `idx`. Because no column is repeated within a row, a setting write is well defined.

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

In JAX, all three writes use paired indexing through `.at`, with a `(B, 1)` row index that broadcasts against the column indices:

```python
rows = jnp.arange(B)[:, None]  # (B, 1)
jnp.zeros((B, V), dtype=jnp.float32).at[jnp.arange(B), ids].set(1.0)
jnp.zeros((B, V), dtype=jnp.int32).at[rows, ids].add(1)
x.at[rows, idx].set(values)
```

In MLX, setting writes use `put_along_axis` and the adding write uses `.at`:

```python
mx.put_along_axis(mx.zeros((B, V)), ids[:, None], mx.array(1.0), axis=1)
mx.zeros((B, V), dtype=mx.int32).at[rows, ids].add(1)
mx.put_along_axis(x, idx, values, axis=1)
```

`one_hot_rows` and `token_counts` can also be written without any scatter by broadcasting a comparison against `arange(V)`, though for `token_counts` this allocates a `(B, T, V)` intermediate tensor:

```python
(ids[:, None] == torch.arange(V)).float()  # one_hot_rows
(ids[:, :, None] == torch.arange(V)).sum(dim=1)  # token_counts
```

## Common mistakes

- **Counting with a setting write.** Calling `counts.scatter(1, ids, ones)` writes `1` for every occurrence, so later writes overwrite earlier ones and every count ends up as 0 or 1.
- **Counting with `counts[rows, ids] += 1`.** This reads `counts[rows, ids]`, adds 1 to that copy, and writes it back with a setting write. A repeated entry is read as 0 and written back as 1 multiple times, producing the same bug as a setting scatter.
- **Counting across the whole batch.** `bincount(ids.flatten())` ignores row boundaries. To use `bincount` per row, add row offsets `b * V` to row `b`, count into `B * V` bins, and reshape to `(B, V)`.
- **Scattering on the batch axis.** Calling `scatter(0, idx, values)` writes to `out[idx[b, k], k]`, using the column index as a row number. It raises only when an index is at least `B`, so a large batch hides the error.
- **Passing the index as the value.** Calling `x.scatter(1, idx, idx.to(x.dtype))` runs and produces the right shape, but fills the output with column numbers instead of `values`. In `scatter(dim, index, src)`, the second argument is the destination and the third is the value to write.
- **Using one row's columns for every row.** Writing `out[:, idx[0]] = values` writes every row at the columns named by row 0. A per-row index requires a paired row index (`out[rows, idx]`) or a scatter.
- **Mutating `x` in place.** Calling `x.scatter_` or `x[rows, idx] = values` in PyTorch or MLX modifies the caller's tensor. Use the out-of-place `scatter` or copy `x` first (`x.clone()`, `mx.array(x)`).
- **Returning the wrong dtype.** Returning floats from `token_counts` or integers from `one_hot_rows` fails the dtype contract. Note that `torch.nn.functional.one_hot` returns `int64` while `jax.nn.one_hot` returns `float32`, which is another reason to build the table with an explicit dtype.

## Complexity

On top of allocating the $O(B\,V)$ output, each scatter touches one entry per index element: $O(B)$ writes for `one_hot_rows`, $O(B\,T)$ for `token_counts`, and $O(B\,K)$ for `write_at`. The broadcasted comparison version of `token_counts` takes $O(B\,T\,V)$ time and memory.

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

Every entry was verified on PyTorch 2.15, JAX 0.12, and MLX 0.32.4.

The biggest difference across frameworks is the adding write. PyTorch provides `scatter_add` along an axis. JAX and MLX only offer adding writes through `.at[...].add` with paired indices, so you must construct the row index yourself. MLX's `put_along_axis` has no adding variant, and MLX's `.at` supports `add`, `subtract`, `multiply`, `divide`, `maximum`, and `minimum`, but not `set`.

Bounds checking also varies. PyTorch's scatter operations raise on out-of-range indices. JAX silently drops out-of-range writes, so a row count that comes up short in JAX often points to an id equal to `V`. MLX does not check bounds at all and writes to that flat offset in memory (landing in a later row, or dropping only past the end of the array), where a later row's own write can overwrite it and make the bug look like a dropped write.
