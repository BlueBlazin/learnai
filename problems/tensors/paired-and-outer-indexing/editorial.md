## The idea

Indexing several axes with integer tensors follows standard broadcasting. Two index tensors `i` and `j` are broadcast to a common shape `S`. The output has shape `S` followed by any unindexed trailing axes, and for every position `s` in `S`, the result reads one index from each tensor at `s`:

$$
\text{out}[s] = x[\,i[s],\ j[s]\,] \quad \text{for every position } s \text{ of } S .
$$

There is no separate "pairs" mode or "block" mode. The output pattern depends entirely on the shapes you pass.

- **`pick_pairs`.** Both `rows` and `cols` have shape `(N,)`, so `S = (N,)` and entry `n` is `x[rows[n], cols[n]]`.
- **`pick_block`.** A column of row indices `(R, 1)` and a row of column indices `(1, C)` broadcast to `S = (R, C)`. At position `[r, c]`, the first holds `rows[r]` and the second holds `cols[c]`, so every chosen row pairs with every chosen column.
- **`pick_per_row`.** `cols` has shape `(B, K)` and names step indices. Each entry belongs to its own batch row `b`, which must also be passed as an index tensor. `arange(B)[:, None]` has shape `(B, 1)`, broadcasts against `(B, K)`, and supplies `b` at every position `[b, k]`. The last axis of length `D` is unindexed and carries through, giving shape `(B, K, D)`.

## Reference solution (PyTorch)

```python
def pick_pairs(x, rows, cols):
    return x[rows, cols]


def pick_block(x, rows, cols):
    return x[rows[:, None], cols[None, :]]


def pick_per_row(x, cols):
    batch = torch.arange(x.shape[0], device=x.device)
    return x[batch[:, None], cols]
```

The JAX and MLX versions are identical, using `jnp.arange` and `mx.arange`.

`pick_block` can also be written in two steps as `x[rows][:, cols]`. The first indexing step selects the chosen rows whole, and the second selects columns from each row, at the cost of an intermediate copy. `pick_per_row` can also be written as a gather along axis 1.

## Common mistakes

- **Chained indexing for pairs.** `x[rows][:, cols]` returns the `(N, N)` block rather than the `N` pairs, which sit along its diagonal.
- **Paired indexing for a block.** `x[rows, cols]` with shapes `(R,)` and `(C,)` raises an error when `R` and `C` differ and neither is 1. When `R == C`, it silently returns `R` values along the block's diagonal. When one length is 1, the two broadcast to a 1D tensor of shape `(C,)` or `(R,)` instead of a 2D block.
- **Rows and columns swapped.** `x[cols, rows]` stays in bounds whenever `x` is square or the indices are small, and silently reads the transposed positions.
- **The block transposed.** `x[rows[None, :], cols[:, None]]` broadcasts to `(C, R)`. The shape reveals the bug unless `R == C`.
- **A slice for the batch axis.** `x[:, cols]` applies all of `cols` to every row, producing shape `(B, B, K, D)`. Writing `x[:, cols[0]]` gives the right shape, but copies row 0's choices to every row.
- **Row numbers that are not a column.** `x[arange(B), cols]` broadcasts `(B,)` against `(B, K)` from the right, aligning the row numbers with axis `K`. It raises an error unless `K == B` or one of them is 1. When `K == B`, entry `[b, k]` is read from row `k`. When `K == 1`, the indices broadcast to `(B, B)` and produce `(B, B, D)`. It only matches the spec when `B == 1`.

## Complexity

Each function copies one scalar (or one vector of `D` values) per output entry, giving time and space complexity $O(N)$, $O(R\,C)$, and $O(B\,K\,D)$.

## In the three frameworks

Here `i`, `j`, and `k` are integer tensors of shape `(N,)`, and `x` has shape `(B, T, D)`.

| | PyTorch | JAX | MLX |
|---|---|---|---|
| `x[i, j]`, `x[i[:, None], j]` | as above | the same | the same |
| `x[:, i, k]` | `(B, N)` | `(B, N)` | `(B, N)` |
| `x[i, :, k]` | `(N, T)` | `(N, T)` | `(N, T)` |
| `x[0, :, k]` | `(T, N)` | `(N, T)` | `(N, T)` |
| an index tensor out of range | `IndexError` | clamped to the last valid index | not checked, the value is unspecified |
| a Python int out of range | `IndexError` | clamped to the last valid index | `ValueError` |
| a negative index | counts from the end | counts from the end | counts from the end |
| index dtype | int32 or int64; a uint8 tensor is read as a mask | any integer type | any integer type |
| the 2D block in one call | none | `x[jnp.ix_(rows, cols)]` | none |
| a boolean mask as index | supported | supported | reading raises `ValueError` |

When integer index tensors are adjacent, their broadcast axes take their place in the output (`x[:, i, k]`). When a slice separates them, the broadcast axes move to the front (`x[i, :, k]`). All three frameworks agree on both cases. They disagree when one of the separated indices is a plain integer. JAX and MLX follow NumPy and move the indexed axis to the front, whereas PyTorch leaves it in place. Writing `x[0][:, k]` gives `(T, N)` in all three.

The second difference is bounds checking. Only PyTorch raises on an out-of-range index tensor. JAX clamps it to the nearest valid index, and MLX raises only for a Python `int` while leaving out-of-range index tensors unchecked. An off-by-one bug that PyTorch catches immediately can silently return a plausible value in JAX or MLX.
