## The idea

Indexing several axes with integer tensors has one rule, and it is the broadcasting
rule again. The index tensors are broadcast to a common shape `S`. The result has
shape `S` (followed by the axes that were not indexed), and each entry is read with
one index taken from each tensor at the same position:

$$
\text{out}[s] = x[\,i[s],\ j[s]\,] \quad \text{for every position } s \text{ of } S .
$$

There is no separate "pairs" mode and "block" mode. What you get follows from the
shapes you pass:

- **`pick_pairs`.** `rows` and `cols` are both `(N,)`, so `S = (N,)` and entry `n` is
  `x[rows[n], cols[n]]`.
- **`pick_block`.** A column of row indices `(R, 1)` and a row of column indices
  `(1, C)` broadcast to `S = (R, C)`. At position `[r, c]` the first holds `rows[r]`
  and the second `cols[c]`: every row index meets every column index.
- **`pick_per_row`.** `cols` is `(B, K)` and names a step. The row it belongs to is
  its own first coordinate `b`, which has to be supplied as an index tensor too:
  `arange(B)[:, None]`, shape `(B, 1)`, broadcasts against `(B, K)` and holds `b` at
  position `[b, k]`. The last axis is not indexed and comes along: `(B, K, D)`.

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

The JAX and MLX versions are the same lines with `jnp.arange` and `mx.arange`.

`pick_block` can also be written as two steps, `x[rows][:, cols]`: the first keeps the
chosen rows whole, the second picks columns from each. That is the block, at the cost
of an intermediate copy. `pick_per_row` can be written as a gather along axis 1, which
is the subject of Gathering Along an Axis.

## Common mistakes

- **Chained indexing for pairs.** `x[rows][:, cols]` is the `(N, N)` block. The pairs
  are its diagonal.
- **Paired indexing for a block.** `x[rows, cols]` with `(R,)` and `(C,)` raises when
  `R` and `C` differ and neither is 1. When `R == C` it returns `R` values, the
  diagonal of the block that was wanted. When one of them is 1 the two broadcast, and
  it returns a flat `(C,)` or `(R,)` tensor in place of a block with two axes.
- **Rows and columns swapped.** `x[cols, rows]` is in range whenever the table is
  square or the indices are small, and then reads the mirror entries.
- **The block transposed.** `x[rows[None, :], cols[:, None]]` broadcasts to `(C, R)`.
  The shape gives it away unless `R == C`.
- **A slice for the batch axis.** `x[:, cols]` applies all of `cols` to every row:
  shape `(B, B, K, D)`. `x[:, cols[0]]` has the right shape and gives every row the
  choices of row 0.
- **Row numbers that are not a column.** `x[arange(B), cols]` broadcasts `(B,)` against
  `(B, K)` from the right, so the row numbers line up with the `K` axis. It raises
  unless `K == B` or one of the two is 1. With `K == B`, entry `[b, k]` is read from
  row `k`. With `K == 1` the two broadcast to `(B, B)` and the result is `(B, B, D)`.
  Only with `B == 1` is it right.

## Complexity

Each function copies one value (or one vector of `D` values) per entry of the result:
$O(N)$, $O(R\,C)$, and $O(B\,K\,D)$.

## In the three frameworks

`i`, `j`, `k` are integer tensors of the same shape `(N,)`, and `x` has shape
`(B, T, D)`.

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
| the block in one call | none | `x2[jnp.ix_(rows, cols)]` | none |
| a boolean mask as index | supported | supported | reading raises `ValueError` |

Where the index tensors are next to each other, the broadcast axes take their place in
the result (`x[:, i, k]`). Where a slice separates them, the broadcast axes go to the
front (`x[i, :, k]`). The three agree on both. They differ when one of the separated
indices is a plain integer: JAX and MLX follow NumPy and still move the indexed axis to
the front, and PyTorch leaves it in place. `x[0][:, k]` is `(T, N)` in all three.

The second difference is error handling. Only PyTorch raises for an index that is out
of range. JAX clamps it. MLX raises for a Python int and does not check an index tensor. So an
off-by-one in an index tensor that PyTorch reports
returns a plausible value in the other two.
