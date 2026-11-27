Implement `pick_pairs`, `pick_block`, and `pick_per_row` to select elements from a tensor using integer index tensors. All three use the same multi-axis indexing syntax, where the shapes of the index tensors determine whether indices are paired, crossed into a block, or matched per batch row.

## Background

When two or more axes are indexed with integer tensors, as in `x[i, j]`, the index tensors are first **broadcast against each other** and then read together element by element. The output takes the broadcast shape, and entry `[...]` of the result is `x[i[...], j[...]]`. Any axis that is not indexed, or is sliced with `:`, is kept whole.

Because of this rule, the shapes of `i` and `j` determine what `x[i, j]` returns. You can use `arange` to build an index tensor that names every row in order.

> [!WARNING]
> `x[rows, cols]` never means "these rows and these columns." It pairs the two index tensors element by element. Likewise, `x[:, cols]` does not give each row its own columns. The slice selects every row and applies all of `cols` to each one.

## Specification

```python
def pick_pairs(x, rows, cols): ...
def pick_block(x, rows, cols): ...
def pick_per_row(x, cols): ...
```

**`pick_pairs(x, rows, cols)`** takes a 2D tensor `x` of shape `(H, W)` and two 1D index tensors `rows` and `cols` of shape `(N,)`. For each `n`, it reads the entry at `(rows[n], cols[n])` and returns shape `(N,)`:

$$
\text{out}[n] = x[\text{rows}[n],\ \text{cols}[n]] .
$$

**`pick_block(x, rows, cols)`** takes `x` of shape `(H, W)`, `rows` of shape `(R,)`, and `cols` of shape `(C,)`. It returns the `(R, C)` block formed where the selected rows intersect the selected columns:

$$
\text{out}[r, c] = x[\text{rows}[r],\ \text{cols}[c]] .
$$

**`pick_per_row(x, cols)`** takes a 3D tensor `x` of shape `(B, T, D)` and an index tensor `cols` of shape `(B, K)`. Row `b` of `cols` lists `K` step indices to gather from row `b` of `x`, keeping the last axis of length `D` intact. It returns shape `(B, K, D)`:

$$
\text{out}[b, k, d] = x[b,\ \text{cols}[b, k],\ d] .
$$

`rows` and `cols` use your framework's default integer dtype. Every index is non-negative and in range, and indices may repeat.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` of `pick_pairs` and `pick_block` | `(H, W)` | float32 |
| `rows`, `cols` of `pick_pairs` | `(N,)` | integer |
| result of `pick_pairs` | `(N,)` | float32 |
| `rows` of `pick_block` | `(R,)` | integer |
| `cols` of `pick_block` | `(C,)` | integer |
| result of `pick_block` | `(R, C)` | float32 |
| `x` of `pick_per_row` | `(B, T, D)` | float32 |
| `cols` of `pick_per_row` | `(B, K)` | integer |
| result of `pick_per_row` | `(B, K, D)` | float32 |

## Example

In the first two examples, every entry of `x` equals `10 * i + j` so each value encodes its row `i` and column `j`:

```python
x = [[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]]  # shape (3, 4)

pick_pairs(x, [2, 0, 1], [1, 2, 0])  # [21.0, 2.0, 10.0]

pick_block(x, [2, 0], [3, 1, 0])
# [[23.0, 21.0, 20.0],
#  [ 3.0,  1.0,  0.0]]                            # shape (2, 3)
```

For `pick_per_row`, every entry of `x` equals `100 * b + 10 * t + d`:

```python
x = [
    [[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]],  # row 0, steps 0, 1, 2
    [[100.0, 101.0], [110.0, 111.0], [120.0, 121.0]],
]  # row 1, steps 0, 1, 2
cols = [[2, 0, 2], [1, 1, 0]]  # shape (2, 3)

pick_per_row(x, cols)
# [[[ 20.0,  21.0], [  0.0,   1.0], [ 20.0,  21.0]],       # row 0: steps 2, 0, 2
#  [[110.0, 111.0], [110.0, 111.0], [100.0, 101.0]]]       # row 1: steps 1, 1, 0
#                                                          # shape (2, 3, 2)
```

## Constraints

- Results are compared with the reference exactly, because values are copied without arithmetic.
- Every dimension size is between 1 and 16. Any size can be 1, and any two sizes can be equal.
- No Python loops are needed.

## Why this matters

Gathering the last real token in a padded batch is `pick_pairs` with `arange(B)` as the row indices. Speculative decoding verification reads the probability of each drafted token using the same pairing, and ragged batch acceptance reads a different set of positions per row with `pick_per_row`.

Mistakes in integer indexing often fail silently. Using paired indexing where a block was intended raises an error when the two index tensors have different lengths, but quietly returns a diagonal when their lengths happen to match. Using a slice `:` where a per-row index was intended adds an extra axis that a later broadcast can silently absorb.
