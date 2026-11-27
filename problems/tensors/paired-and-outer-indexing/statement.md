Write three functions that read a tensor through integer index tensors: one entry per
pair of indices, a block of rows crossed with columns, and a different choice of steps
for every row of a batch. The same indexing syntax does all three, and what it returns
depends only on the shapes of the index tensors.

## Specification

```python
def pick_pairs(x, rows, cols): ...
def pick_block(x, rows, cols): ...
def pick_per_row(x, cols): ...
```

**`pick_pairs(x, rows, cols)`** takes `x` of shape `(H, W)` and two index tensors of
shape `(N,)`. It returns the `N` entries named by the pairs `(rows[n], cols[n])`:

$$
\text{out}[n] = x[\text{rows}[n],\ \text{cols}[n]] .
$$

**`pick_block(x, rows, cols)`** takes `x` of shape `(H, W)`, `rows` of shape `(R,)`,
and `cols` of shape `(C,)`. It returns the `(R, C)` block where the chosen rows cross
the chosen columns:

$$
\text{out}[r, c] = x[\text{rows}[r],\ \text{cols}[c]] .
$$

**`pick_per_row(x, cols)`** takes `x` of shape `(B, T, D)` and `cols` of shape `(B, K)`.
Row `b` of `cols` lists `K` steps to take from row `b` of `x`. It returns shape
`(B, K, D)`:

$$
\text{out}[b, k, d] = x[b,\ \text{cols}[b, k],\ d] .
$$

- `x`: a float32 tensor.
- `rows`, `cols`: integer tensors of your framework's default integer type. Every
  index is in range and not negative. An index may appear more than once.
- Every function returns a float32 tensor.

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

## Tools

When two or more axes are indexed with integer tensors, as in `x[i, j]`, the index
tensors are first **broadcast against each other** and then read together, entry by
entry. The result has the broadcast shape, and entry `[...]` of it is
`x[i[...], j[...]]`. Axes that are not indexed, or are sliced with `:`, are kept whole.

So the shapes of `i` and `j` decide what you get, by the rule of Broadcasting on
Purpose. `arange` makes the index tensor that names every row in turn.

> [!WARNING]
> `x[rows, cols]` never means "these rows and these columns". It pairs the two tensors.
> And `x[:, cols]` does not give each row its own columns: the slice takes every row,
> and all of `cols` is applied to each of them.

## Example

Every entry of `x` is `10 i + j`, so it names its own row and column:

```python
x = [[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]]  # shape (3, 4)

pick_pairs(x, [2, 0, 1], [1, 2, 0])  # [21.0, 2.0, 10.0]

pick_block(x, [2, 0], [3, 1, 0])
# [[23.0, 21.0, 20.0],
#  [ 3.0,  1.0,  0.0]]                            # shape (2, 3)
```

For `pick_per_row`, every entry of `x` is `100 b + 10 t + d`:

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

- Results are compared with the reference exactly: values are copied.
- Every size is between 1 and 16. Any of them can be 1, and any two can be equal.
- No Python loops are needed.

## Why this matters

Gathering the Last Real Token is `pick_pairs` with `arange(B)` as the rows.
Verifying a Block of Sampled Drafts reads the probability of each drafted token with
the same pairing, and Ragged Acceptance in a Batch reads a different position in every
row, which is `pick_per_row`. The mistakes here do not raise in the cases that matter.
Pairing where a block was meant fails loudly when the two index tensors differ in
length, and returns a diagonal when they happen to match. A slice where a per-row
index was meant returns a tensor with one axis too many, which the next broadcast
often absorbs.
