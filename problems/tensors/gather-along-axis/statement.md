Write three functions that read a tensor along one axis, with a different index at
every position of the other axes. Each framework has one operation for this, `gather`
in PyTorch and `take_along_axis` in JAX and MLX, and nearly all of the work is giving
it an index of the shape it expects.

## Specification

```python
def token_scores(logits, ids): ...
def take_columns(values, idx): ...
def reorder_steps(x, perm): ...
```

**`token_scores(logits, ids)`** takes `logits` of shape `(B, T, V)` and token ids of
shape `(B, T)`, and returns the score of the chosen token at every position, shape
`(B, T)`:

$$
\text{out}[b, t] = \text{logits}[b,\ t,\ \text{ids}[b, t]] .
$$

**`take_columns(values, idx)`** takes `values` of shape `(B, V)` and `idx` of shape
`(B, K)`, and returns `K` entries of every row, shape `(B, K)`:

$$
\text{out}[b, k] = \text{values}[b,\ \text{idx}[b, k]] .
$$

**`reorder_steps(x, perm)`** takes `x` of shape `(B, T, D)` and `perm` of shape
`(B, T)`, where every row of `perm` is a permutation of $0, \dots, T - 1$. It reorders
the steps of each row by that row's permutation and returns shape `(B, T, D)`:

$$
\text{out}[b, t, d] = x[b,\ \text{perm}[b, t],\ d] .
$$

- `logits`, `values`, `x`: float32 tensors.
- `ids`, `idx`, `perm`: integer tensors of your framework's default integer type
  (int64 in PyTorch, int32 in JAX and MLX). Every index is in range and not negative.
  Entries of `ids` and `idx` may repeat.
- Every function returns a float32 tensor.

## Shapes

| name | shape | dtype |
|---|---|---|
| `logits` | `(B, T, V)` | float32 |
| `ids` | `(B, T)` | integer |
| result of `token_scores` | `(B, T)` | float32 |
| `values` | `(B, V)` | float32 |
| `idx` | `(B, K)` | integer |
| result of `take_columns` | `(B, K)` | float32 |
| `x` | `(B, T, D)` | float32 |
| `perm` | `(B, T)` | integer |
| result of `reorder_steps` | `(B, T, D)` | float32 |

## Tools

`torch.gather(input, dim, index)`, `jnp.take_along_axis(arr, indices, axis)`, and
`mx.take_along_axis(a, indices, axis)` all do the same thing. For three axes and
axis 1:

$$
\text{out}[i, j, k] = \text{input}[i,\ \text{index}[i, j, k],\ k] .
$$

The index replaces the coordinate along the chosen axis, and every other coordinate is
the output's own. The output has the shape of the index.

![Along axis 1, entry `k` of the output row is read from the position that entry `k` of the index row names.](assets/gather-row.svg)

The index must have **as many axes as the input**. An index with fewer axes gains them
with `None`, as in Broadcasting on Purpose, and the result then has an axis of size 1
to remove by name, as in Axes of Size One.

> [!WARNING]
> The formula has an index entry for every output entry, feature axis included. In
> PyTorch, `gather` does not broadcast the index: an index of size 1 along an axis
> returns only the first slice of that axis.

Paired indexing, from Paired and Outer Indexing, can express the same reads, and is a
correct way to solve this problem too.

## Example

Every entry of `logits` and of `x` is `100 b + 10 t + v`, and every entry of `values`
is `10 b + v`, so each one names its own position:

```python
logits = [
    [[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]],
    [[100.0, 101.0], [110.0, 111.0], [120.0, 121.0]],
]  # shape (2, 3, 2)
ids = [[1, 0, 1], [0, 0, 1]]
token_scores(logits, ids)
# [[  1.0,  10.0,  21.0],
#  [100.0, 110.0, 121.0]]                                       # shape (2, 3)

values = [[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]]  # shape (2, 4)
idx = [[3, 0, 3], [1, 2, 2]]
take_columns(values, idx)
# [[ 3.0,  0.0,  3.0],
#  [11.0, 12.0, 12.0]]                                          # shape (2, 3)

x = logits  # shape (2, 3, 2)
perm = [[2, 0, 1], [0, 2, 1]]
reorder_steps(x, perm)
# [[[ 20.0,  21.0], [  0.0,   1.0], [ 10.0,  11.0]],            # row 0: steps 2, 0, 1
#  [[100.0, 101.0], [120.0, 121.0], [110.0, 111.0]]]            # row 1: steps 0, 2, 1
```

In the last result both features of a step move together: `[20.0, 21.0]`, never
`[20.0, 20.0]`.

## Constraints

- Results are compared with the reference exactly: values are copied.
- Every size is between 1 and 16. Any of them can be 1, and any two can be equal.
- No Python loops are needed.

## Why this matters

Sequence Log-Likelihood and Perplexity is `token_scores` applied to log-probabilities:
the score of each target token, summed. Experts in a Narrower Space reads the scores
of the experts that top-k chose with `take_columns`, as every top-k and top-p filter
does after its sort. Heavy-Hitter KV Eviction keeps a different set of positions for
every head, which is `reorder_steps` with a shorter index, expanded over the feature
axis in the same way. Sorting and Unsorting, later in this module, depends on all
three. A gather along the wrong axis does not raise when the sizes allow it, and an
index that was not expanded returns one feature where a vector was meant.
