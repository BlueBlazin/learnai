Write three functions that read elements from a tensor along one axis using a different index at every position of the other axes. Each framework provides a dedicated operation for this pattern (`gather` in PyTorch, `take_along_axis` in JAX and MLX). The main task is shaping the index tensor to match what that operation expects.

## Specification

```python
def token_scores(logits, ids): ...
def take_columns(values, idx): ...
def reorder_steps(x, perm): ...
```

**`token_scores(logits, ids)`** takes token scores `logits` over a batch of size `B`, sequence length `T`, and vocabulary size `V`, along with chosen token indices `ids` for each batch item and sequence position. At each batch index `b` and step `t`, it returns the score of the token selected by `ids[b, t]`:

$$
\text{out}[b, t] = \text{logits}[b,\ t,\ \text{ids}[b, t]] .
$$

**`take_columns(values, idx)`** takes a 2D tensor `values` with `B` rows and `V` columns, and column indices `idx` selecting `K` entries per row. At each row `b` and selection index `k`, it reads the column named by `idx[b, k]`:

$$
\text{out}[b, k] = \text{values}[b,\ \text{idx}[b, k]] .
$$

**`reorder_steps(x, perm)`** takes a sequence of feature vectors `x` with batch size `B`, sequence length `T`, and feature dimension `D`, along with a tensor `perm` whose rows are each a permutation of $0, \dots, T - 1$. It reorders the time steps of each batch item according to that item's permutation while keeping every feature vector intact. For each batch item `b`, step `t`, and feature `d`:

$$
\text{out}[b, t, d] = x[b,\ \text{perm}[b, t],\ d] .
$$

All index tensors (`ids`, `idx`, `perm`) use your framework's default integer type (`int64` in PyTorch, `int32` in JAX and MLX). Every index is non-negative and within bounds. Entries of `ids` and `idx` may repeat.

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

`torch.gather(input, dim, index)`, `jnp.take_along_axis(arr, indices, axis)`, and `mx.take_along_axis(a, indices, axis)` all perform the same gather operation. For a 3D input gathered along axis 1, each output entry at `(i, j, k)` looks up its axis-1 coordinate from `index[i, j, k]`:

$$
\text{out}[i, j, k] = \text{input}[i,\ \text{index}[i, j, k],\ k] .
$$

The index replaces the coordinate on the chosen axis, while every other coordinate comes directly from the output position. The output matches the shape of the index.

![Along axis 1, entry `k` of the output row is read from the position that entry `k` of the index row names.](assets/gather-row.svg)

The index tensor must have **the same number of axes as the input**. If your index has fewer axes, insert size-1 axes with `None` (as in Broadcasting on Purpose). When that leaves a size-1 axis in the output, remove it by explicit axis index (as in Axes of Size One).

> [!WARNING]
> The gather formula needs an index entry for every output entry, including along any feature axis. In PyTorch, `torch.gather` does not broadcast the index. Passing an index of size 1 along a feature axis returns only the first slice of that axis instead of the full vector.

Paired indexing (from Paired and Outer Indexing) can express the same reads and is also a valid way to solve this problem.

## Example

In the examples below, each entry encodes its original coordinates: `logits[b, t, v] = 100 * b + 10 * t + v` (and `x = logits`), while `values[b, v] = 10 * b + v`.

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

In `reorder_steps`, both features of a step move together, producing `[20.0, 21.0]` rather than `[20.0, 20.0]`.

## Constraints

- Results are compared against the reference for exact equality because values are copied without arithmetic.
- Every dimension (`B`, `T`, `V`, `K`, `D`) is between 1 and 16. Any dimension can be 1, and any two dimensions can be equal.
- No Python loops are needed.

## Why this matters

These three gather patterns appear across transformer training and inference. Sequence Log-Likelihood and Perplexity applies `token_scores` to log-probabilities and sums the selected target scores. Experts in a Narrower Space uses `take_columns` to read the scores of top-k experts, just as top-k and top-p sampling do after sorting. Heavy-Hitter KV Eviction keeps a different set of token positions per attention head, which is `reorder_steps` with a shorter index expanded across the feature axis. Sorting and Unsorting later in this module builds on all three.

Gather bugs often fail silently. Gathering along the wrong axis does not raise an error whenever the indices happen to stay within that axis's bounds, and forgetting to expand an index returns a single feature where a full vector is expected.
