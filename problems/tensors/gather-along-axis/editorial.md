## The idea

A gather along an axis follows a single rule. For a 3D input gathered along axis 1, each output entry at `(i, j, k)` reads from the input at that same `i` and `k`, using `index[i, j, k]` as the axis-1 coordinate:

$$
\text{out}[i, j, k] = \text{input}[i,\ \text{index}[i, j, k],\ k] .
$$

The output takes the shape of `index`. Because each output position looks up its gathered coordinate in `index`, `index` must have the same number of axes as `input`. Each of the three functions applies this rule after reshaping the index to match.

**`token_scores`** reads along the vocabulary axis (`V`, axis 2), computing `out[b, t, k] = logits[b, t, index[b, t, k]]`. Because we select one token per `(b, t)` position, `k` has size 1. Adding a trailing size-1 axis to `ids` gives shape `(B, T, 1)`, and removing axis 2 from the `(B, T, 1)` output restores `(B, T)`.

**`take_columns`** reads along axis 1 of a 2D tensor, computing `out[b, k] = values[b, idx[b, k]]`. The index `idx` already has two axes, so it passes directly into the gather.

**`reorder_steps`** reads along the time axis (`T`, axis 1), computing `out[b, t, d] = x[b, index[b, t, d], d]`. We want `x[b, perm[b, t], d]`, which means `index[b, t, d]` must equal `perm[b, t]` for every feature `d`. Repeating `perm[:, :, None]` along the feature axis with `expand` creates that `(B, T, D)` index without copying memory.

## Reference solution (PyTorch)

```python
def token_scores(logits, ids):
    return torch.gather(logits, 2, ids[:, :, None]).squeeze(2)


def take_columns(values, idx):
    return torch.gather(values, 1, idx)


def reorder_steps(x, perm):
    index = perm[:, :, None].expand(-1, -1, x.shape[2])
    return torch.gather(x, 1, index)
```

In JAX and MLX, `take_along_axis` broadcasts size-1 axes automatically:

```python
jnp.take_along_axis(logits, ids[:, :, None], axis=2).squeeze(2)
jnp.take_along_axis(values, idx, axis=1)
jnp.take_along_axis(x, perm[:, :, None], axis=1)  # the (B, T, 1) index is broadcast over D
```

For comparison, here are the same three reads written with paired indexing:

```python
b = torch.arange(B)[:, None]  # (B, 1)
t = torch.arange(T)[None, :]  # (1, T)
logits[b, t, ids]  # token_scores
values[b, idx]  # take_columns
x[b, perm]  # reorder_steps
```

Paired indexing supplies an explicit index tensor for every indexed axis. A gather names only the axis that varies and leaves the remaining coordinates implicit, which is why the index must match the input's rank.

## Common mistakes

- **Gathering along the wrong axis.** Calling `gather(logits, 1, ids[:, :, None])` reads `logits[b, ids[b, t], 0]`, treating the token id as a sequence position. It only raises an error when an id is at least `T`, so a small vocabulary or a long sequence hides the bug. The same happens with `gather(values, 0, idx)`, which treats column indices as row numbers.
- **Using a bare `squeeze()` after the gather.** In `token_scores`, `gather(logits, 2, ids[:, :, None])` has shape `(B, T, 1)`. Calling `squeeze()` without an axis removes every size-1 axis, turning a batch of 1 into `(T,)` and a single step into `(B,)`. Always name the axis with `squeeze(2)` or index it away with `[:, :, 0]`.
- **Using plain indexing with the index tensor.** Writing `logits[:, :, ids]` applies the entire `(B, T)` index tensor at every `(b, t)` position and returns shape `(B, T, B, T)`.
- **Forgetting to expand the index in PyTorch.** Calling `gather(x, 1, perm[:, :, None])` returns shape `(B, T, 1)`, containing only feature 0 of each chosen step. Expanding the *result* to `(B, T, D)` afterwards copies feature 0 across all features. Expand `perm[:, :, None]` before calling `gather`.
- **Applying `perm` on the batch axis.** Calling `gather(x, 0, index)` picks a batch row using a step number (`out[b, t] = x[perm[b, t], t]`). This runs without error whenever `T <= B`.
- **Reusing one permutation for the whole batch.** Writing `x[:, perm[0]]` works only when `B == 1` or when every batch item shares the same permutation.

## Complexity

Each function copies one float per output entry, taking $O(B\,T)$ time for `token_scores`, $O(B\,K)$ for `take_columns`, and $O(B\,T\,D)$ for `reorder_steps`. In PyTorch, `expand` returns a view and allocates no extra memory.

## In the three frameworks

| | PyTorch | JAX | MLX |
|---|---|---|---|
| the operation | `torch.gather(x, dim, index)` | `jnp.take_along_axis(x, index, axis=a)` | `mx.take_along_axis(x, index, axis=a)` |
| NumPy-style spelling | `torch.take_along_dim(x, index, dim=d)` | the same function | the same function |
| index with fewer axes than `x` | `RuntimeError` | `ValueError` | `ValueError` |
| index of size 1 where `x` is larger | `gather`: output has size 1 there. `take_along_dim`: broadcast | broadcast | broadcast |
| index dtype | `gather`: int32 or int64. `take_along_dim`: int64 only | any integer type | any integer type |
| an index out of range | `gather`: `RuntimeError`. `take_along_dim`: not checked, returns some in-range value | NaN there for a float input, the minimum integer for an integer input | not checked, the value is unspecified |
| a negative index | `gather`: `RuntimeError`. `take_along_dim`: counts from the end | counts from the end | counts from the end |

The fourth row is the difference most likely to change results. `torch.gather` takes the index shape literally without broadcasting, so the output always matches `index.shape`. By contrast, `torch.take_along_dim`, `jnp.take_along_axis`, and `mx.take_along_axis` broadcast size-1 axes of the index against `x`, so a `(B, T, 1)` index reorders full feature vectors. Porting code from JAX or MLX to `torch.gather` therefore requires an explicit `expand`.

The frameworks also differ on out-of-range and negative indices. `torch.gather` is the only variant that raises an error for both. On PyTorch 2.15, `torch.take_along_dim` returned in-range values without raising. JAX's `take_along_axis` fills out-of-range reads with `NaN` for float arrays and the minimum integer for integer arrays (unlike plain indexing, which clamps), while MLX performs no bounds check.
