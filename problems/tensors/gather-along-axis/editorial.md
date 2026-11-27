## The idea

A gather along an axis is defined by one formula. For an input with three axes, read
along axis 1:

$$
\text{out}[i, j, k] = \text{input}[i,\ \text{index}[i, j, k],\ k] .
$$

Read it as: the output has the shape of the index; for each output entry, keep every
coordinate except the one on the gathered axis, and look that one up in the index. So
the index must have one entry per output entry, which means one axis per input axis.
Each function is this formula with the index reshaped to fit.

**`token_scores`** reads along `V`: `out[b, t, k] = logits[b, t, index[b, t, k]]`.
One token per position means `k` has a single value, so the index is `ids` with a
last axis of size 1, and the `(B, T, 1)` result loses that axis again.

**`take_columns`** reads along axis 1 of two: `out[b, k] = values[b, idx[b, k]]`. The
formula and the specification are the same line.

**`reorder_steps`** reads along `T`: `out[b, t, d] = x[b, index[b, t, d], d]`. The
specification wants `x[b, perm[b, t], d]`, so `index[b, t, d] = perm[b, t]` for every
`d`: the permutation repeated along the feature axis. `expand` makes that repetition
without copying.

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

In JAX and MLX:

```python
jnp.take_along_axis(logits, ids[:, :, None], axis=2).squeeze(2)
jnp.take_along_axis(values, idx, axis=1)
jnp.take_along_axis(x, perm[:, :, None], axis=1)  # the (B, T, 1) index is broadcast over D
```

The same reads as paired indexing, for comparison:

```python
b = torch.arange(B)[:, None]  # (B, 1)
t = torch.arange(T)[None, :]  # (1, T)
logits[b, t, ids]  # token_scores
values[b, idx]  # take_columns
x[b, perm]  # reorder_steps
```

Paired indexing names every indexed coordinate with a tensor. A gather names only the
one that varies and leaves the others implicit, which is why its index must have the
full rank.

## Common mistakes

- **The gather on the wrong axis.** `gather(logits, 1, ids[:, :, None])` reads
  `logits[b, ids[b, t], 0]`: the id is used as a position. It raises only when an id
  is at least `T`, so a small vocabulary or a long sequence hides it. The same holds for
  `gather(values, 0, idx)`, which uses the index as a row number.
- **A bare `squeeze()` after the gather.** `gather(logits, 2, ids[:, :, None])` is
  `(B, T, 1)`, and `squeeze()` removes the last axis together with every other axis of
  size 1: a batch of one comes back as `(T,)` and a single position as `(B,)`. Name
  the axis, `squeeze(2)`, or index it away with `[:, :, 0]`.
- **Plain indexing with the index tensor.** `logits[:, :, ids]` applies all of `ids`
  at every position and returns `(B, T, B, T)`.
- **An unexpanded index in PyTorch.** `gather(x, 1, perm[:, :, None])` returns
  `(B, T, 1)`: feature 0 of each chosen step and nothing else. The shape looks like
  something to fix afterwards, and expanding the *result* to `(B, T, D)` copies
  feature 0 into every feature. Expand the index, before the gather.
- **`perm` on the batch axis.** `gather(x, 0, index)` picks a batch row with a step
  number: `out[b, t] = x[perm[b, t], t]`. It runs whenever `T <= B`.
- **One permutation for the whole batch.** `x[:, perm[0]]` is right for a batch of one
  and for batches whose rows share a permutation.

## Complexity

Each function copies one value per entry of the result: $O(B\,T)$, $O(B\,K)$, and
$O(B\,T\,D)$. The expanded index of `reorder_steps` is a view in PyTorch and costs no
memory.

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

The difference that changes results is the fourth row. `torch.gather` takes
the index literally: the output has exactly the index's shape, and nothing is
broadcast. `torch.take_along_dim` and the two `take_along_axis` functions broadcast
the index against the other axes of `x`, so `(B, T, 1)` reorders whole steps. Code
ported from JAX or MLX to `torch.gather` therefore needs the `expand`, and without it
returns one feature.

The other difference is what happens to a bad index. `torch.gather` raises for an index
that is out of range or negative, and it is the only one that does:
`torch.take_along_dim` returned in-range values without raising when this was run on
PyTorch 2.15. JAX's `take_along_axis` fills an out-of-range read with NaN in a float
array and with the minimum integer in an integer array (plain indexing clamps
instead), and MLX does not check at all.
