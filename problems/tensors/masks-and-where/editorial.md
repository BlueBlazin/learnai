## The idea

A mask says which entries exist. Everything computed afterwards should behave as if
the others were not there, and the way to get that is to **replace** them, before the
computation, with a value the computation ignores:

| reduction | replace a dropped entry with | because |
|---|---|---|
| sum, mean | `0` | adding 0 changes nothing |
| max, and softmax over scores | `-inf` | it loses to every real value, and `exp(-inf) = 0` |
| min | `inf` | it loses the other way |

`where(mask, x, fill)` does the replacement by selection. It reads the mask and
copies either `x` or the fill. The dropped entry of `x` takes part in no arithmetic,
so it can be `inf` or NaN.

**`fill_masked`** is that sentence as code.

**`masked_mean`.** Replace with 0, sum over the steps, divide by the count. The mask
is `(B, T)` and `x` is `(B, T, D)`. Broadcasting aligns from the right, which would
pair `T` with `D`, so the mask becomes `(B, T, 1)`. The count is `(B,)` and the sums
are `(B, D)`: the count needs its own second axis for the same reason. A row with no
kept step has sum 0 and count 0. Dividing by `max(count, 1)` turns 0 / 0 into 0 / 1
and leaves every other row alone.

**`masked_max`.** Replace with `-inf` and take the max. A row with no kept step is all
`-inf`, and its max is `-inf` without a special case. That is also the honest answer:
`-inf` is the max of nothing, in the way 0 is the sum of nothing.

## Reference solution (PyTorch)

```python
def fill_masked(scores, keep):
    return torch.where(keep, scores, float("-inf"))


def masked_mean(x, mask):
    kept = torch.where(mask[:, :, None], x, 0.0)
    count = mask.sum(dim=1)[:, None]  # (B, 1)
    return kept.sum(dim=1) / count.clamp(min=1)


def masked_max(x, mask):
    return torch.where(mask[:, :, None], x, float("-inf")).amax(dim=1)
```

In JAX and MLX the same lines read `jnp.where` / `mx.where`, `axis=1`, `.max(axis=1)`,
and `jnp.maximum(count, 1)` / `mx.maximum(count, 1)`.

## Common mistakes

- **Multiplying by the mask.** `x * mask` is `x` where the mask is 1 and `0 * x`
  where it is 0, and `0 * inf` and `0 * nan` are NaN. The NaN then spreads through
  the sum. It works on every test whose padding happens to hold finite numbers. In
  JAX the outcome also depends on the shapes (see the table below): a boolean mask of
  the shape of `x` gave 0, a broadcast one NaN. A result that changes with the
  framework and with the shape of the mask is the reason to select with `where`.
- **The max over `x * mask`.** Even with finite data, a dropped entry becomes 0 and
  wins against a row of negative values.
- **Adding a mask of 0 and `-inf`.** `scores + where(keep, 0, -inf)` is the usual
  additive attention mask, and it is fine while the scores are finite. A dropped score
  of `inf` gives `inf - inf`, which is NaN.
- **The mask the wrong way round.** `scores.masked_fill(keep, -inf)` fills the kept
  entries. `masked_fill` takes the places to overwrite, so it needs `~keep`; `where`
  takes the places to keep.
- **The mask on the feature axis.** `mask[:, None, :]` is `(B, 1, T)` and lines up
  with the last axis of `x`. It raises unless `T == D`, and then it drops features
  where steps were meant. The steps are axis 1: `mask[:, :, None]`.
- **A 0 for the max of nothing.** Replacing the `-inf` of a row that keeps no step
  with 0 makes that row look like one whose largest value is 0.
- **`-1e9` for `-inf`.** It is not smaller than every float32, and a max or a softmax
  over a row that keeps nothing then returns an ordinary-looking number.
- **Dividing by `T`.** The mean of the kept steps divides by how many were kept.
- **No guard for a count of 0.** `0 / 0` is NaN. Clamp the count; do not add an
  epsilon to it, which changes every other row slightly.
- **Counts that are not aligned.** `(B, D) / (B,)` raises for most shapes and divides
  column `d` by the count of row `d` when `B == D`.
- **A bare `squeeze()` at the end.** A sum with `keepdim=True` followed by `squeeze()`
  loses the batch axis of a single row. Reduce without `keepdim`, or name the axis.

## Complexity

One pass over the input for each function: $O(B\,V)$ and $O(B\,T\,D)$. `where`
allocates the selected copy; nothing else is stored.

## In the three frameworks

| | PyTorch | JAX | MLX |
|---|---|---|---|
| select | `torch.where(c, a, b)` | `jnp.where(c, a, b)` | `mx.where(c, a, b)` |
| a Python number for `a` or `b` | yes, result float32 | yes, result float32 | yes, result float32 |
| an integer condition | `RuntimeError` | nonzero counts as true | nonzero counts as true |
| `where(c)` with one argument | the indices of the true entries | the indices of the true entries | `TypeError` |
| fill by method | `x.masked_fill(~keep, v)` | none | none |
| fill by assignment | `x[~keep] = v` | none (arrays are immutable) | `x[~keep] = v` |
| read `x[mask]` | a flat list of the kept entries | the same; `NonConcreteBooleanIndexError` under `jit` | `ValueError` |
| a reduction with a mask argument | none | `jnp.sum(x, axis, where=m)`, `jnp.max(x, axis, where=m, initial=-jnp.inf)` | none |
| `x * mask` at a dropped `inf` or NaN, boolean mask | NaN | 0 when the mask has the shape of `x`, NaN when it is broadcast | NaN |
| invert | `~mask`; `1 - mask` raises `NotImplementedError` | `~mask`; `1 - mask` is int32 | `~mask`; `1 - mask` is int32 |
| `mask.sum()` | int64 | int32 | int32 |

Every entry was run on PyTorch 2.15, JAX 0.12, and MLX 0.32.4.

Three differences matter. MLX cannot read through a boolean index, so code that must
run there selects with `where` and keeps its shapes; that is also the form that
works under `jax.jit`, where the length of `x[mask]` would depend on the data. JAX
alone has reductions that take the mask directly. And multiplication by a boolean mask
in JAX gave 0 at a dropped `inf` when the mask had the shape of `x`, and NaN when the
mask was `(B, T, 1)` against `(B, T, D)`. With a float mask it was NaN in all three.
Do not build on the first case.
