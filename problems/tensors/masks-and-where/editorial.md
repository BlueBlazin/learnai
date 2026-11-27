## The idea

A mask marks which entries belong to the computation. Every downstream operation should behave as if the masked-out entries did not exist. To achieve that without changing tensor shapes, replace each masked-out entry with a neutral value that the reduction ignores:

| reduction | replace a dropped entry with | because |
|---|---|---|
| sum, mean | `0` | adding 0 changes nothing |
| max, and softmax over scores | `-inf` | it loses to every real value, and `exp(-inf) = 0` |
| min | `inf` | it loses the other way |

`where(mask, x, fill)` performs this replacement by selection. It reads the mask and copies either `x` or `fill`. Because no arithmetic runs on the unchosen entry of `x`, masked-out `inf` and `NaN` values cannot leak into the output.

**`fill_masked`** applies `where` directly, keeping `scores` where `keep` is true and filling `-inf` elsewhere.

**`masked_mean`** replaces masked-out steps with `0`, sums over the step axis, and divides by the number of kept steps in each row. Because `x` has shape `(B, T, D)` and `mask` has shape `(B, T)`, right-aligned broadcasting would pair `T` with `D`. Adding a trailing axis gives `mask[:, :, None]` of shape `(B, T, 1)`. The per-row counts have shape `(B,)` while the sums have shape `(B, D)`, so the counts need a trailing axis `(B, 1)` for the same reason. A row with no kept steps has sum `0` and count `0`. Clamping the denominator with `max(count, 1)` turns `0 / 0` into `0 / 1` while leaving every other row unchanged.

**`masked_max`** replaces masked-out steps with `-inf` and takes the maximum over the step axis. A row with no kept steps becomes all `-inf`, so its maximum is `-inf` with no special case. That matches the mathematical identity for an empty maximum, just as `0` is the identity for an empty sum.

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

In JAX and MLX, use `jnp.where` or `mx.where`, `axis=1`, `.max(axis=1)`, and `jnp.maximum(count, 1)` or `mx.maximum(count, 1)`.

## Common mistakes

- **Multiplying by the mask.** `x * mask` computes `0 * x` on masked-out entries, and `0 * inf` and `0 * nan` are `NaN`. That `NaN` then spreads through the reduction. This bug hides whenever test padding happens to be finite. In JAX, the result also depends on tensor shapes (see the table below), returning `0` for a same-shape boolean mask and `NaN` for a broadcast one. Always select with `where`.
- **Taking the max over `x * mask`.** Even on finite data, masked-out entries become `0` and beat negative kept values.
- **Adding a mask of `0` and `-inf`.** `scores + where(keep, 0, -inf)` works only when `scores` is finite. A masked-out `inf` gives `inf - inf`, which is `NaN`.
- **Inverting the condition.** `scores.masked_fill(keep, -inf)` overwrites the kept entries instead of the dropped ones. `masked_fill` takes the positions to overwrite (`~keep`), whereas `where` takes the positions to keep.
- **Putting the mask on the feature axis.** `mask[:, None, :]` has shape `(B, 1, T)` and aligns with axis `D`. It raises an error unless `T == D`, in which case it silently masks features instead of steps. Steps are axis 1, so the mask must be `mask[:, :, None]`.
- **Returning `0` for an empty max.** Replacing `-inf` with `0` on a row with no kept steps makes it indistinguishable from a row whose largest value is `0`.
- **Using `-1e9` instead of `-inf`.** `-1e9` is not smaller than every float32 value. A max or softmax over an all-masked row then returns a misleading finite value.
- **Dividing by `T`.** The mean must divide by the number of kept steps in each row, not the padded length `T`.
- **Not guarding against a count of `0`.** Computing `0 / 0` produces `NaN`. Clamp the count to at least `1` rather than adding a small epsilon, which slightly shifts every non-empty row.
- **Unaligned counts.** Dividing `(B, D)` by `(B,)` fails for most shapes and silently divides feature `d` by row `d`'s count when `B == D`.
- **Using a bare `squeeze()`.** Reducing with `keepdim=True` and calling `squeeze()` drops the batch axis when `B == 1`. Reduce without `keepdim`, or pass the axis explicitly to `squeeze`.

## Complexity

Each function makes one pass over its input in $O(B\,V)$ time for `fill_masked` and $O(B\,T\,D)$ time for `masked_mean` and `masked_max`. `where` allocates one selected copy of the input, and no other large intermediates are stored.

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

Three differences stand out. First, MLX cannot read through a boolean index, so portable code selects with `where` and preserves shapes. That same pattern is also required under `jax.jit`, where the length of `x[mask]` would be data-dependent. Second, only JAX provides reductions that accept a `where` mask directly. Third, multiplying by a boolean mask in JAX returns `0` on a masked-out `inf` when the mask matches `x`'s shape, but returns `NaN` when a `(B, T, 1)` mask broadcasts against `(B, T, D)`. With a float mask, all three frameworks return `NaN`. Never rely on the same-shape JAX special case.
