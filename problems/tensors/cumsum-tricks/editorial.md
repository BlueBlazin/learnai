## The idea

A loop that scans a row maintains a running variable and branches on its value. A cumulative sum records the entire history of that variable as a tensor, turning the branch into an elementwise comparison.

**`exclusive_cumsum`.** `cumsum` is inclusive, so entry `i` is the total after adding `x[i]`. The total before adding `x[i]` is the previous inclusive sum. Shifting the inclusive sums one position to the right — prepending a zero and dropping the last entry — gives the exclusive sum.

**`first_true`.** Consider the running count of `True` entries for `[False, True, False, True, False]`:

| position | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| `mask` | F | T | F | T | F |
| `cumsum(mask)` | 0 | 1 | 1 | 2 | 2 |
| `cumsum == 0` | T | F | F | F | F |

The running count is `0` at every position strictly before the first `True` entry, and at least `1` from that entry onward. Summing `cumsum == 0` along the row counts those leading positions, which equals the index of the first `True`. When a row has no `True` entry, the running count stays `0` everywhere and the sum returns `T` with no special case.

**`through_first_eos`.** Position `t` is kept when no `eos` token appears strictly before it, meaning the exclusive count of `eos` tokens at position `t` is `0`. The first `eos` has zero `eos` tokens before it, so it stays `True`, while every later position has at least one and becomes `False`.

## Reference solution (PyTorch)

```python
def exclusive_cumsum(x):
    total = torch.cumsum(x, dim=-1)
    zero = torch.zeros_like(x[..., :1])
    return torch.cat([zero, total[..., :-1]], dim=-1)


def first_true(mask):
    seen = torch.cumsum(mask, dim=1)  # (B, T) int64
    return (seen == 0).sum(dim=1)


def through_first_eos(ids, eos):
    is_eos = (ids == eos).to(torch.int64)
    return exclusive_cumsum(is_eos) == 0
```

The JAX version is identical using `jnp.concatenate` and `axis=`. MLX provides the exclusive sum directly as `mx.cumsum(x, axis=-1, inclusive=False)`.

Another common way to write `exclusive_cumsum` is `cumsum(x) - x`, which nucleus (top-p) filtering often uses. It is exact for integers, but in float32 it subtracts after rounding. For `x = [0.1, 1e8, 0.3]`, `cumsum(x) - x` returns `[0, 0, 1e8]`, whereas shifting returns `[0, 0.1, 1e8]`. It also produces `NaN` (`inf - inf`) whenever `x` contains `inf`.

You can also compose `through_first_eos` from `first_true`: `arange(T)[None, :] <= first_true(ids == eos)[:, None]`.

## Common mistakes

- **Returning the inclusive sum.** `cumsum` alone includes the current entry. In a top-p filter, that off-by-one error drops the token that crosses the threshold.
- **Shifting with `roll`.** `roll` wraps the end of the row around to the front, putting the full row total at position `0` instead of `0`.
- **Using `argmax` for `first_true`.** `argmax` works when a row has a `True` entry, but returns `0` when a row has none, which is indistinguishable from a match at position `0`.
- **Counting all false entries.** `(~mask).sum()` also counts `False` entries that appear after the first `True`.
- **Calling bare `squeeze()` on the output.** Summing with `keepdim=True` and then calling `squeeze()` drops the batch axis when `B == 1`, returning a scalar. Sum without `keepdim` so only the sequence axis is removed.
- **Dropping the first `eos`.** `cumsum(is_eos) == 0` is `False` at the first `eos` itself. A position should be kept when no `eos` came strictly *before* it.
- **Using `ids != eos`.** This is `False` at every `eos` and `True` everywhere else, including tokens after the first `eos`.
- **Using `cumsum(is_eos) <= 1`.** This keeps the first `eos` and also keeps every token after it up to the second `eos`.
- **Summing along the wrong axis.** Running `cumsum` along axis `0` accumulates across the batch while preserving the tensor shape.

## Complexity

Each function makes one pass along the axis, taking $O(B\,T)$ time and memory (or $O(\text{numel}(x))$ for `exclusive_cumsum`).

## In the three frameworks

| | PyTorch | JAX | MLX |
|---|---|---|---|
| running sum | `torch.cumsum(x, dim=d)` | `jnp.cumsum(x, axis=d)` | `mx.cumsum(x, axis=d)` |
| no axis given | `TypeError` | flattens first | flattens first |
| exclusive form | none | `jnp.cumulative_sum(x, axis=d, include_initial=True)`, then drop the last entry | `mx.cumsum(x, axis=d, inclusive=False)` |
| from the end | flip, `cumsum`, flip | flip, `cumsum`, flip; `jax.lax.cumsum(x, reverse=True)` | `mx.cumsum(x, axis=d, reverse=True)` |
| `cumsum` of a boolean tensor | int64 | int32 | int32 |
| integer count minus a boolean tensor | `NotImplementedError` (cast the mask first) | int32 | int32 |
| `argmax` of a boolean tensor | `NotImplementedError` (cast first) | int32 indices | uint32 indices |
| `argmax` of a row of equal values | 0 | 0 | 0 |
| running max | `torch.cummax(m, dim=d).values` (supports bool) | `jax.lax.cummax(m, axis=d)` (`TypeError` for bool) | `mx.cummax(m, axis=d)` (supports bool) |
| `cumprod` of a boolean tensor | int64 | int32 | bool |

Every entry above was verified on PyTorch 2.15, JAX 0.12, and MLX 0.32.4.

MLX is the only framework of the three with built-in flags for exclusive and reversed cumulative sums. PyTorch is the strictest about boolean arithmetic. `cumsum` accepts a boolean mask, but `argmax` and subtraction do not, so PyTorch code casts masks to integers explicitly where JAX and MLX do not need to. In all three frameworks, the cumulative sum of a boolean tensor is an integer tensor of the default width, which is what the comparisons in this problem require.
