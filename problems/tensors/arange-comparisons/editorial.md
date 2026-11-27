## The idea

A position mask answers a yes-or-no question about indices for every cell in a 2D table. Write the condition for a single entry, then place each index vector along its own axis so broadcasting evaluates the whole table at once.

**`length_mask`** asks whether position `j` is a real token in row `b`. Put the positions `arange(T)` along axis 1 with shape `(1, T)` and `lengths` along axis 0 with shape `(B, 1)`. With right padding, the real tokens are the first `lengths[b]` positions, so `j < lengths[b]`. With left padding, they are the last `lengths[b]` positions starting at `T - lengths[b]`, so `j >= T - lengths[b]`. Checking the boundary cases confirms `<` versus `<=`. A row of length `0` must be all false, and a row of length `T` must be all true.

**`band_mask`** asks how far column `j` lies behind row `i`. Subtracting a `(1, T)` row `j` from a `(T, 1)` column `i` produces the `(T, T)` distance table `i - j`. Entries are `0` on the diagonal, positive below it, and negative above it. The causal window keeps distances $0, 1, \dots, w - 1$, which is `w` positions including the diagonal.

**`same_segment`** compares `seg[i]` with `seg[j]` by broadcasting `seg` as a `(T, 1)` column against `seg` as a `(1, T)` row. This checks equality directly and makes no assumptions about segment ordering or numbering.

## Reference solution (PyTorch)

```python
def length_mask(lengths, T, side):
    pos = torch.arange(T)[None, :]  # (1, T)
    n = lengths[:, None]  # (B, 1)
    if side == "right":
        return pos < n
    return pos >= T - n


def band_mask(T, w):
    i = torch.arange(T)[:, None]  # (T, 1)
    j = torch.arange(T)[None, :]  # (1, T)
    back = i - j
    return (back >= 0) & (back < w)


def same_segment(seg):
    return seg[:, None] == seg[None, :]
```

The JAX and MLX solutions differ only in the name of `arange`.

You can also construct the band from triangular masks by starting with a boolean lower triangle and clearing everything below diagonal `-(w - 1)` with `ones.tril().triu(-(w - 1))`. Keep those tensors boolean, because subtracting float triangles produces a float tensor.

## Common mistakes

- **Using `<=` instead of `<`.** `pos <= n` marks `n + 1` positions instead of `n`, because positions $0, \dots, n - 1$ already contain `n` entries.
- **Swapping `"left"` and `"right"`.** `side` names where the padding sits, not where the real tokens sit. `"right"` keeps the first `n` positions.
- **Calling a bare `squeeze()` on the mask.** This drops the batch axis when `B == 1` or the position axis when `T == 1`. Comparing `(1, T)` with `(B, 1)` already produces the exact `(B, T)` shape.
- **Taking the complement for left padding.** `pos >= n` has `T - n` true entries instead of `n`, so it only matches the right answer when `n == T / 2`.
- **Reversing the axes.** `arange(T)[:, None] < lengths[None, :]` has shape `(T, B)`. That bug is invisible in the shape whenever `B == T`.
- **Checking only half of the band.** `i - j < w` alone is true for every column past the diagonal, where `i - j` is negative.
- **Making the window one position too wide.** `i - j <= w` keeps distances $0, \dots, w$, which is `w + 1` positions. The diagonal counts as one of the `w`.
- **Reaching forward instead of backward.** Using `j - i` instead of `i - j` produces the transpose. Decide which index is the row before writing the subtraction.
- **Omitting parentheses around comparisons.** `back >= 0 & back < w` parses as `back >= (0 & back) < w`, which is a chained comparison and raises an error.
- **Comparing segment IDs to positions.** `seg[:, None] == arange(T)[None, :]` tests whether the ID at position `i` equals the index `j`. Both sides must be `seg`.
- **Assuming segments are contiguous runs.** Comparing run indices or adjacent elements treats two separated stretches with the same ID as different segments.

## Complexity

`length_mask` computes $B\,T$ boolean entries, and `band_mask` and `same_segment` compute $T^2$ entries, each in a single vectorized comparison. The 1D index vectors take $O(T)$ space.

## In the three frameworks

| | PyTorch | JAX | MLX |
|---|---|---|---|
| positions | `torch.arange(T)`, int64 | `jnp.arange(T)`, int32 | `mx.arange(T)`, int32 |
| a comparison returns | `torch.bool` | `bool` | `mx.bool_` |
| and, or, not | `&`, `\|`, `~`; `torch.logical_and` | `&`, `\|`, `~`; `jnp.logical_and` | `&`, `\|`, `~`; `mx.logical_and` |
| `a >= 0 & a < w` | `RuntimeError` | `ValueError` | `TypeError` |
| `0 <= a < w`, `(a >= 0) and (a < w)`, more than one element | `RuntimeError` | `ValueError` | `ValueError` |
| triangle of a boolean matrix | `m.tril()`, `m.triu(-1)`; stays bool | `jnp.tril(m)`, `jnp.triu(m, -1)`; stays bool | `mx.tril(m)`, `mx.triu(m, -1)`; stays bool |
| `arange` of a 0-d tensor | accepted | accepted | `TypeError`: pass a Python int |
| `mask + mask`, `mask * mask` | bool (or, and) | bool (or, and) | bool (or, and) |

Every entry was run on PyTorch 2.15, JAX 0.12, and MLX 0.32.4.

The code is identical across all three frameworks. Only the default integer dtype of `arange` and the exception type for the precedence bug differ. Integer width can matter when comparing positions against lengths from another source, though PyTorch compares `int64` positions with `int32` lengths cleanly and still returns a boolean tensor. In MLX, `mx.arange` requires a Python number, so a scalar array length needs `.item()` first.
