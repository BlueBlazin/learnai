## The idea

A mask over positions is a yes-or-no question about indices, and a comparison answers
it for a whole table at once. Write the condition for a single entry, then let
broadcasting supply every entry: each index that the condition mentions becomes a
vector along its own axis.

**`length_mask`.** Entry `[b, j]` asks whether position `j` is one of the real tokens
of row `b`. The position is `arange(T)` along axis 1, shape `(1, T)`. The length is
`lengths` along axis 0, shape `(B, 1)`. With right padding the real tokens are the
first `lengths[b]` positions, `j < lengths[b]`. With left padding they are the last
`lengths[b]`, which start at `T - lengths[b]`, so `j >= T - lengths[b]`. Checking the
ends settles `<` against `<=`: a row of length 0 must be all false and a row of
length `T` all true.

**`band_mask`.** Entry `[i, j]` asks how far column `j` is behind row `i`. The table
`i - j`, a `(T, 1)` column minus a `(1, T)` row, holds that distance: 0 on the
diagonal, positive below it, negative above. The window keeps distances
$0, 1, \dots, w - 1$, which are `w` positions including the diagonal.

**`same_segment`.** Entry `[i, j]` compares `seg[i]` with `seg[j]`: the same vector
as a column and as a row. Nothing is assumed about the ids.

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

The JAX and MLX versions differ only in the name of `arange`.

The band can also be cut from triangles: a boolean lower triangle with the part
below diagonal `-(w - 1)` removed, `ones.tril().triu(-(w - 1))`. Keep the triangles
boolean; a difference of float triangles is a float tensor.

## Common mistakes

- **`<=` for `<`.** `pos <= n` marks `n + 1` positions. Count: positions
  $0, \dots, n - 1$ are `n` of them.
- **The sides swapped.** `side` names where the padding is, not where the tokens
  are: `"right"` keeps the first `n` positions.
- **A bare `squeeze()` on the mask.** It removes the batch axis of a single row and
  the position axis when `T == 1`. A comparison of `(1, T)` with `(B, 1)` already has
  the right two axes.
- **The complement for the other side.** `pos >= n` is not the left-padded mask. It
  has `T - n` true entries, and it agrees with the right answer only when
  `n == T / 2`.
- **Axes the wrong way round.** `arange(T)[:, None] < lengths[None, :]` is `(T, B)`.
  The mistake is invisible in the shape when `B == T`.
- **Half of the band.** `i - j < w` alone is true for every column after the
  diagonal, where `i - j` is negative.
- **A window one too wide.** `i - j <= w` keeps distances $0, \dots, w$, which are
  `w + 1` positions. The diagonal counts as one of the `w`.
- **The band reaching forward.** `j - i` in place of `i - j` gives the transpose.
  Decide which index is the row before writing the subtraction.
- **Missing parentheses.** `back >= 0 & back < w` is `back >= (0 & back) < w`, a
  chained comparison, and raises.
- **Ids compared with positions.** `seg[:, None] == arange(T)[None, :]` asks whether
  the id of position `i` equals the number `j`. Both sides of the comparison are
  `seg`.
- **Assuming segments are runs.** Comparing run numbers, or neighbors, treats two
  separate stretches with the same id as different segments.

## Complexity

`length_mask` builds $B\,T$ entries and the other two $T^2$, one comparison each.
The index vectors are $O(T)$.

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

The code is the same in all three; what differs is the dtype of `arange` and what
the precedence mistake raises. The integer width matters when a mask is compared
with lengths that came from somewhere else: PyTorch compares int64 positions with
int32 lengths without complaint, and the result is boolean either way. In MLX the
size given to `arange` has to be a Python number, so a length that is still an array
needs `.item()` first.
