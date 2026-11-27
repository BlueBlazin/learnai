## The idea

`cat` and `stack` differ in one thing: whether the axis they join along exists
before the call. Count the axes of the pieces and of the result, and the choice is
made.

**`pad_rows`** is both, one after the other. Padding a row is a join along the axis
the row has: `pad` values and tokens, in the order the side asks for. That gives `B`
tensors of shape `(T_max,)`. Turning them into `(B, T_max)` adds an axis, so it is a
`stack`.

The mask is a statement about positions, and the lengths determine it: on the right
side row `i` is real at positions `t < n_i`, on the left at `t >= T_max - n_i`. Both
are a comparison of `arange(T_max)` with a per-row number. Comparing `ids` with
`pad` looks equivalent and is not, because many models pad with a token that also
occurs in text (their end-of-sequence token).

**`shift_right`** joins a `(B, 1)` column of `bos` with the `(B, T - 1)` block
`ids[:, :-1]` along axis 1, which both have: a `cat`. The slice `ids[:, :1]` keeps
the column axis, where `ids[:, 0]` would drop it and leave nothing to `cat` along.

**`stack_steps`** makes `(B, T, V)` from pieces of shape `(B, V)`: a new axis at
position 1.

## Reference solution (PyTorch)

```python
def pad_rows(rows, pad, side):
    lengths = torch.tensor([row.shape[0] for row in rows])
    width = int(lengths.max())
    padded = []
    for row in rows:
        gap = width - row.shape[0]
        padded.append(F.pad(row, (gap, 0) if side == "left" else (0, gap), value=pad))
    ids = torch.stack(padded)
    position = torch.arange(width)[None, :]
    if side == "left":
        mask = position >= (width - lengths)[:, None]
    else:
        mask = position < lengths[:, None]
    return ids, mask


def shift_right(ids, bos):
    first = torch.full_like(ids[:, :1], bos)
    return torch.cat([first, ids[:, :-1]], dim=1)


def stack_steps(steps):
    return torch.stack(steps, dim=1)
```

The JAX and MLX versions use `pad(row, (gap, 0), constant_values=pad)`, `stack`, and
`concatenate(..., axis=1)`.

Other correct forms: allocate `torch.full((B, T_max), pad)` and assign each row into
a slice of it (in JAX with `.at[i, :n].set(row)`);
`torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=pad,
padding_side=side)` for `ids`; `F.pad(ids, (1, -1), value=bos)` for the shift, since
a negative amount crops in PyTorch; and `torch.roll(ids, 1, dims=1)` followed by
writing `bos` into column 0.

## Common mistakes

- **The sides exchanged.** `F.pad(row, (0, gap))` pads after the tokens. Equal-length
  rows cannot show it, and neither can the longest row of any batch.
- **The mask of the padding.** Some libraries use true for "ignore this position"
  (`key_padding_mask` in `torch.nn.MultiheadAttention`). Here true means a real
  token.
- **`mask = ids != pad`.** False at every real token that has the pad id.
- **An integer mask.** `mask.to(torch.int64)` holds the right numbers, and `~mask`
  on it is `-1` and `-2`. The specification asks for booleans.
- **The shift by `roll`.** `torch.roll(ids, 1, dims=1)` puts the last token of each
  row in column 0. Without `dims`, PyTorch (and JAX and MLX without `axis`) rolls the
  flattened tensor, so column 0 of row `b` receives the last token of row `b - 1`.
- **Forgetting to drop the last column.** `cat([first, ids], dim=1)` is
  `(B, T + 1)`.
- **Shifting left.** Dropping the first column and appending at the end is the
  target sequence of a training step, not the input.
- **`stack` on the default axis.** `torch.stack(steps)` is `(T, B, V)`. It has the
  right shape when `B == T`, with rows and steps exchanged.
- **`stack` on the last axis.** `dim=-1` gives `(B, V, T)`, right in shape when
  `T == V`.
- **`cat` where a new axis is needed.** `torch.cat(steps, dim=1)` is `(B, T * V)`:
  the steps of a row laid end to end. A `reshape(B, T, V)` after it happens to be
  right, and after `cat(steps, dim=0)` it is not.
- **A bare `squeeze()` on the result.** It removes the batch axis of a batch of one
  and the step axis of a single step.

## Complexity

Every function copies each output entry once: $O(B\,T_{max})$, $O(B\,T)$, and
$O(B\,T\,V)$. A decode loop that calls `cat` on a growing tensor at every step
copies the whole history each time, $O(T^2)$ in total, which is why
`stack_steps` takes the list and joins once.

## In the three frameworks

| | PyTorch | JAX | MLX |
|---|---|---|---|
| join along an existing axis | `torch.cat(xs, dim=d)` | `jnp.concatenate(xs, axis=d)` | `mx.concatenate(xs, axis=d)` |
| join along a new axis | `torch.stack(xs, dim=d)` | `jnp.stack(xs, axis=d)` | `mx.stack(xs, axis=d)` |
| default axis of both | 0 | 0 | 0 |
| `cat` of tensors with different numbers of axes | `RuntimeError` | `TypeError` | `ValueError` |
| `stack` of different shapes | `RuntimeError` | `ValueError` | `ValueError` |
| `cat` of integer and float32 tensors | float32 | float32 | float32 |
| constant padding | `F.pad(x, pairs, value=v)`, pairs from the last axis backward | `jnp.pad(x, pairs, constant_values=v)`, one pair per axis in order | `mx.pad(x, pairs, constant_values=v)`, one pair per axis in order |
| one pair `(1, 2)` on a `(2, 3)` tensor | pads the last axis: `(2, 6)` | pads every axis: `(5, 6)` | pads every axis: `(5, 6)` |
| a negative pad amount | crops | `ValueError` | `ValueError` |
| `roll` without an axis | rolls the flattened tensor | the same | the same |
| `split(x, 2, axis)` on 5 columns | pieces of size 2: `2, 2, 1` | 2 equal pieces: `ValueError` | 2 equal pieces: `ValueError` |
| `split(x, [2], axis)` | pieces of the listed sizes: `RuntimeError` unless they sum to 5 | cuts at index 2: `2, 3` | cuts at index 2: `2, 3` |

Every entry was run on PyTorch 2.15, JAX 0.12, and MLX 0.32.4.

Two rows change results without raising. A single pair given to `pad` means the last
axis in PyTorch and every axis in JAX and MLX, so `(1, 2)` ported unchanged pads rows
as well as columns. And the integer argument of `split` is a piece size in PyTorch
and a number of pieces in JAX and MLX: `split(x, 2)` on six columns gives three
pieces in one and two in the others. `cat` promotes silently in all three, so a
`bos` column built as a float tensor turns the whole batch of ids into float32.
