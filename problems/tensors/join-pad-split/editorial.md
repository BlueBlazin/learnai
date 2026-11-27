## The idea

`cat` and `stack` differ in one way. `cat` joins along an axis that already exists, while `stack` creates a new axis. Comparing the number of axes in the inputs and the output tells you which one you need.

**`pad_rows`** uses both operations in sequence. Padding a single row joins `pad` values and tokens along the row's existing 1D axis, giving `B` tensors of shape `(T_max,)`. Combining those `B` rows into a `(B, T_max)` batch adds a batch axis, so it uses `stack`.

The mask depends only on sequence lengths. For row `i` of length $n_i$, right padding places real tokens at positions `t < n_i`, and left padding places them at positions `t >= T_max - n_i`. Both compare `arange(T_max)` against a per-row threshold. Checking `ids != pad` looks equivalent, but it breaks whenever a model uses a token that also appears in text (such as the end-of-sequence token) as its pad ID.

**`shift_right`** joins a `(B, 1)` column filled with `bos` and the `(B, T - 1)` slice `ids[:, :-1]` along axis 1. Both pieces already have that column axis, so this is a `cat`. Indexing `ids[:, :1]` keeps the 2D shape `(B, 1)`, whereas `ids[:, 0]` would drop axis 1 and leave a 1D tensor that cannot be concatenated along axis 1.

**`stack_steps`** turns `T` tensors of shape `(B, V)` into one tensor of shape `(B, T, V)`. That inserts a new axis at position 1, so it is a `stack` with axis 1.

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

The JAX and MLX versions use `pad(row, (gap, 0), constant_values=pad)`, `stack`, and `concatenate(..., axis=1)`.

Several other implementations are also valid. You can allocate a `(B, T_max)` tensor filled with `pad` and write each row into its slice (using `.at[i, :n].set(row)` in JAX). In PyTorch, `torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=pad, padding_side=side)` builds `ids` directly, and `F.pad(ids, (1, -1), value=bos)` shifts right because negative pad amounts crop in PyTorch. You can also call `torch.roll(ids, 1, dims=1)` and then overwrite column 0 with `bos`.

## Common mistakes

- **Swapping left and right padding.** Calling `F.pad(row, (0, gap))` pads after the tokens. Tests with equal-length rows or the longest row in a batch will not catch this.
- **Inverting the mask.** Some APIs (such as `key_padding_mask` in `torch.nn.MultiheadAttention`) use `True` for padding positions to ignore. Here `True` marks a real token.
- **Computing `mask = ids != pad`.** This marks real tokens as `False` whenever their token ID equals `pad`.
- **Returning an integer mask.** Converting the mask to `int64` keeps `0` and `1`, but bitwise negation `~mask` produces `-1` and `-2` instead of flipping booleans. The specification requires a boolean tensor.
- **Shifting with `roll` alone.** `torch.roll(ids, 1, dims=1)` wraps the last token of each row into column 0. Calling `roll` without `dims` (or `axis` in JAX and MLX) flattens the tensor first, so column 0 of row `b` gets the last token of row `b - 1`.
- **Forgetting to drop the last column.** `cat([first, ids], dim=1)` returns shape `(B, T + 1)` instead of `(B, T)`.
- **Shifting left instead of right.** Dropping column 0 and appending at the end builds a training target sequence, not a right-shifted decoder input.
- **Stacking on the default axis.** `torch.stack(steps)` produces shape `(T, B, V)`. When `B == T`, the shape still matches, but batch rows and steps are transposed.
- **Stacking on the last axis.** Using `dim=-1` produces `(B, V, T)`, which has the right shape whenever `T == V`.
- **Using `cat` instead of `stack`.** `torch.cat(steps, dim=1)` produces `(B, T * V)`, laying each row's steps end to end. A subsequent `reshape(B, T, V)` recovers the right layout after `dim=1`, but not after `cat(steps, dim=0)`.
- **Calling bare `squeeze()` on the output.** This removes the batch axis when `B == 1` or the step axis when `T == 1`.

## Complexity

Each function copies every output element once, giving $O(B\,T_{\max})$ for `pad_rows`, $O(B\,T)$ for `shift_right`, and $O(B\,T\,V)$ for `stack_steps`. In a decode loop, calling `cat` on a growing tensor at every step copies the full history each time for $O(T^2)$ total work. Collecting step outputs in a list and calling `stack_steps` once avoids that quadratic cost.

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

Two of these cross-framework differences change results without raising an error. Passing a single pair `(1, 2)` to `pad` pads only the last axis in PyTorch, but pads every axis in JAX and MLX, so porting it unchanged pads rows as well as columns. Also, an integer argument to `split` means chunk size in PyTorch and number of chunks in JAX and MLX, so `split(x, 2)` on six columns returns three chunks in PyTorch and two in JAX and MLX. In addition, `cat` promotes dtypes silently in all three frameworks, so concatenating a float `bos` column with integer `ids` turns the entire batch into `float32`.
