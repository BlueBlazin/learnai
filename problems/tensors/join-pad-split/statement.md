Implement three tensor-joining functions used in batched autoregressive decoding. You will pad variable-length sequences into a rectangular batch with a boolean mask, shift token IDs right by one position, and stack per-step outputs into a single tensor.

## Specification

```python
def pad_rows(rows, pad, side): ...
def shift_right(ids, bos): ...
def stack_steps(steps): ...
```

**`pad_rows(rows, pad, side)`** takes a Python list `rows` of `B` one-dimensional integer tensors, where row `i` has length $n_i \ge 1$. Let $T_{\max} = \max_i n_i$. The function also takes a Python integer `pad` and a string `side` (`"left"` or `"right"`), and returns a tuple `(ids, mask)`.

- `ids` is an integer tensor of shape `(B, T_max)`. Row `i` contains the $n_i$ tokens of `rows[i]` in order, together with $T_{\max} - n_i$ copies of `pad`. When `side == "left"`, the padding comes first and the tokens occupy the last $n_i$ positions. When `side == "right"`, the tokens come first and the padding follows.
- `mask` is a boolean tensor of shape `(B, T_max)`. Each entry is `True` where `ids` holds a token from `rows[i]` and `False` at padding positions.

The mask tracks positions, not values. A real token whose ID happens to equal `pad` still has a `True` mask entry.

**`shift_right(ids, bos)`** takes an integer tensor `ids` of shape `(B, T)` and a Python integer `bos`. It returns an integer tensor of shape `(B, T)` where every row shifts right by one position, `bos` fills column 0, and the last column is dropped:

$$
\text{out}[b, 0] = \text{bos}, \qquad \text{out}[b, t] = \text{ids}[b, t - 1] \quad (t \ge 1).
$$

**`stack_steps(steps)`** takes a non-empty Python list `steps` of `T` float32 tensors, each of shape `(B, V)`. It joins them along a new step axis to return a float32 tensor of shape `(B, T, V)`:

$$
\text{out}[b, t, v] = \text{steps}[t][b, v].
$$

Input integer tensors (`rows[i]` and `ids`) use your framework's default integer type (`int64` in PyTorch, `int32` in JAX and MLX). The integer outputs of `pad_rows` and `shift_right` may have any integer width.

## Shapes

| name | shape | dtype |
|---|---|---|
| `rows[i]` | `(n_i,)`, list of `B` | integer |
| `ids` from `pad_rows` | `(B, T_max)` | integer |
| `mask` from `pad_rows` | `(B, T_max)` | bool |
| `ids` in `shift_right` | `(B, T)` | integer |
| result of `shift_right` | `(B, T)` | integer |
| `steps[t]` | `(B, V)`, list of `T` | float32 |
| result of `stack_steps` | `(B, T, V)` | float32 |

## Tools

- **`cat`** (`torch.cat(tensors, dim=d)`, `jnp.concatenate(arrays, axis=d)`, `mx.concatenate`) joins tensors along an axis they **already have**. The result keeps the same number of axes, the sizes along axis `d` add up, and all other axis sizes must match.
- **`stack`** (`torch.stack`, `jnp.stack`, `mx.stack`) joins tensors along a **new** axis inserted at position `d` (default `0`). Every input must have the same shape, and the output has one more axis than the inputs.
- **Padding** can be written as `cat` with a fill tensor, or with a single pad call. `torch.nn.functional.pad(x, (before, after), value=v)` lists pairs starting from the **last** axis. `jnp.pad(x, pad_width, constant_values=v)` and `mx.pad` take one `(before, after)` pair per axis starting from axis 0, or a single pair for a 1D tensor.
- **`roll`** (`torch.roll(x, shifts, dims)`, `jnp.roll(x, shift, axis)`, `mx.roll`) shifts entries along an axis.

> [!WARNING]
> `roll` is a circular rotation. Elements pushed past one end wrap around to the other end, so nothing is dropped and no new value is inserted.

Because `pad_rows` receives a Python list of variable-length tensors, looping over `rows` in Python is fine there.

## Example

```python
rows = [[5, 6, 7], [8], [9, 10]]  # three tensors, lengths 3, 1, 2
ids, mask = pad_rows(rows, pad=0, side="left")
ids   # [[5, 6, 7], [0, 0, 8], [0, 9, 10]]
mask  # [[True, True, True], [False, False, True], [False, True, True]]

ids, mask = pad_rows(rows, pad=0, side="right")
ids   # [[5, 6, 7], [8, 0, 0], [9, 10, 0]]
mask  # [[True, True, True], [True, False, False], [True, True, False]]

shift_right([[1, 2, 3, 4], [5, 6, 7, 8]], bos=99)
# [[99, 1, 2, 3],
#  [99, 5, 6, 7]]                    # 4 and 8 are gone

steps = [
    [[0.0, 1.0], [10.0, 11.0]],      # step 0, shape (2, 2): rows 0 and 1
    [[100.0, 101.0], [110.0, 111.0]],  # step 1
    [[200.0, 201.0], [210.0, 211.0]],  # step 2
]
stack_steps(steps)
# [[[  0.0,   1.0], [100.0, 101.0], [200.0, 201.0]],   # row 0: its three steps
#  [[ 10.0,  11.0], [110.0, 111.0], [210.0, 211.0]]]   # row 1; shape (2, 3, 2)
```

With left padding, the final column (`7, 8, 10`) holds the last real token of every row.

## Constraints

- Outputs are compared to the reference for exact equality.
- $1 \le B \le 8$, with each row having length $1 \le n_i \le 12$.
- $1 \le T \le 8$ and $1 \le V \le 8$. Any dimension may be 1, and any two dimensions may be equal.
- Token IDs, `pad`, and `bos` lie in $[0, 1000)$.
- `shift_right` and `stack_steps` need no Python loops.

## Why this matters

Batched generation relies on all three patterns. Decoders left-pad prompts so every sequence's last real token sits in the final column, while training pairs logits at position `t` with target tokens at `t + 1` using the same right shift. KV caches grow along the sequence axis with one `cat` per step, and multi-token prediction heads collect per-head outputs with `stack`.

Small axis or boundary mistakes fail silently. Using `roll` for `shift_right` feeds each sequence's last token back in as its first input. Calling `stack` on the default axis returns `(T, B, V)` instead of `(B, T, V)`, which still has the expected shape whenever a test uses `B == T`.
