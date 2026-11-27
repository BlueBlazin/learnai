Write the three joins around a batched decode loop: pad prompts of different lengths
into one rectangle with its mask, shift a batch of ids one position to the right, and
collect the outputs of `T` steps into one tensor. The questions are always the same
two: is the axis you are joining along already there, and at which end does the new
material go.

## Specification

```python
def pad_rows(rows, pad, side): ...
def shift_right(ids, bos): ...
def stack_steps(steps): ...
```

**`pad_rows(rows, pad, side)`** takes a Python list of `B` one-dimensional integer
tensors, row `i` of length $n_i \ge 1$, and returns the tuple `(ids, mask)`. With
$T_{max} = \max_i n_i$:

- `ids` has shape `(B, T_max)`. Row `i` holds the $n_i$ tokens of `rows[i]`, in order
  and next to each other, and $T_{max} - n_i$ copies of `pad`. With `side="left"`
  the padding comes first and the tokens are the last $n_i$ entries; with
  `side="right"` the tokens come first.
- `mask` is boolean, shape `(B, T_max)`: true at the positions of `ids` that hold a
  token of the row, false at the padding.

The mask describes positions. A token whose id equals `pad` is still a real token,
with mask true.

**`shift_right(ids, bos)`** takes `ids` of shape `(B, T)` and returns shape `(B, T)`:
every row moved one position to the right, `bos` in the first column, and the last
column dropped.

$$
\text{out}[b, 0] = \text{bos}, \qquad \text{out}[b, t] = \text{ids}[b, t - 1] \quad (t \ge 1) .
$$

**`stack_steps(steps)`** takes a Python list of `T` tensors, each of shape `(B, V)`,
and returns them along a new step axis, shape `(B, T, V)`:

$$
\text{out}[b, t, v] = \text{steps}[t][b, v] .
$$

- The tensors in `rows`, and `ids`: integer tensors of your framework's default
  integer type (int64 in PyTorch, int32 in JAX and MLX). `pad` and `bos`: Python
  ints. `side`: the string `"left"` or `"right"`; no other value is passed.
- `ids` from `pad_rows` and the result of `shift_right` are integer tensors; any
  integer width is accepted. `mask` is a boolean tensor.
- The tensors in `steps` are float32, and `stack_steps` returns float32. The list has
  at least one tensor.

## Shapes

| name | shape | dtype |
|---|---|---|
| `rows[i]` | `(n_i,)`, a list of `B` | integer |
| `ids` from `pad_rows` | `(B, T_max)` | integer |
| `mask` from `pad_rows` | `(B, T_max)` | bool |
| `ids` in `shift_right` | `(B, T)` | integer |
| result of `shift_right` | `(B, T)` | integer |
| `steps[t]` | `(B, V)`, a list of `T` | float32 |
| result of `stack_steps` | `(B, T, V)` | float32 |

## Tools

- **`cat`** (`torch.cat(tensors, dim=d)`, `jnp.concatenate(arrays, axis=d)`,
  `mx.concatenate`) joins along an axis the inputs **already have**. The result has
  the same number of axes, the sizes along `d` add up, and all other sizes must
  match.
- **`stack`** (`torch.stack`, `jnp.stack`, `mx.stack`) joins along a **new** axis,
  inserted at position `d`. The inputs must have one shape, and the result has one
  more axis than they do. The default position is 0.
- **Padding** is a `cat` with a block of fill values, and each framework also has it
  as one call. `torch.nn.functional.pad(x, (before, after), value=v)` lists its
  pairs starting from the **last** axis. `jnp.pad(x, pad_width, constant_values=v)`
  and `mx.pad` take one `(before, after)` pair per axis, first axis first, and a
  single pair for a one-dimensional tensor.
- **`roll`** (`torch.roll(x, shifts, dims)`, `jnp.roll(x, shift, axis)`, `mx.roll`)
  moves entries along an axis.

> [!WARNING]
> `roll` is a rotation: what leaves one end comes back at the other. Nothing is
> dropped and nothing new comes in.

`pad_rows` receives a list, so a Python loop over its rows is fine there.

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

With left padding the last column, `7, 8, 10`, is the last real token of every row.

## Constraints

- Results are compared with the reference exactly: values are copied.
- $1 \le B \le 8$. Rows have between 1 and 12 tokens. `T` and `V` are between 1 and
  8. Any size can be 1, and any two can be equal.
- Token ids, `pad`, and `bos` lie in $[0, 1000)$.
- `shift_right` and `stack_steps` need no Python loop.

## Why this matters

Padding a Batch for Generation is `pad_rows` on Python lists, and explains why a
decoder wants the padding on the left; Gathering the Last Real Token then reads a
different position depending on the side. Sequence Log-Likelihood and Perplexity
pairs the logits at position `t` with the token at `t + 1`, the same shift written as
two slices. Attention with an Appended Cache grows its keys and values with one `cat`
per step along the sequence axis, and Drafting with Multi-Token Prediction Heads
returns its draft as a `stack` of per-head tokens. A `roll` for the shift gives the
model its own last token as the first input; a `stack` on the default axis returns
`(T, B, V)`, which is the right shape whenever the test has as many steps as rows.
