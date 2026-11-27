Write three functions that build a boolean mask from positions. None of them reads
any data: each mask is a comparison between index vectors that broadcast against each
other, and the exercise is choosing the two vectors, their axes, and `<` or `<=`.

## Specification

```python
def length_mask(lengths, T, side): ...
def band_mask(T, w): ...
def same_segment(seg): ...
```

**`length_mask(lengths, T, side)`** describes a batch of `B` sequences padded to a
common length `T`. Row `b` holds `lengths[b]` real tokens and `T - lengths[b]`
padding tokens, and `side` says where the padding is. The function returns the
boolean mask of real tokens, shape `(B, T)`:

- `side == "right"` (real tokens first, padding after): `out[b, j]` is true when
  $j < \text{lengths}[b]$;
- `side == "left"` (padding first, real tokens last): `out[b, j]` is true when
  $j \ge T - \text{lengths}[b]$.

Row `b` of the result has exactly `lengths[b]` true entries. A row of length 0 is all
false, and a row of length `T` is all true. No other value of `side` is passed.

**`band_mask(T, w)`** returns the boolean `(T, T)` mask of a causal window of width
`w`: row `i` may see column `j` when `j` is not after `i` and is fewer than `w`
positions back,

$$
\text{out}[i, j] = \big(0 \le i - j < w\big) .
$$

Row `i` therefore has $\min(i + 1,\ w)$ true entries, ending at the diagonal. `w`
may be larger than `T`.

**`same_segment(seg)`** takes segment ids of shape `(T,)` and returns the boolean
`(T, T)` mask of pairs of positions in the same segment:

$$
\text{out}[i, j] = \big(\text{seg}[i] = \text{seg}[j]\big) .
$$

Segment ids are arbitrary non-negative integers. Equal ids need not be adjacent, and
ids need not start at 0.

- `lengths`, `seg`: integer tensors of your framework's default integer type (int64 in
  PyTorch, int32 in JAX and MLX).
- `T`, `w`: positive Python ints. `side`: the string `"left"` or `"right"`.
- Every function returns a **boolean** tensor. A float or integer tensor of zeros and
  ones is not accepted.

## Shapes

| name | shape | dtype |
|---|---|---|
| `lengths` | `(B,)` | integer |
| result of `length_mask` | `(B, T)` | bool |
| result of `band_mask` | `(T, T)` | bool |
| `seg` | `(T,)` | integer |
| result of `same_segment` | `(T, T)` | bool |

## Tools

- `arange(n)` (`torch.arange`, `jnp.arange`, `mx.arange`) is the vector of positions
  $0, 1, \dots, n - 1$, in the default integer type.
- A comparison (`<`, `<=`, `>=`, `==`) between two tensors broadcasts them exactly as
  arithmetic does, and returns a boolean tensor of the broadcast shape. A table
  indexed by `(row, column)` is a comparison between a `(n, 1)` column and a `(1, m)`
  row, with `None` placing the axes as in Broadcasting on Purpose.
- Boolean tensors combine entry by entry with `&` (and), `|` (or), and `~` (not).

> [!WARNING]
> In Python, `&` and `|` bind more tightly than comparisons: `a >= 0 & a < w` means
> `a >= (0 & a) < w`. Put each comparison in parentheses. The chained form
> `0 <= a < w` and the keywords `and`, `or`, and `not` do not work on tensors either:
> they ask the tensor for a single truth value, and that raises for a tensor of more
> than one element.

## Example

```python
length_mask([3, 1, 4], T=4, side="right")
# [[ True,  True,  True, False],
#  [ True, False, False, False],
#  [ True,  True,  True,  True]]             # shape (3, 4)

length_mask([3, 1, 4], T=4, side="left")
# [[False,  True,  True,  True],
#  [False, False, False,  True],
#  [ True,  True,  True,  True]]

band_mask(T=4, w=2)
# [[ True, False, False, False],             # row 0 sees column 0
#  [ True,  True, False, False],             # row 1 sees columns 0 and 1
#  [False,  True,  True, False],             # row 2 sees columns 1 and 2
#  [False, False,  True,  True]]

same_segment([7, 7, 2, 7])
# [[ True,  True, False,  True],
#  [ True,  True, False,  True],
#  [False, False,  True, False],
#  [ True,  True, False,  True]]
```

## Constraints

- Results are compared with the reference exactly, and must be boolean.
- $1 \le B \le 16$, $1 \le T \le 16$, $0 \le \text{lengths}[b] \le T$, and
  $1 \le w \le 32$.
- Segment ids lie in $[0, 1000]$.
- No Python loops are needed.

## Why this matters

Causal and Padding Masks combines a mask of real tokens with a causal mask, with the
same two meanings of `side`, and Gathering the Last Real Token finds its position
from the same lengths. Sliding-Window Attention hides everything outside
`band_mask`. Frequency and Presence Penalties separates the prompt from the generated
tokens of each row with one comparison of `arange(T)` against a `(B, 1)` column of
lengths. Each of these is off by one position if `<` and `<=` are confused, and a mask
that is off by one lets a model read one padding token or one token too far back,
which changes the output without raising anything.
