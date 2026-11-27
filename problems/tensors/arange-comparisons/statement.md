Implement three functions that build boolean masks from position indices without reading any data. Each function broadcasts two index vectors against each other and compares them element by element.

## Specification

```python
def length_mask(lengths, T, side): ...
def band_mask(T, w): ...
def same_segment(seg): ...
```

**`length_mask(lengths, T, side)`** builds a padding mask for a batch of `B` sequences padded to a common length `T`, where `lengths` is a 1D integer tensor of length `B`. Row `b` contains `lengths[b]` real tokens and `T - lengths[b]` padding tokens, and `side` specifies which end holds the padding. The function returns a `(B, T)` boolean mask that is true at real tokens and false at padding:

- `side == "right"` (real tokens first, padding at the end): `out[b, j]` is true when $j < \text{lengths}[b]$.
- `side == "left"` (padding first, real tokens at the end): `out[b, j]` is true when $j \ge T - \text{lengths}[b]$.

Row `b` has exactly `lengths[b]` true entries. A row of length `0` is all false, and a row of length `T` is all true. No other value of `side` is passed.

**`band_mask(T, w)`** builds a `(T, T)` boolean causal sliding-window mask of width `w`. Row `i` can attend to column `j` when `j` is at or before `i` and fewer than `w` positions behind `i`:

$$
\text{out}[i, j] = \big(0 \le i - j < w\big) .
$$

Row `i` has $\min(i + 1, w)$ true entries, ending on the main diagonal `j = i`. The window width `w` may be larger than `T`.

**`same_segment(seg)`** takes a 1D tensor of `T` segment IDs and returns a `(T, T)` boolean mask that is true where positions `i` and `j` share the same segment ID:

$$
\text{out}[i, j] = \big(\text{seg}[i] = \text{seg}[j]\big) .
$$

Segment IDs are arbitrary non-negative integers. Equal IDs do not need to be adjacent, and IDs do not need to start at `0`.

`lengths` and `seg` use your framework's default integer dtype (`int64` in PyTorch, `int32` in JAX and MLX). `T` and `w` are positive Python ints, and `side` is `"left"` or `"right"`. Every function must return a **boolean** tensor, not an integer or float tensor of zeros and ones.

## Shapes

| name | shape | dtype |
|---|---|---|
| `lengths` | `(B,)` | integer |
| result of `length_mask` | `(B, T)` | bool |
| result of `band_mask` | `(T, T)` | bool |
| `seg` | `(T,)` | integer |
| result of `same_segment` | `(T, T)` | bool |

## Tools

- `arange(n)` (`torch.arange`, `jnp.arange`, `mx.arange`) builds the 1D tensor $0, 1, \dots, n - 1$ in the default integer dtype.
- Elementwise comparisons (`<`, `<=`, `>=`, `==`) broadcast two tensors just like arithmetic and return a boolean tensor. Comparing a `(n, 1)` column against a `(1, m)` row produces an `(n, m)` table indexed by `(row, column)`, using `None` to place the size-1 axes as in Broadcasting on Purpose.
- Combine boolean tensors element by element with `&` (and), `|` (or), and `~` (not).

> [!WARNING]
> In Python, `&` and `|` bind more tightly than comparison operators, so `a >= 0 & a < w` parses as `a >= (0 & a) < w`. Always wrap each comparison in parentheses. Chained comparisons (`0 <= a < w`) and the keywords `and`, `or`, and `not` also fail on multi-element tensors because they ask the tensor for a single truth value.

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

- Results must be boolean tensors and are checked for exact equality against the reference.
- $1 \le B \le 16$, $1 \le T \le 16$, $0 \le \text{lengths}[b] \le T$, and $1 \le w \le 32$.
- Segment IDs lie in $[0, 1000]$.
- No Python loops are needed.

## Why this matters

Causal and Padding Masks combines a real-token mask with a causal mask using the same two meanings of `side`, and Gathering the Last Real Token locates the final token from those same lengths. Sliding-Window Attention hides every position outside `band_mask`. Frequency and Presence Penalties separates prompt tokens from generated tokens by comparing `arange(T)` against a `(B, 1)` column of lengths.

Confusing `<` with `<=` shifts any of these masks by one position. An off-by-one mask lets a model attend to a padding token or one extra past token, changing outputs silently without raising an error.
