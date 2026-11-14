Implement the two shape changes that turn a single attention layer into a multi-head
one: `split_heads`, which cuts the feature axis into heads and moves the head axis next
to the batch axis, and `merge_heads`, its inverse. Each is two operations, a reshape
and a transpose, and the whole exercise is that these two are not interchangeable.

## Specification

```python
def split_heads(x, n_heads): ...
def merge_heads(x): ...
```

A model of width `D` with `H` heads gives every head a contiguous block of
$d = D / H$ features: head 0 owns features $0, \dots, d - 1$, head 1 owns features
$d, \dots, 2d - 1$, and so on.

**`split_heads(x, n_heads)`** takes `x` of shape `(B, T, D)` and returns `y` of shape
`(B, H, T, d)` with

$$
y[b, h, t, i] = x[b, t, h \cdot d + i] .
$$

So `y[b, h]` is the sequence as head `h` sees it: `T` vectors of width `d`.

**`merge_heads(x)`** takes `x` of shape `(B, H, T, d)` and returns `y` of shape
`(B, T, H * d)` with

$$
y[b, t, h \cdot d + i] = x[b, h, t, i] .
$$

It concatenates, for every token, the vectors of head 0, head 1, ..., head `H - 1`.

- `x`: a float32 tensor. Its values are arbitrary.
- `n_heads`: `H`, a positive Python int that divides `D`.
- Both functions return float32 tensors that hold the same values as the input, moved
  and not changed. Whether the result shares memory with the input does not matter.

The two functions are inverses of each other: `merge_heads(split_heads(x, H))` equals
`x`, and `split_heads(merge_heads(y), H)` equals `y`.

Both functions must work on any tensor of the right shape, however it was produced and
however it is laid out in memory: a freshly created tensor, or the output of a
transpose (as when `merge_heads` is applied to what `split_heads` returned).

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` of `split_heads` | `(B, T, D)` | float32 |
| result of `split_heads` | `(B, H, T, D / H)` | float32 |
| `x` of `merge_heads` | `(B, H, T, d)` | float32 |
| result of `merge_heads` | `(B, T, H * d)` | float32 |

## Example

One sequence of three tokens of width 4, split into two heads of width 2:

```python
x = [[[0.0, 1.0,  2.0,  3.0],    # token 0
      [4.0, 5.0,  6.0,  7.0],    # token 1
      [8.0, 9.0, 10.0, 11.0]]]   # token 2        shape (1, 3, 4)

split_heads(x, 2)
# [[[[0.0, 1.0], [4.0, 5.0], [ 8.0,  9.0]],      head 0: features 0-1 of each token
#   [[2.0, 3.0], [6.0, 7.0], [10.0, 11.0]]]]     head 1: features 2-3 of each token
#                                                 shape (1, 2, 3, 2)
```

`merge_heads` of that result returns `x`.

Compare with reshaping `x` straight to `(1, 2, 3, 2)`, which has the right shape:

```python
# [[[[0.0, 1.0], [2.0, 3.0], [ 4.0,  5.0]],
#   [[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]]]]
```

Here "head 0" holds all of token 0 and half of token 1. A reshape never moves a value
in memory order; it only redraws the boundaries. Putting the head axis in front of the
sequence axis changes the order, and that takes a transpose.

## Constraints

- Results are compared with the reference exactly: no arithmetic is involved.
- $1 \le B, T \le 16$, $1 \le H \le 16$, and $1 \le D / H \le 16$. `D` is always a
  multiple of `H`; `H = 1` and `H = D` both occur.
- No Python loops over batch items, tokens, or heads are needed.

## Why this matters

Multi-head attention is single-head attention run on `H` slices in parallel, and the
slicing is done entirely by these two functions: split after the query, key, and value
projections, merge before the output projection. A checkpoint's projection matrices
assume the head-major layout above, so the feature order is part of the weights'
contract. Every bug here produces a tensor of the right shape. With a reshape in place
of the transpose, each "head" attends over a mixture of positions and features, nothing
fails, and the model produces fluent-looking nonsense, or worse, slightly degraded text
when the bug is confined to one path such as a cached decode step. Shape errors in this
step are the most common reason a hand-written forward pass does not match the
reference implementation.
