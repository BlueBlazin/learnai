Write three small functions that each decide what happens to one axis: kept with size
1, or removed. The values are easy. The shapes are the exercise, and they have to be
right when a batch, a sequence, or a feature axis happens to have size 1.

## Specification

```python
def normalize_over(x, dim): ...
def last_step(x): ...
def column(x, j): ...
```

**`normalize_over(x, dim)`** divides `x` by its sum over the axis `dim`. The result has
the shape of `x`, and every slice of it along `dim` sums to 1. For a matrix and
`dim = 1`:

$$
\text{out}[i, j] = \frac{x[i, j]}{\sum_{k} x[i, k]} .
$$

`x` has one to four axes, and `dim` may be negative (`-1` is the last axis). Every
entry of `x` is positive, so no sum is zero.

**`last_step(x)`** takes `x` of shape `(B, T, D)` and returns the last step of every
row, shape `(B, D)`:

$$
\text{out}[b, d] = x[b, T - 1, d] .
$$

**`column(x, j)`** takes `x` of shape `(B, V)` and a Python int `j` with $0 \le j < V$,
and returns column `j` as a column, shape `(B, 1)`:

$$
\text{out}[b, 0] = x[b, j] .
$$

The result shapes are exactly these for every input. `last_step` returns two axes when
`B = 1` or `D = 1`, and `column` returns two axes when `B = 1`.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` of `normalize_over` | `(d_1, ..., d_k)`, $1 \le k \le 4$ | float32 |
| result of `normalize_over` | `(d_1, ..., d_k)` | float32 |
| `x` of `last_step` | `(B, T, D)` | float32 |
| result of `last_step` | `(B, D)` | float32 |
| `x` of `column` | `(B, V)` | float32 |
| result of `column` | `(B, 1)` | float32 |

## Tools

- A reduction (`sum`, `mean`, `max`) removes the axis it reduces. With `keepdim=True`
  (PyTorch) or `keepdims=True` (JAX, MLX) the axis stays, with size 1.
- An integer index removes the axis it indexes. A slice keeps its axis, even when the
  slice has length 1.
- `squeeze(dim)` (PyTorch) or `squeeze(axis)` (JAX, MLX) removes one axis of size 1,
  the one you name.

> [!WARNING]
> `squeeze()` with no argument removes **every** axis of size 1, including a batch axis
> that is 1 only because this batch holds one row. Code that uses it passes every test
> with `B > 1` and returns the wrong shape for a single request.

## Example

```python
x = [[1.0, 1.0, 2.0], [3.0, 4.0, 1.0]]  # shape (2, 3)
normalize_over(x, 1)  # row sums are 4 and 8
# [[0.25,  0.25, 0.5  ],
#  [0.375, 0.5,  0.125]]
normalize_over(x, 0)  # column sums are 4, 5, and 3
# [[0.25, 0.2, 0.6667],
#  [0.75, 0.8, 0.3333]]

x = [
    [[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]],
    [[100.0, 101.0], [110.0, 111.0], [120.0, 121.0]],
]  # shape (2, 3, 2)
last_step(x)  # [[20.0, 21.0], [120.0, 121.0]], shape (2, 2)

y = [[[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]]]  # shape (1, 2, 4)
last_step(y)  # [[10.0, 11.0, 12.0, 13.0]], shape (1, 4)

x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]  # shape (2, 3)
column(x, 1)  # [[2.0], [5.0]], shape (2, 1)
```

`last_step(y)` has shape `(1, 4)`, not `(4,)`: a batch of one row is still a batch.

## Constraints

- `normalize_over` is compared with the reference to a relative tolerance of `1e-4` and
  an absolute tolerance of `1e-6`. `last_step` and `column` copy values and are compared
  exactly.
- Every size is between 1 and 16, and entries of the input of `normalize_over` lie in
  $[0.5, 4]$.
- $-k \le \text{dim} < k$ for an input with $k$ axes.
- No Python loops are needed.

## Why this matters

RMSNorm divides by a mean that has to keep its axis, exactly as `normalize_over` does,
and so does every softmax and every renormalization after a top-k or top-p filter.
Unembedding Only What You Need starts from `last_step`. Gathering the Last Real Token
asks for a `(B, V)` result, and a solution that ends in a bare `squeeze()` passes
every case there but the one with a single row or a single vocabulary entry. That
is how this bug behaves in a server: batched load tests pass, and the first request
that arrives alone, a batch of one, either fails with a shape error two calls later
or is broadcast against something it should not be.
