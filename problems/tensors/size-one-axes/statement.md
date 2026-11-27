Implement three tensor functions (`normalize_over`, `last_step`, and `column`) that control whether a size-1 axis is preserved or removed. Each function must return the exact target shape even when a batch, sequence, or feature axis happens to have size 1.

## Specification

```python
def normalize_over(x, dim): ...
def last_step(x): ...
def column(x, j): ...
```

**`normalize_over(x, dim)`** divides `x` by its sum along axis `dim`. The input `x` has between 1 and 4 axes and strictly positive entries, so no sum is zero. The integer `dim` may be negative, where `-1` refers to the last axis. The output has the same shape as `x`, and every slice along `dim` sums to 1. For a 2D matrix with `dim = 1`:

$$
\text{out}[i, j] = \frac{x[i, j]}{\sum_{k} x[i, k]} .
$$

**`last_step(x)`** takes a 3D tensor `x` of shape `(B, T, D)` (batch size `B`, sequence length `T`, feature size `D`) and returns the final time step of each batch item with shape `(B, D)`:

$$
\text{out}[b, d] = x[b, T - 1, d] .
$$

**`column(x, j)`** takes a 2D tensor `x` of shape `(B, V)` and a Python integer `j` ($0 \le j < V$), and returns column `j` as a 2D column tensor of shape `(B, 1)`:

$$
\text{out}[b, 0] = x[b, j] .
$$

These output shapes apply to every input. `last_step` always returns a 2D tensor even when `B = 1` or `D = 1`, and `column` always returns a 2D tensor even when `B = 1`.

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

- A reduction (`sum`, `mean`, `max`) removes the reduced axis by default. Passing `keepdim=True` (PyTorch) or `keepdims=True` (JAX, MLX) keeps that axis with size 1.
- Indexing with an integer removes that axis. Indexing with a slice keeps the axis, even if the slice has length 1.
- Calling `squeeze(dim)` (PyTorch) or `squeeze(axis)` (JAX, MLX) removes only the named size-1 axis.

> [!WARNING]
> Calling `squeeze()` with no argument removes **every** axis of size 1, including a batch axis that happens to have size 1 for a single input. Code using bare `squeeze()` passes every test where `B > 1` and breaks on a batch of size 1.

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

`last_step(y)` has shape `(1, 4)` rather than `(4,)` because a batch of one item is still a batch.

## Constraints

- `normalize_over` is compared against the reference with a relative tolerance of `1e-4` and an absolute tolerance of `1e-6`. `last_step` and `column` copy values and are compared exactly.
- Every axis size is between 1 and 16. Entries of the input to `normalize_over` lie in $[0.5, 4]$.
- $-k \le \text{dim} < k$ for an input with $k$ axes.
- No Python loops are needed.

## Why this matters

RMSNorm, softmax, and top-k/top-p renormalization all divide by a reduction whose axis must be kept, just like `normalize_over`. Unembedding Only What You Need starts with `last_step`. Gathering the Last Real Token expects a `(B, V)` output, where a bare `squeeze()` passes every case except a single-item batch or a size-1 vocabulary.

In production inference servers, this bug passes batched load tests and fails only when a lone request arrives as a batch of size 1. At that point, the dropped batch axis either triggers a shape error downstream or broadcasts silently against the wrong dimension.
