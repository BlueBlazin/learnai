Implement three tensor functions that rely on broadcasting: `relative_positions`, `scale_rows`, and `pairwise_sq_dist`. Each function requires aligning tensor axes so that elementwise operations pair the intended dimensions.

## Specification

```python
def relative_positions(T): ...
def scale_rows(x, s): ...
def pairwise_sq_dist(a, b): ...
```

**`relative_positions(T)`** takes a Python integer $T \ge 1$ and returns an integer tensor of shape `(T, T)` containing pairwise index differences for $0 \le i, j < T$:

$$
\text{out}[i, j] = i - j .
$$

The output uses your framework's default integer dtype (the dtype returned by `arange`).

**`scale_rows(x, s)`** scales each item in a batch by its own factor. Given a 3D tensor `x` with batch size `B`, sequence length `T`, and feature size `D`, and a 1D tensor `s` with one scale factor per batch item, it computes:

$$
\text{out}[b, t, d] = x[b, t, d] \cdot s[b] .
$$

**`pairwise_sq_dist(a, b)`** computes the squared Euclidean distance between every pair of points from two sets. Given `N` points in `a` and `M` points in `b`, each in `D` dimensions, it returns an `(N, M)` tensor where:

$$
\text{out}[n, m] = \sum_{d=0}^{D-1} \big(a[n, d] - b[m, d]\big)^2 .
$$

## Shapes

| name | shape | dtype |
|---|---|---|
| result of `relative_positions` | `(T, T)` | integer |
| `x` | `(B, T, D)` | float32 |
| `s` | `(B,)` | float32 |
| result of `scale_rows` | `(B, T, D)` | float32 |
| `a` | `(N, D)` | float32 |
| `b` | `(M, D)` | float32 |
| result of `pairwise_sq_dist` | `(N, M)` | float32 |

## Tools

Two tensors broadcast when their shapes, **lined up from the right**, match at every position or have size 1. If one tensor has fewer axes, its shape is padded with 1s on the left. Each size-1 axis is then repeated to match the size of its partner.

To insert an axis of size 1 at a specific position, index with `None`. For a 1D tensor `v` of shape `(n,)`, `v[:, None]` has shape `(n, 1)` and `v[None, :]` has shape `(1, n)`. This syntax works identically in PyTorch, JAX, and MLX.

![The shape of `s` is lined up with the shape of `x` from the right, so without new axes its only axis meets `D`.](assets/align.svg)

> [!WARNING]
> Broadcasting never checks what an axis means. When two axes happen to have the same size, a tensor aligned with the wrong axis runs without error and produces wrong values.

## Example

```python
relative_positions(3)
# [[ 0, -1, -2],
#  [ 1,  0, -1],
#  [ 2,  1,  0]]

x = [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]  # shape (3, 1, 2)
s = [1.0, 10.0, 100.0]  # shape (3,)
scale_rows(x, s)
# [[[1.0, 2.0]], [[30.0, 40.0]], [[500.0, 600.0]]]  # shape (3, 1, 2)

a = [[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]]  # 3 points in the plane
b = [[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [6.0, 8.0]]  # 4 points
pairwise_sq_dist(a, b)
# [[  0.0,  9.0, 16.0, 100.0],
#  [ 25.0, 16.0,  9.0,  25.0],
#  [100.0, 73.0, 52.0,   0.0]]                      # shape (3, 4)
```

In the `pairwise_sq_dist` output, row 1, column 2 is the squared distance between `(3, 4)` and `(0, 4)`, which equals $(3 - 0)^2 + (4 - 4)^2 = 9$.

## Constraints

- `relative_positions` is compared exactly. `scale_rows` and `pairwise_sq_dist` are compared against the reference with a relative tolerance of `1e-4` and an absolute tolerance of `1e-4`, so the expanded distance formula $\lVert a \rVert^2 + \lVert b \rVert^2 - 2\,a \cdot b$ also passes.
- $1 \le T, B, D, N, M \le 16$. Entries of `a` and `b` have magnitude at most 10, and entries of `x` and `s` have magnitude at most 1000.
- Any dimension size can be 1, and any two sizes can be equal.
- No Python loops are needed.

## Why this matters

These three broadcasting patterns appear throughout machine learning codebases. Sliding-Window Attention builds its mask from a column of query positions and a row of key positions, which is `relative_positions` followed by a comparison. One Step of the Gated Delta Rule and Batching Requests Across Adapters both scale a batch by one factor per item, just like `scale_rows`. An Inverted-File Index assigns each vector to its nearest centroid using the distance table from `pairwise_sq_dist`.

In every case, a misaligned broadcast raises an error only while the mismatched axes have different sizes. When those sizes happen to be equal, the operation succeeds silently and returns a tensor of the expected shape filled with wrong numbers.
