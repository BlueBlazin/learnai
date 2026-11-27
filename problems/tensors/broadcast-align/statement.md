Write three small functions that are each built on one broadcast. The arithmetic is a
subtraction, a multiplication, and a subtraction followed by a sum of squares; the work
is putting every axis where the broadcasting rule will pair it with the right partner.

## Specification

```python
def relative_positions(T): ...
def scale_rows(x, s): ...
def pairwise_sq_dist(a, b): ...
```

**`relative_positions(T)`** returns the integer table of shape `(T, T)` with

$$
\text{out}[i, j] = i - j .
$$

**`scale_rows(x, s)`** multiplies each row of a batch by that row's own factor. `x` has
shape `(B, T, D)` and `s` has shape `(B,)`:

$$
\text{out}[b, t, d] = x[b, t, d] \cdot s[b] .
$$

**`pairwise_sq_dist(a, b)`** takes `N` points `a` of shape `(N, D)` and `M` points `b` of
shape `(M, D)` and returns the `(N, M)` table of squared distances between every point
of `a` and every point of `b`:

$$
\text{out}[n, m] = \sum_{d} \big(a[n, d] - b[m, d]\big)^2 .
$$

- `T`: a Python int, $T \ge 1$.
- `x`, `s`, `a`, `b`: float32 tensors.
- `relative_positions` returns a tensor of your framework's default integer type (what
  `arange` gives). The other two return float32 tensors.

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

Two tensors broadcast when their shapes, **lined up from the right**, agree in every
position or have a 1 there; a shape that is too short is padded with 1s on the left.
An axis of size 1 is then repeated to the size of its partner.

To put an axis of size 1 where you need it, index with `None`: for `v` of shape `(n,)`,
`v[:, None]` has shape `(n, 1)` and `v[None, :]` has shape `(1, n)`. This works in
PyTorch, JAX, and MLX alike.

![The shape of `s` is lined up with the shape of `x` from the right, so without new axes its only axis meets `D`.](assets/align.svg)

> [!WARNING]
> Broadcasting never asks which axis you meant. When two axes happen to have the same
> size, a tensor lined up with the wrong one is multiplied without any error.

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

In the last table, row 1 column 2 is the squared distance from `(3, 4)` to `(0, 4)`:
$3^2 + 0^2 = 9$.

## Constraints

- `relative_positions` is compared exactly. `scale_rows` and `pairwise_sq_dist` are
  compared with the reference to a relative tolerance of `1e-4` and an absolute
  tolerance of `1e-4`, so the expanded form
  $\lVert a \rVert^2 + \lVert b \rVert^2 - 2\,a \cdot b$ of the distance passes too.
- $1 \le T, B, D, N, M \le 16$. Entries of `a` and `b` have magnitude at most 10, and
  entries of `x` and `s` have magnitude at most 1000.
- Any of the sizes can be 1, and any two of them can be equal.
- No Python loops are needed.

## Why this matters

These three lines come back all through the catalog. Sliding-Window Attention builds
its mask from a column of query positions and a row of key positions, which is
`relative_positions` followed by a comparison. One Step of the Gated Delta Rule and
Batching Requests Across Adapters both multiply a batch by one factor per row with
`[:, None, None]`. An Inverted-File Index assigns every vector to its nearest centroid
from the table that `pairwise_sq_dist` returns. In each of them a wrong alignment
raises while the two axes differ in size. On the day they are equal it runs, and
returns a tensor of the right shape with wrong values.
