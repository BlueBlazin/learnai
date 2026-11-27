## The idea

Broadcasting has one rule. Write the shapes aligned at the right edge. In every column
the two sizes must be equal, or one of them must be 1 (a missing entry on the left
counts as 1). The result takes the larger size in each column, and a size-1 axis is
repeated to fill it.

The rule looks only at sizes. It does not know that one axis is a batch and another a
feature, so the code has to place each axis in the column of its partner. Indexing with
`None` does that: it inserts an axis of size 1 at that position and copies nothing.

- **`relative_positions`.** `out[i, j] = i - j` has `i` on axis 0 and `j` on axis 1.
  So the positions are needed as a column `(T, 1)`, which varies along axis 0, and as a
  row `(1, T)`, which varies along axis 1. Their difference broadcasts to `(T, T)`.
- **`scale_rows`.** `(B, T, D)` over `(B,)` puts `B` under `D`. Over `(B, 1, 1)` it
  puts `B` under `B`, and the two 1s are repeated along `T` and `D`.
- **`pairwise_sq_dist`.** "Every point of `a` against every point of `b`" is a table
  with an axis for each, so `a` becomes `(N, 1, D)` and `b` becomes `(1, M, D)`. The
  difference is `(N, M, D)`, and summing its squares over the last axis leaves
  `(N, M)`.

## Reference solution (PyTorch)

```python
def relative_positions(T):
    pos = torch.arange(T)
    return pos[:, None] - pos[None, :]


def scale_rows(x, s):
    return x * s[:, None, None]


def pairwise_sq_dist(a, b):
    diff = a[:, None, :] - b[None, :, :]
    return (diff * diff).sum(dim=-1)
```

The JAX and MLX versions are the same lines with `jnp.arange` or `mx.arange` and
`sum(axis=-1)`.

`pairwise_sq_dist` builds an `(N, M, D)` tensor before it sums. The expanded form
$\lVert a_n \rVert^2 + \lVert b_m \rVert^2 - 2\, a_n \cdot b_m$ avoids it with one
matrix product, and is itself a broadcast of a column `(N, 1)` and a row `(1, M)`:

```python
(a * a).sum(-1)[:, None] + (b * b).sum(-1)[None, :] - 2 * a @ b.T
```

It subtracts numbers of similar size, so two nearly equal points can come out as a
small negative number in place of zero. Clamp at zero before taking a square root.

## Common mistakes

- **`j - i`.** `pos[None, :] - pos[:, None]` has the right shape and the wrong sign
  everywhere off the diagonal. Check one entry by hand: `out[2, 0]` must be `2`.
- **`x * s` with no new axes.** `(B,)` lines up with `D`. For most shapes that raises,
  which is the lucky case. When `B == D` it runs and scales feature `d` by `s[d]`.
- **The factor on the wrong axis.** `s[None, :, None]` has shape `(1, B, 1)` and lines
  up with `T`. It runs when `B == T`, and when `T == 1` it returns `(B, B, D)`.
- **Elementwise `a - b`.** With no new axes the subtraction pairs point `n` of `a` with
  point `n` of `b`. It raises unless `N == M` (or one of them is 1), and then returns
  `N` distances and not the table.
- **The table transposed.** `a[None, :, :] - b[:, None, :]` gives `(M, N)`. When
  `N == M` the shape is right and the table is the transpose.
- **Summing the wrong axis.** `sum(dim=1)` on the `(N, M, D)` differences removes the
  points of `b` and keeps the features: shape `(N, D)`, which passes for `(N, M)` when
  `M == D`.

## Complexity

`relative_positions` takes $O(T^2)$ time and memory and `scale_rows` $O(B\,T\,D)$.
`pairwise_sq_dist` takes $O(N\,M\,D)$ time, and $O(N\,M\,D)$ memory in the direct form
or $O(N\,M)$ in the expanded form.

## In the three frameworks

| | PyTorch | JAX | MLX |
|---|---|---|---|
| new axis by indexing | `v[:, None]` | `v[:, None]` | `v[:, None]` |
| new axis by function | `v.unsqueeze(1)` | `jnp.expand_dims(v, 1)` | `mx.expand_dims(v, 1)` |
| `arange(T)` dtype | int64 | int32 | int32 |
| explicit broadcast | `torch.broadcast_to(v, shape)`, `v.expand(shape)` | `jnp.broadcast_to(v, shape)` | `mx.broadcast_to(v, shape)` |
| shapes that do not broadcast | `RuntimeError` | `TypeError` for `(3,) + (4,)`, `ValueError` for `(2, 3, 4) * (2,)` and in `broadcast_to` | `ValueError` |
| built-in pairwise distance | `torch.cdist(a, b)` (not squared) | none in `jax.numpy` | none |

The rule itself is the same in all three, and so is `None` indexing. The differences
are names: only PyTorch has `unsqueeze` and `expand`, and only JAX and MLX have
`expand_dims`. JAX raises two different exception types for shapes that do not
broadcast, so catch neither by name. `torch.cdist` returns distances, so it needs squaring, and the square
root it took on the way loses a little precision near zero.
