## The idea

Broadcasting follows a single rule. Align the two shapes along their right edges. In each column, the two sizes must be equal, or one of them must be 1 (missing positions on the left count as 1). The output takes the larger size in each column, repeating any size-1 axis to match.

Because the rule only compares sizes and ignores axis semantics, your code must place each axis in the same column as its partner. Indexing with `None` inserts a size-1 axis at that position without copying data.

- **`relative_positions`.** In `out[i, j] = i - j`, `i` varies along axis 0 and `j` varies along axis 1. Reshaping the positions into a column `(T, 1)` and a row `(1, T)` lets their difference broadcast to `(T, T)`.
- **`scale_rows`.** Lining up `(B, T, D)` with `(B,)` places `B` under `D`. Reshaping `s` to `(B, 1, 1)` places `B` under `B`, and the two size-1 axes repeat across `T` and `D`.
- **`pairwise_sq_dist`.** Pairing every point of `a` with every point of `b` requires separate axes for `N` and `M`. Reshaping `a` to `(N, 1, D)` and `b` to `(1, M, D)` yields a difference tensor of shape `(N, M, D)`. Summing the squared differences over the last axis produces `(N, M)`.

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

The JAX and MLX versions are identical except for using `jnp.arange` or `mx.arange` and `sum(axis=-1)`.

`pairwise_sq_dist` above allocates an `(N, M, D)` intermediate tensor before summing. The expanded identity $\lVert a_n \rVert^2 + \lVert b_m \rVert^2 - 2\, a_n \cdot b_m$ avoids that intermediate by using a matrix multiplication and broadcasting a column `(N, 1)` with a row `(1, M)`:

```python
(a * a).sum(-1)[:, None] + (b * b).sum(-1)[None, :] - 2 * a @ b.T
```

Because this form subtracts numbers of similar magnitude, two nearly identical points can produce a small negative value instead of zero. Clamp at zero before taking a square root.

## Common mistakes

- **Computing `j - i` instead of `i - j`.** `pos[None, :] - pos[:, None]` has the right shape but flips the sign off the diagonal. Spot-check one entry by hand, such as verifying that `out[2, 0]` equals `2`.
- **Multiplying `x * s` without adding axes.** `(B,)` right-aligns with `D`. For most shapes this raises an error, but when `B == D` it succeeds silently and scales feature `d` by `s[d]`.
- **Placing the scale factor on the wrong axis.** `s[None, :, None]` has shape `(1, B, 1)` and aligns with `T`. It runs silently when `B == T`, and when `T == 1` it broadcasts to `(B, B, D)`.
- **Subtracting `a - b` directly.** Without new axes, subtraction pairs point `n` of `a` with point `n` of `b`. It fails unless `N == M` (or one of them is 1), in which case it returns `N` distances instead of an `(N, M)` table.
- **Transposing the distance table.** `a[None, :, :] - b[:, None, :]` produces shape `(M, N)`. When `N == M`, the shape matches `(N, M)` even though the table is transposed.
- **Summing over the wrong axis.** Calling `sum(dim=1)` on the `(N, M, D)` differences sums over `b`'s points instead of the feature axis. The result has shape `(N, D)`, which looks like `(N, M)` when `M == D`.

## Complexity

`relative_positions` takes $O(T^2)$ time and memory. `scale_rows` takes $O(B\,T\,D)$ time and memory. `pairwise_sq_dist` takes $O(N\,M\,D)$ time, with $O(N\,M\,D)$ memory in the direct form or $O(N\,M)$ memory in the expanded form.

## In the three frameworks

| | PyTorch | JAX | MLX |
|---|---|---|---|
| new axis by indexing | `v[:, None]` | `v[:, None]` | `v[:, None]` |
| new axis by function | `v.unsqueeze(1)` | `jnp.expand_dims(v, 1)` | `mx.expand_dims(v, 1)` |
| `arange(T)` dtype | int64 | int32 | int32 |
| explicit broadcast | `torch.broadcast_to(v, shape)`, `v.expand(shape)` | `jnp.broadcast_to(v, shape)` | `mx.broadcast_to(v, shape)` |
| shapes that do not broadcast | `RuntimeError` | `TypeError` for `(3,) + (4,)`, `ValueError` for `(2, 3, 4) * (2,)` and in `broadcast_to` | `ValueError` |
| built-in pairwise distance | `torch.cdist(a, b)` (not squared) | none in `jax.numpy` | none |

The broadcasting rule and `None` indexing work the same way across all three frameworks. Only the helper names and error types differ. PyTorch provides `unsqueeze` and `expand`, while JAX and MLX use `expand_dims`. JAX raises two different exception types for incompatible shapes, so avoid catching a specific one. `torch.cdist` returns Euclidean distances rather than squared distances, so squaring its output loses a small amount of precision near zero because of the internal square root.
