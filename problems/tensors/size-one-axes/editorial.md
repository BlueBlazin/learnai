## The idea

Each indexing or reduction operation follows a fixed rule for keeping or removing axes, independent of the input sizes:

- An integer index removes its axis, while a slice keeps its axis.
- A reduction removes its axis unless `keepdim`/`keepdims` is enabled.
- `squeeze` with an explicit axis argument removes only that axis.

Code built from these rules always returns the same number of axes. The exception is `squeeze()` with no arguments, which inspects the runtime sizes and drops every axis that happens to equal 1.

- **`normalize_over`.** Keeping the reduced axis gives the sum size 1 along `dim` and the original size of `x` along every other axis. When you divide `x` by this sum, broadcasting expands only along `dim`. Without keeping the axis, the sum loses a dimension and right-aligns against the wrong axes of `x`.
- **`last_step`.** `x[:, -1]` indexes axis 1 with an integer, removing axis 1 and returning `(B, D)` regardless of whether `B` or `D` is 1.
- **`column`.** `x[:, j : j + 1]` slices axis 1, preserving it with size 1 to return `(B, 1)`.

## Reference solution (PyTorch)

```python
def normalize_over(x, dim):
    return x / x.sum(dim=dim, keepdim=True)


def last_step(x):
    return x[:, -1]


def column(x, j):
    return x[:, j : j + 1]
```

The JAX and MLX versions use `axis=dim, keepdims=True` in `normalize_over`. The two indexing functions are identical.

## Common mistakes

- **Omitting `keepdim`.** `x / x.sum(dim)` raises a shape error for most inputs. On a square `(n, n)` matrix summed over its last axis, however, the `(n,)` row sums right-align with the columns, so entry `[i, j]` is silently divided by the sum of row `j`.
- **Calling bare `squeeze()`.** `x[:, -1:, :].squeeze()` works when `B > 1` and `D > 1`. When `B == 1` it returns `(D,)`, when `D == 1` it returns `(B,)`, and when both are 1 it returns a 0D scalar. Always specify the axis, as in `squeeze(1)`.
- **Keeping the time axis in `last_step`.** `x[:, -1:]` returns `(B, 1, D)` instead of `(B, D)`. This often slips by because `(B, 1, D)` still broadcasts with `(B, D)` tensors, silently producing a `(B, B, D)` result.
- **Indexing the last feature instead of the last step.** `x[:, :, -1]` or `x[..., -1]` returns shape `(B, T)`, which matches `(B, D)` whenever `T == D`.
- **Using `x[:, j]` for `column`.** Integer indexing removes the second axis and returns shape `(B,)`. Combining a `(B,)` tensor with a `(B, 1)` tensor later broadcasts to `(B, B)`.
- **Selecting row `j` instead of column `j`.** `x[j][:, None]` produces a column of the wrong slice and has the expected shape whenever `B == V`.
- **Re-inserting the reduced axis by position.** Manually inserting `None` at index `dim` works for `dim >= 0`, but places the axis in the wrong position when `dim` is negative. Using `keepdim`/`keepdims` handles negative axes automatically.

## Complexity

`normalize_over` reads and writes every element once. `last_step` and `column` touch $B\,D$ and $B$ elements respectively (and in PyTorch they return views without copying memory).

## In the three frameworks

| | PyTorch | JAX | MLX |
|---|---|---|---|
| keep a reduced axis | `x.sum(dim=d, keepdim=True)` | `x.sum(axis=d, keepdims=True)` | `x.sum(axis=d, keepdims=True)` |
| also accepted | `axis=`, `keepdims=` | no `dim=`, no `keepdim=` | no `dim=`, no `keepdim=` |
| remove one axis | `x.squeeze(1)` | `x.squeeze(1)` | `x.squeeze(1)` |
| `squeeze(1)` when axis 1 is not of size 1 | returns `x` unchanged | `ValueError` | `ValueError` |
| remove every size-1 axis | `x.squeeze()` | `x.squeeze()` | `x.squeeze()` |
| add an axis | `x.unsqueeze(1)`, `x[:, None]` | `jnp.expand_dims(x, 1)`, `x[:, None]` | `mx.expand_dims(x, 1)`, `x[:, None]` |
| integer index, slice, `[j]` list | removes, keeps, keeps | the same | the same |

Two framework differences are worth noting. First, PyTorch accepts the NumPy names `axis=` and `keepdims=`, whereas JAX and MLX reject `dim=` and `keepdim=`. Second, calling `squeeze(1)` on an axis whose size is not 1 raises a `ValueError` in JAX and MLX, but silently returns `x` unchanged in PyTorch, delaying any shape error until a later operation.
