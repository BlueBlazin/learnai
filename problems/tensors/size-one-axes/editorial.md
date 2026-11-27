## The idea

Every operation here either keeps an axis or removes it, and the rule for which is
fixed by the operation, never by the data:

- an integer index removes its axis, and a slice keeps it;
- a reduction removes its axis unless told to keep it;
- `squeeze` with an axis removes that axis.

Code written from these rules returns the same number of axes for every input. The one
operation that breaks the pattern is `squeeze()` with no argument, which looks at the
sizes: it removes whatever happens to be 1 in this call.

**`normalize_over`.** The sum over `dim` must be divided into `x` slice by slice. With
the axis kept, the sum has size 1 along `dim` and the size of `x` everywhere else, so
it broadcasts back along `dim` and nowhere else. Without it, the sum has lost an axis
and is lined up from the right against the wrong axes of `x`.

**`last_step`.** `x[:, -1]` indexes axis 1 with an integer, which removes axis 1 and
leaves `(B, D)` whatever `B` and `D` are.

**`column`.** `x[:, j : j + 1]` slices axis 1, which keeps it with size 1.

## Reference solution (PyTorch)

```python
def normalize_over(x, dim):
    return x / x.sum(dim=dim, keepdim=True)


def last_step(x):
    return x[:, -1]


def column(x, j):
    return x[:, j : j + 1]
```

The JAX and MLX versions use `axis=dim, keepdims=True`; the two indexing lines are the
same.

## Common mistakes

- **No `keepdim`.** `x / x.sum(dim)` raises for most shapes. On a square matrix summed
  over its last axis it runs: the row sums, shape `(n,)`, line up with the columns, and
  entry `[i, j]` is divided by the sum of row `j`.
- **A bare `squeeze()`.** `x[:, -1:, :].squeeze()` is right whenever `B > 1` and
  `D > 1`. With one row it returns `(D,)`, with one feature `(B,)`, and with both a
  tensor with no axes. Name the axis: `squeeze(1)`.
- **Keeping the step axis.** `x[:, -1:]` is `(B, 1, D)`. It often goes unnoticed
  because it still broadcasts against `(B, D)` tensors, to a `(B, B, D)` result.
- **The last feature for the last step.** `x[:, :, -1]` or `x[..., -1]` is `(B, T)`,
  which has the shape of `(B, D)` when `T == D`.
- **`x[:, j]` for a column.** The integer removes the axis and the result is `(B,)`.
  Against a `(B, 1)` tensor a `(B,)` tensor broadcasts to `(B, B)`.
- **Row `j` for column `j`.** `x[j][:, None]` is a column, of the wrong thing; it has
  the right shape when `B == V`.
- **Putting the axis back by position.** Rebuilding the summed axis by hand, for
  example by indexing the sum with `dim` colons and a `None`, works for `dim >= 0` and
  puts the axis in front for a negative `dim`. `keepdim` has no such case.

## Complexity

`normalize_over` reads and writes every element once. `last_step` and `column` copy
$B\,D$ and $B$ values (PyTorch returns views and copies nothing).

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

Two differences matter. PyTorch accepts the NumPy spellings `axis=` and `keepdims=`,
but JAX and MLX reject `dim=` and `keepdim=`, so code moves from them to PyTorch and
not back. And `squeeze(1)` on an axis that is not of size 1 is an error in JAX and MLX
but does nothing in PyTorch, where a wrong axis number passes silently and the shape
error shows up later.
