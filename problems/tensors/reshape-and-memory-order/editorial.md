## The idea

Think of a tensor as a flat list of values paired with a shape. The list is in row-major order, meaning the last index changes fastest. Two families of operations act on this pair.

- **`reshape` changes the shape and keeps the list.** It can split an axis (`C -> (C / g, g)`) or merge neighboring axes (`(n, 2) -> 2n`), because both operations only move axis boundaries.
- **`transpose` and `permute` change the list.** Reordering the axes changes which index runs fastest, so the row-major order of the elements changes with it.

Every tensor rearrangement follows a three-step recipe. First, reshape to give a separate axis to every index you care about. Second, permute the axes so the ones you want to merge sit next to each other in the right order. Third, reshape to merge them.

**`group_columns`** needs only the first step. Reading `out[r, k, i] = w[r, k * g + i]` visits the elements in their original row-major order.

**`interleave`** needs the flat output order `a0, b0, a1, b1, ...`. Stacking `a` and `b` along a new last axis produces shape `(..., n, 2)`, whose row-major order is already that exact sequence. Merging the last two axes is then a plain reshape.

**`patchify`** needs all three steps. Splitting both spatial axes turns `x` into shape `(B, H / p, p, W / p, p)`, indexed as `x[b, i, u, j, v]` for pixel `(u, v)` of patch `(i, j)`. The output needs `(i, j)` merged and `(u, v)` merged, but `u` sits between `i` and `j`. Permuting to `(b, i, j, u, v)` places both pairs adjacent to each other, and the final reshape merges them.

## Reference solution (PyTorch)

```python
def group_columns(w, g):
    rows, cols = w.shape
    return w.reshape(rows, cols // g, g)


def interleave(a, b):
    pairs = torch.stack((a, b), dim=-1)
    return pairs.reshape(*a.shape[:-1], 2 * a.shape[-1])


def patchify(x, p):
    batch, height, width = x.shape
    rows, cols = height // p, width // p
    x = x.reshape(batch, rows, p, cols, p)
    x = x.permute(0, 1, 3, 2, 4)
    return x.reshape(batch, rows * cols, p * p)
```

In JAX and MLX, `stack` takes `axis=-1` and the permutation is `x.transpose(0, 1, 3, 2, 4)`.

**`view` after a transpose.** In PyTorch, a transpose copies nothing and returns a view whose strides walk the existing memory in a new order. Because `view` can only reinterpret contiguous memory as it lies, calling `x.permute(0, 1, 3, 2, 4).view(...)` raises `"view size is not compatible with input tensor's size and stride"`. `reshape` copies when necessary, and `.contiguous().view(...)` does the same in two steps. JAX and MLX do not make this distinction, so their `reshape` always works.

## Common mistakes

- **Interleaving by concatenation.** `cat((a, b), dim=-1)` produces the right shape, but it puts all of `a` before all of `b`.
- **Stacking on the wrong axis.** `stack((a, b), dim=0).reshape(...)` places every element of `a` before every element of `b` in row-major order, and the reshape cannot undo that. The new axis must be last so each `(a_i, b_i)` pair runs fastest.
- **Patches by reshape alone.** `x.reshape(B, -1, p * p)` groups every `p * p` consecutive pixels in reading order, producing horizontal strips rather than 2D blocks. In the worked example, it returns `x` unchanged.
- **Patches in column-major order.** Permuting to `(b, j, i, u, v)` numbers patches top to bottom before left to right. Square patch grids hide this bug in the output shape.
- **Pixels in column-major order.** Permuting to `(b, i, j, v, u)` transposes every patch.
- **Reading the matrix down its columns.** `w.T.reshape(R, C // g, g)` groups values that came from the same column. A misplaced transpose before a reshape is a common way to get the right shape with the wrong values.
- **Strided groups.** `w.reshape(R, g, C // g).transpose(1, 2)` has the right shape, but group `k` ends up holding columns `k`, `k + C / g`, `k + 2 * C / g`, and so on. It splits each row into `g` groups of size `C / g` and then transposes them.

## Complexity

Each function touches every element at most once. In PyTorch, `group_columns` returns a view without copying, while `interleave` and `patchify` copy the data once.

## In the three frameworks

| | PyTorch | JAX | MLX |
|---|---|---|---|
| reshape | `x.reshape(2, -1)` | `x.reshape(2, -1)` | `x.reshape(2, -1)` |
| reshape without a copy, or an error | `x.view(2, -1)` | none | none |
| swap two axes | `x.transpose(1, 2)`, `x.swapaxes(1, 2)` | `jnp.swapaxes(x, 1, 2)` | `mx.swapaxes(x, 1, 2)` |
| permute all axes | `x.permute(0, 2, 1)` | `x.transpose(0, 2, 1)` | `x.transpose(0, 2, 1)` |
| `x.transpose(1, 2)` on three axes | swaps axes 1 and 2 | `TypeError` | `ValueError` |
| move one axis | `x.movedim(1, 2)`, `torch.moveaxis` | `jnp.moveaxis(x, 1, 2)` | `mx.moveaxis(x, 1, 2)` |
| stack on a new last axis | `torch.stack((a, b), dim=-1)` | `jnp.stack((a, b), axis=-1)` | `mx.stack((a, b), axis=-1)` |
| merge axes | `x.flatten(1, 2)` | `reshape` only | `x.flatten(1, 2)` |
| split one axis | `x.unflatten(1, (2, 3))` | `reshape` only | `mx.unflatten(x, 1, (2, 3))` |

The main trap in this table is `transpose`. In PyTorch, `transpose` takes two axes and swaps them. In JAX and MLX, as in NumPy, it takes a full axis permutation (matching PyTorch's `permute`), so passing two arguments for a 3D tensor raises an error. `swapaxes` behaves the same way in all three frameworks.
