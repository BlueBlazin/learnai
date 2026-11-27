## The idea

Think of a tensor as a flat list of values plus a shape. The list is in row-major
order: the last index runs fastest. Two families of operations act on that pair.

- **`reshape` changes the shape and keeps the list.** It can split an axis
  (`C -> (C / g, g)`) or merge neighboring axes (`(n, 2) -> 2n`), because both only
  move the boundaries.
- **`transpose` and `permute` change the list.** After a permutation of the axes, a
  different index runs fastest, so the row-major order of the values is a different
  one.

Every regrouping is then a recipe of three steps: reshape to give a name to every
index you care about, permute so that the indices you want to merge are neighbors and
in the right order, reshape to merge them.

**`group_columns`** needs only the first step. `out[r, k, i] = w[r, k g + i]` reads the
values in their original order.

**`interleave`** needs the output list `a0, b0, a1, b1, ...`. Stacking `a` and `b` on a
new last axis gives shape `(..., n, 2)`, whose row-major order is exactly that list.
Merging the last two axes is then a plain reshape.

**`patchify`** needs all three steps. Splitting both image axes gives
`x[b, i, u, j, v]`: pixel `(u, v)` of patch `(i, j)`. The output wants `(i, j)` merged
and `(u, v)` merged, but `u` sits between `i` and `j`. Permuting to `(b, i, j, u, v)`
makes both pairs neighbors, and the final reshape merges them.

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

In JAX and MLX, `stack` takes `axis=-1` and the permutation is
`x.transpose(0, 1, 3, 2, 4)`.

**`view` after a transpose.** In PyTorch a transpose copies nothing: it returns a view
whose strides walk the old memory in a new order. `view` can only reinterpret memory
as it lies, so `x.permute(0, 1, 3, 2, 4).view(...)` raises "view size is not compatible
with input tensor's size and stride". `reshape` copies when it has to, and
`.contiguous().view(...)` does the same in two steps. JAX and MLX have no such
distinction: their `reshape` always works.

## Common mistakes

- **Interleaving by concatenation.** `cat((a, b), dim=-1)` has the right shape and
  holds all of `a`, then all of `b`.
- **Stacking on the wrong axis.** `stack((a, b), dim=0).reshape(...)` puts every value
  of `a` before every value of `b` in the list, and the reshape cannot undo that. The
  new axis must be the last one, so that the pair `(a_i, b_i)` is what runs fastest.
- **Patches by reshape alone.** `x.reshape(B, -1, p * p)` gives each "patch" the next
  `p * p` pixels in reading order, which is a strip of the image and not a block. In
  the worked example it returns `x` unchanged.
- **Patches in column-major order.** Permuting to `(b, j, i, u, v)` numbers the patches
  top to bottom first. Square grids of patches hide this in the shape.
- **Pixels in column-major order.** Permuting to `(b, i, j, v, u)` transposes every
  patch.
- **Reading the matrix down its rows.** `w.T.reshape(R, C // g, g)` groups values that
  were in the same column. A transpose before a reshape is the usual way to get a
  right shape from wrong values.
- **Strided groups.** `w.reshape(R, g, C // g).transpose(1, 2)` has the right shape,
  and group `k` holds columns `k`, `k + C / g`, `k + 2C / g`, ...: `g` groups of
  `C / g` were made, then relabeled.

## Complexity

Each function touches every element at most once. `group_columns` copies nothing in
PyTorch (a view). `interleave` and `patchify` copy the data once.

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

The trap in this table is `transpose`. In PyTorch it takes two axes and swaps them. In
JAX and MLX, as in NumPy, it takes a full permutation and is what PyTorch calls
`permute`; with two arguments on a tensor of three axes it raises. `swapaxes` means the
same thing in all three.
