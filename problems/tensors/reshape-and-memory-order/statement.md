Write three functions that only move values around: one regroups a matrix, one
alternates two tensors, one cuts images into patches. None of them computes anything.
The exercise is knowing which regroupings a reshape can do on its own and which need
a transpose first.

## Specification

```python
def group_columns(w, g): ...
def interleave(a, b): ...
def patchify(x, p): ...
```

**`group_columns(w, g)`** takes `w` of shape `(R, C)` and cuts the columns of every row
into `C / g` consecutive groups of `g`. It returns shape `(R, C / g, g)` with

$$
\text{out}[r, k, i] = w[r,\; k\,g + i] .
$$

**`interleave(a, b)`** takes two tensors of the same shape `(..., n)` and alternates
their entries along the last axis. It returns shape `(..., 2n)` with

$$
\text{out}[\ldots, 2i] = a[\ldots, i], \qquad \text{out}[\ldots, 2i + 1] = b[\ldots, i] .
$$

**`patchify(x, p)`** takes a batch of images `x` of shape `(B, H, W)` and cuts each one
into `p` by `p` patches. With $n_w = W / p$ patches per row of patches, it returns
shape `(B, (H / p) * (W / p), p * p)` with

$$
\text{out}[b,\; i\,n_w + j,\; u\,p + v] = x[b,\; i\,p + u,\; j\,p + v] .
$$

So patches are numbered left to right and then top to bottom, and the pixels inside a
patch are listed in the same order.

- `w`, `a`, `b`, `x`: float32 tensors. `a` and `b` have one to three axes.
- `g`: a Python int that divides `C`. `p`: a Python int that divides `H` and `W`.
- Every function returns a float32 tensor that holds exactly the values of its input,
  each one once, rearranged. Whether the result shares memory with the input does not
  matter.

## Shapes

| name | shape | dtype |
|---|---|---|
| `w` | `(R, C)` | float32 |
| result of `group_columns` | `(R, C / g, g)` | float32 |
| `a`, `b` | `(..., n)` | float32 |
| result of `interleave` | `(..., 2n)` | float32 |
| `x` | `(B, H, W)` | float32 |
| result of `patchify` | `(B, (H / p) * (W / p), p * p)` | float32 |

## Tools

A tensor's elements have one order, **row-major**: the last index changes fastest, as
in reading `[[0, 1, 2], [3, 4, 5]]` left to right, top to bottom. `reshape` keeps that
order and only redraws the boundaries between axes. It can split one axis into several
or merge neighboring axes into one, and it can do nothing else.

`transpose` and `permute` change the order of the axes, and with it the row-major
order of the elements. `stack` joins tensors along a new axis.

> [!WARNING]
> A reshape to the right shape is not a regrouping. `reshape` accepts any shape with
> the same number of elements and never raises because the values ended up in the
> wrong places.

## Example

```python
w = [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]]  # shape (2, 6)
group_columns(w, 3)
# [[[0.0, 1.0, 2.0], [3.0,  4.0,  5.0]],
#  [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]]          # shape (2, 2, 3)

interleave([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
# [1.0, 10.0, 2.0, 20.0, 3.0, 30.0]

x = [
    [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0]]
]  # shape (1, 4, 4)
patchify(x, 2)
# [[[ 0.0,  1.0,  4.0,  5.0],                     # patch 0: top left
#   [ 2.0,  3.0,  6.0,  7.0],                     # patch 1: top right
#   [ 8.0,  9.0, 12.0, 13.0],                     # patch 2: bottom left
#   [10.0, 11.0, 14.0, 15.0]]]                    # patch 3: bottom right
#                                                 # shape (1, 4, 4)
```

In the last example the result has the shape of `x`, and `x` itself is not the answer:
its first row `[0, 1, 2, 3]` is the top row of two different patches.

## Constraints

- Results are compared with the reference exactly: no arithmetic is involved.
- Every size is between 1 and 16. `g = 1`, `g = C`, `p = 1`, and `p = H = W` all occur.
- Inputs are freshly created tensors, contiguous in memory.
- No Python loops are needed.

## Why this matters

Splitting and Merging Attention Heads is `group_columns` followed by a transpose, and
its inverse. Rotary Position Embeddings (RoPE) rotates the even and the odd features
separately and puts them back with `interleave`, and A Quantized Model Runner unpacks
two 4-bit codes from every byte the same way. `patchify` is the first layer of every
vision encoder. All of them share one failure: a reshape that produces the expected
shape with the values in the wrong places. Nothing raises, the model runs, and its
output is quietly wrong, because every later layer receives the right shape.
