Implement three functions that rearrange tensor elements without modifying their values. `group_columns` splits each row of a matrix into fixed-size groups, `interleave` alternates elements from two tensors along the last axis, and `patchify` divides a batch of 2D images into square patches.

## Background

A tensor's elements have a single **row-major** order in which the last index changes fastest. For example, `[[0, 1, 2], [3, 4, 5]]` reads left to right, then top to bottom, as `0, 1, 2, 3, 4, 5`.

`reshape` preserves that flat element order and only redraws the boundaries between axes. It can split one axis into several or merge neighboring axes into one, and nothing else. `transpose` and `permute` reorder the axes, which changes the row-major order of the elements. `stack` joins tensors along a new axis.

> [!WARNING]
> Getting the right output shape does not guarantee the right rearrangement. `reshape` accepts any target shape with the same element count and never raises an error when values land in the wrong positions.

## Specification

```python
def group_columns(w, g): ...
def interleave(a, b): ...
def patchify(x, p): ...
```

**`group_columns(w, g)`** takes a 2D tensor `w` of shape `(R, C)` and a Python `int` `g` that divides `C`. It splits the columns of every row into `C / g` consecutive groups of size `g`. For row `r`, group `k`, and within-group index `i`, the output has shape `(R, C / g, g)` with

$$
\text{out}[r, k, i] = w[r,\; k\,g + i] .
$$

**`interleave(a, b)`** takes two tensors `a` and `b` of the same shape `(..., n)`, with 1 to 3 axes, and alternates their elements along the last axis. For each index `i` along the last axis, the output has shape `(..., 2n)` with

$$
\text{out}[\ldots, 2i] = a[\ldots, i], \qquad \text{out}[\ldots, 2i + 1] = b[\ldots, i] .
$$

**`patchify(x, p)`** takes a batch of 2D images `x` of shape `(B, H, W)` and a Python `int` `p` that divides both `H` and `W`. It partitions each image into a grid of `H / p` patch rows and $n_w = W / p$ patch columns, where each patch is `p` by `p`. Patches are numbered left to right and then top to bottom, and pixels inside each patch follow the same row-major order. For batch index `b`, patch row `i`, patch column `j`, and in-patch pixel coordinates `(u, v)`, the output has shape `(B, (H / p) * (W / p), p * p)` with

$$
\text{out}[b,\; i\,n_w + j,\; u\,p + v] = x[b,\; i\,p + u,\; j\,p + v] .
$$

Each function returns a float32 tensor containing the exact elements of its inputs, each once, rearranged. Whether the result shares memory with the input does not matter.

## Shapes

| name | shape | dtype |
|---|---|---|
| `w` | `(R, C)` | float32 |
| result of `group_columns` | `(R, C / g, g)` | float32 |
| `a`, `b` | `(..., n)` | float32 |
| result of `interleave` | `(..., 2n)` | float32 |
| `x` | `(B, H, W)` | float32 |
| result of `patchify` | `(B, (H / p) * (W / p), p * p)` | float32 |

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

In the `patchify` example, the output has the same shape `(1, 4, 4)` as `x`, yet `x` itself is not the answer. Its first row `[0.0, 1.0, 2.0, 3.0]` spans the top rows of two different patches rather than forming one `2 x 2` patch.

## Constraints

- Results are compared with the reference exactly, because no arithmetic is performed.
- Every dimension size is between 1 and 16. Edge cases `g = 1`, `g = C`, `p = 1`, and `p = H = W` all occur.
- Inputs are freshly created tensors, contiguous in memory.
- No Python loops are needed.

## Why this matters

Splitting and merging attention heads in a transformer is `group_columns` followed by a transpose, and its inverse. Rotary Position Embeddings (RoPE) rotate even and odd features separately and recombine them with `interleave`. Quantized model runners unpack two 4-bit codes from every byte the same way. `patchify` is the first layer of every vision encoder.

All of these operations share the same silent failure mode. A wrong reshape still produces the expected shape, so later layers run without raising an error while reading values from the wrong positions.
