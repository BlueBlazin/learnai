## The idea

A tensor consists of a flat memory buffer together with a rule for mapping multi-dimensional indices to buffer offsets. In the default row-major layout, the last axis varies fastest. A `(B, T, D)` tensor therefore stores all features of token 0, followed by all features of token 1, and so on.

Two families of shape operations act on that buffer in fundamentally different ways:

- **Reshape** keeps elements in the same flat order and redraws the axis boundaries. It can split one axis into two adjacent axes or merge adjacent axes into one, but it cannot move an axis past another axis.
- **Transpose** (also called `permute` or `swapaxes`) keeps the axes intact and changes their order. Every element stays at the same coordinate along each axis, while the flat order changes.

`split_heads` requires both operations. The specification $y[b, h, t, i] = x[b, t, h d + i]$ requires decomposing the feature index as $h d + i$ and moving the head axis ahead of the token axis. Reshaping the last axis from `(B, T, D)` to `(B, T, H, d)` performs the $h d + i$ decomposition in row-major order. Swapping axes 1 and 2 then moves the head axis ahead of the sequence axis.

`merge_heads` is the inverse, so it applies the inverse steps in reverse order. First swap axes 1 and 2 back to `(B, T, H, d)`, then reshape the last two axes into `(B, T, H * d)`. Both steps are natural inverses: swapping two axes undoes itself, and merging two adjacent axes undoes splitting them.

## Reference solution (PyTorch)

```python
def split_heads(x, n_heads):
    batch, seq, width = x.shape
    x = x.reshape(batch, seq, n_heads, width // n_heads)
    return x.transpose(1, 2)


def merge_heads(x):
    batch, n_heads, seq, head_width = x.shape
    return x.transpose(1, 2).reshape(batch, seq, n_heads * head_width)
```

JAX and MLX use `swapaxes(x, 1, 2)` along with the same reshapes.

In PyTorch, a transpose does not copy data. It returns a view with updated strides over the same buffer. The cost is paid by the next operation. `view` requires the target shape to be expressible with the existing strides and raises an error otherwise, whereas `reshape` returns a view when possible and copies when necessary. In `merge_heads`, a copy is unavoidable because the merged layout changes the flat element order. JAX and MLX arrays do not expose contiguity errors to callers, so this distinction does not come up there.

## Common mistakes

- **Reshaping without transposing.** Stopping after `x.reshape(B, T, H, d)` leaves the tensor in shape `(B, T, H, d)`. That fails a shape check unless `T == H`, in which case it has the right shape with scrambled contents. The test suite explicitly includes `T == H` to catch this.
- **Reshaping straight to the target shape.** Calling `x.reshape(B, H, T, d)` gives the right shape for every input, and `x.reshape(B, T, D)` reverses it, so a round-trip test passes. However, each head receives contiguous runs of whole tokens instead of a feature slice from every token.
- **Splitting the feature axis in reverse order.** Reshaping to `(B, T, d, H)` interprets the feature index as $i H + h$, assigning every `H`-th feature to head `h`. A randomly initialized model still trains, but a pretrained checkpoint reads the wrong rows of its projection matrices. Swapping `H` and `d` is also invisible whenever `H == d`, which happens often in small unit tests.
- **Using the wrong permutation in `merge_heads`.** Swapping the last two axes out of habit (`transpose(-2, -1)`, as used for attention scores) or moving the head axis to the end still produces shape `(B, T, H * d)` after reshaping, while scrambling the features.
- **Calling `view` on a transposed tensor.** In PyTorch, `x.transpose(1, 2).view(B, T, D)` raises a contiguity error on an ordinary contiguous input. When `x` happens to be the direct output of `split_heads`, the second transpose restores the original contiguous layout and `view` succeeds, so a test that only checks round-trips will miss the bug.

## Complexity

At most $O(B\,T\,D)$ elements are moved once. In PyTorch, `split_heads` returns a view with zero allocation, and `merge_heads` performs a single copy.

## In the wild

Nearly every transformer implementation contains these two steps inline. On the way into attention, models run `q = q_proj(x).view(B, T, H, d).transpose(1, 2)`. On the way out, they run `out.transpose(1, 2).contiguous().view(B, T, D)` or `.reshape(B, T, D)`. Keeping the token and head-feature axes last in `(B, H, T, d)` lets `matmul` and fused attention kernels treat batch and head as a single combined batch dimension.

Several attention variants build directly on this layout. In grouped-query attention (GQA), keys and values have fewer heads than queries, so they are split with a smaller `H` and then repeated. KV caches store keys and values already split in `(B, H, T, d)` so each token is split only once. Inference engines also choose cache layouts so that appending a new token along `T` avoids transposing previously cached tokens.
