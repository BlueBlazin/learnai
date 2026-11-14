## The idea

A tensor is a flat buffer plus a rule for turning an index into an offset. In the
default (row-major) layout the last axis varies fastest, so a `(B, T, D)` tensor stores
token 0's features, then token 1's features, and so on.

Two families of operations change the shape, and they do different things to that
buffer:

- **Reshape** keeps the order of the elements and redraws the axis boundaries. It can
  split one axis into two adjacent ones, or glue adjacent axes together. It cannot
  move an axis.
- **Transpose** (permute, swapaxes) keeps the axes and changes their order. Every
  element stays attached to the same index along each axis; the flat order changes.

`split_heads` needs both. The definition $y[b, h, t, i] = x[b, t, h d + i]$ says two
things: the feature index is decomposed as $h d + i$, and the head index moves in front
of the token index. The first is a reshape of the last axis, `(B, T, D)` to
`(B, T, H, d)`, which is exactly "feature $= h d + i$" in row-major order. The second
is a swap of axes 1 and 2.

`merge_heads` is the inverse, so it applies the inverse steps in reverse order: swap
axes 1 and 2, then glue the last two axes. A reshape is its own kind of inverse (glue
what was split), and a swap of two axes is its own inverse.

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

JAX and MLX use `swapaxes(x, 1, 2)` and the same reshapes.

In PyTorch a transpose is free: it returns a view with different strides over the same
buffer. The price is paid by the next operation. `view` requires the requested shape to
be expressible as strides over the existing buffer and raises otherwise; `reshape`
returns a view when it can and copies when it must. In `merge_heads` the copy is
unavoidable, because the merged layout is a different order of the elements. JAX and
MLX arrays have no notion of contiguity that a caller can trip over, so there the
question does not arise.

## Common mistakes

- **Reshape without the transpose.** `x.reshape(B, T, H, d)` and stopping there gives
  the shape `(B, T, H, d)`. It is wrong unless `T == H`, and then it has the right shape
  with the wrong contents, which is why the tests contain that case.
- **Reshape straight to the target.** `x.reshape(B, H, T, d)` has the right shape for
  every input, and `x.reshape(B, T, D)` undoes it, so a round-trip test passes. Each
  "head" is then a contiguous run of the flat buffer: whole tokens, not a slice of
  every token.
- **Feature axis decomposed the wrong way round.** Reshaping to `(B, T, d, H)` treats
  the feature index as $i H + h$: head `h` gets every `H`-th feature. With a randomly
  initialized model nothing looks wrong; with real weights every head reads the wrong
  rows of the projection matrices. A variant exchanges `H` and `d` and is invisible
  whenever they are equal, as in many small test configurations.
- **Merge with the wrong permutation.** Swapping the last two axes out of habit (the
  `transpose(-2, -1)` of the attention scores), or moving the head axis to the end,
  both give `(B, T, H * d)` after the reshape and both scramble the features.
- **`view` on a transposed tensor.** `x.transpose(1, 2).view(B, T, D)` raises in
  PyTorch for an ordinary input: "view size is not compatible with input tensor's size
  and stride". It does work when `x` is exactly what `split_heads` returned, because
  the second transpose restores the original layout, so a test that only round-trips
  never sees the error.

## Complexity

$O(B\,T\,D)$ elements are moved at most once. In PyTorch, `split_heads` allocates
nothing (it returns a view) and `merge_heads` copies once.

## In the wild

Every attention implementation contains these lines, usually inline:
`q = q_proj(x).view(B, T, H, d).transpose(1, 2)` on the way in, and
`out.transpose(1, 2).contiguous().view(B, T, D)` (or `.reshape(B, T, D)`) on the way
out. The `(B, H, T, d)` layout is chosen so that the two axes the attention product
works on, tokens and features, are last, and batch and head behave as one combined
batch axis for `matmul` and for fused attention kernels.

Variations on the theme come up later. With grouped-query attention the key and value
projections have fewer heads than the query projection, so they are split with a
different `H` and then repeated. A KV cache stores keys and values already split, in
`(B, H, T, d)`, so the split is paid once per token and never again. Engines that
care about copies choose the layout of the cache so that appending a token along `T`
does not require a transpose of everything stored so far.
