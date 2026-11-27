## The idea

In a causal model, the output at position $p$ depends only on the query at $p$ and the keys and values at positions $0..p$. Earlier positions never change as new tokens arrive, so their keys and values can be computed once and reused. A cached step with $S$ past tokens and $T$ new tokens therefore computes only the bottom $T$ rows of the full $(S + T) \times (S + T)$ attention matrix against all $S + T$ columns.

Viewed this way, the mask follows the same causal rule as full-sequence attention. In the full $(S + T) \times (S + T)$ causal mask, row $p$ allows columns $j \le p$. Keeping rows $p = S, \dots, S + T - 1$ means row $i$ of the step allows columns $j \le S + i$. That forms a rectangular mask made of a fully visible $T \times S$ block for the cache followed by a $T \times T$ lower-triangular block for the new chunk.

Two special cases explain why mask bugs often slip through testing. When $S = 0$, the cached block is empty and the mask reduces to a standard square causal triangle. When $T = 1$, the single row sees every column, so a decode step works even with no mask at all.

Concatenating along the sequence axis with old entries first makes column index equal to absolute sequence position. That alignment lets a single broadcasted comparison of two index ranges build the entire mask.

## Reference solution (PyTorch)

```python
def attend_step(q_new, k_new, v_new, cache):
    if cache is None:
        k, v = k_new, v_new
    else:
        k = torch.cat((cache[0], k_new), dim=2)
        v = torch.cat((cache[1], v_new), dim=2)
    new, total = q_new.shape[2], k.shape[2]
    past = total - new
    visible = torch.arange(total)[None, :] <= past + torch.arange(new)[:, None]  # (T, S + T)
    return ops.attention(q_new, k, v, visible), (k, v)
```

`torch.cat` allocates new tensors, leaving the input cache untouched so the function behaves as a pure map from `(inputs, state)` to `(output, state)`. The JAX and MLX solutions use the same logic with `concatenate` and `axis=2`. Passing and returning a tuple rather than mutating a growing buffer fits all three frameworks, since JAX arrays are immutable and MLX arrays represent nodes in a lazy computation graph.

## Common mistakes

- **Omitting the mask.** Computing unmasked `softmax(q K^T) V` works when `T = 1` but fails for multi-token chunks because each new token attends to later tokens in the same chunk. Outputs then depend on how the sequence is split.
- **Forgetting the cache offset in the mask.** Reusing a square-mask condition `j <= i` on a `(T, S + T)` score matrix hides most of the cache, letting token `i` see only the first `i + 1` entries. This passes when `cache` is `None` and fails as soon as `S > 0`.
- **Attending before appending.** Attending over the old cache alone prevents new tokens from seeing themselves or earlier tokens in their own chunk. You must append first, then attend.
- **Concatenating on the wrong axis.** In `(B, H, T, D)` layout, the sequence axis is axis 2. Code ported from `(B, T, H, D)` layout concatenates on axis 1 and stacks heads instead. It usually crashes with a shape mismatch, except when the cache and chunk happen to have equal lengths.
- **Putting new entries before old ones.** Reversing the concatenation order breaks sequence order, invalidating any mask or position rule that assumes column index equals absolute position.
- **Returning the old cache.** Returning the input `cache` instead of `(k, v)` produces the right output for the current step, but the next step loses the newly processed tokens.

## Complexity

One step takes $O(B\,H\,T\,(S + T)\,D)$ time to compute attention scores and mix values. Feeding $n$ tokens one by one therefore still costs $O(n^2)$ time in total, the same asymptotic order as a single full pass. The cache does not reduce the asymptotic cost of attention itself, but it eliminates recomputing the projections, keys, values, and MLPs for every earlier token in every layer.

Concatenation copies the entire cache on every call, which moves $O(S)$ elements per step and $O(n^2)$ elements over a full generation. Avoiding those repeated copies is why later problems in this track write into a preallocated buffer.

## In the wild

Hugging Face Transformers uses this exact concatenation pattern in its default `DynamicCache`, where each layer updates its state with `torch.cat([self.keys, key_states], dim=-2)`. Its `StaticCache` avoids copies by preallocating a fixed buffer instead. In `mlx-lm`, `KVCache.update_and_fetch(keys, values)` exposes a similar interface (pass new keys and values, get back the full sequence to attend over) backed by a buffer that grows in 256-token increments. Across all of these engines, multi-token steps use a `(T, S + T)` causal mask offset by the current cache length.
