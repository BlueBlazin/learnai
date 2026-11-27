## The idea

`mx.fast.scaled_dot_product_attention` computes `softmax(scale * q @ k.T + mask) @ v` in a single fused operation. Three arguments control its behavior.

**`scale`** is required and has no default. The kernel multiplies scores by whatever value you pass, so $1 / \sqrt{D}$ is `q.shape[-1] ** -0.5`.

**Heads.** Pass `k` and `v` without expanding their head axis. When `N_kv < N_q`, the kernel maps each key/value head to a contiguous block of query heads, matching the grouped-query convention (`mx.repeat` along the head axis, not `mx.tile`).

**`mask`** determines which buffer slots each query may read:

| layout | what each new token may read | mask |
|---|---|---|
| full buffer, one new token, or `causal=False` | every slot | `None` |
| full buffer, several new tokens, causal | everything up to its own slot | `"causal"` |
| unused capacity after the valid slots | valid slots only (up to its own, if causal) | an array |

The `"causal"` string deserves careful attention. It does not mean a square lower-triangular mask. With `T` queries and `S > T` keys, MLX aligns the triangle to the lower right, so the last query sees all `S` keys and query $i$ sees the first $S - T + 1 + i$ keys. When the buffer is full, $S - T$ equals `n_cached`, which is exact for continuing after a cache. Because this alignment anchors to the end of the *buffer* rather than the end of the *valid tokens*, any unused capacity at the tail causes `"causal"` to expose invalid slots.

Whenever unused slots exist (`n_cached + T < S`), pass an explicit mask array. A boolean array is the cleanest option:

$$
\text{mask}[i, j] = \left(j \le \texttt{n\_cached} + i\right)
$$

for the causal case, and $j \le \texttt{n\_cached} + T - 1$ for the non-causal case. `True` means "may attend." The boolean mask broadcasts across batch and heads, uses only `T * S` bytes, and avoids float dtype mismatches.

## Reference solution (MLX)

```python
def attend(q, k, v, n_cached, causal):
    steps, slots = q.shape[2], k.shape[2]
    used = n_cached + steps
    if used == slots:
        mask = "causal" if causal and steps > 1 else None
    else:
        last = n_cached + mx.arange(steps)[:, None] if causal else used - 1
        mask = mx.arange(slots)[None, :] <= last
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5, mask=mask)
```

Slicing the buffers to their valid prefix (`k[:, :, :used]`) and using the string mask is also correct and accepted. However, slicing copies the valid keys and values on every call and changes tensor shapes with `n_cached`, which prevents compiling a fixed-shape decode step.

## What the kernel buys

Measured on an M1 Max (MLX 0.32.4) in `float32` with 8 heads and $D = 64$, reporting peak memory above the inputs and best-of-five runtime:

| shapes | fused kernel | `matmul`, `softmax`, `matmul` |
|---|---|---|
| prefill, $T = S = 2048$ | 3.6 ms, 4 MiB | 3.4 ms, 136 MiB |
| $T = 512$ new tokens, $S = 4096$ | 1 to 3 ms, 1 MiB | 1.6 ms, 66 MiB |
| one token, $S = 4096$ | 0.26 ms, 0.1 MiB | 0.47 ms, 0.1 MiB |

Runtime is similar, but peak memory differs dramatically. The hand-written version materializes the full score tensor (8 × 2048 × 2048 × 4 bytes = 128 MiB), whereas the fused kernel never stores it and peaks at the 4 MiB output. The mask form (string, boolean, or additive) makes no measurable difference to speed or memory. In `float16`, the hand-written peak is 68 MiB versus 2 MiB for the fused kernel, which accumulates softmax in `float32` internally and returns `float16`. For head dimensions without a specialized Metal kernel (specialized kernels exist for 64, 96, 128, 192, 256, and 512), MLX falls back to composed operations with identical semantics.

## Common mistakes

- **Using `"causal"` for every causal call.** This works on a growing cache but fails on a preallocated buffer. With 5 cached tokens, 3 new tokens, and 16 slots, the last new token reads all 16 slots.
- **Inverting the boolean mask.** PyTorch's `masked_fill(mask, -inf)` and `key_padding_mask` use `True` to hide positions. Here, as in `torch.nn.functional.scaled_dot_product_attention`, `True` means "may attend."
- **Omitting `scale` or passing `1.0`.** No error is raised, but the softmax becomes sharper by a factor of $\sqrt{D}$ in the exponent.
- **Passing an additive mask with a wider dtype.** Calling `mx.where(visible, 0.0, -inf)` produces `float32` because Python floats default to `float32`. With `float16` inputs, the kernel raises `Mask type must promote to output type float16`. Cast the additive mask to `q.dtype` or use a boolean mask.
- **Expanding key/value heads with `mx.tile`.** Tiling assigns query head $h$ to key/value head $h \bmod N_{kv}$ instead of $\lfloor h / (N_q / N_{kv}) \rfloor$. Expanding heads manually is unnecessary and copies the cache once per group.
- **Using `<` instead of `<=`.** Every token must attend to itself. With `<`, the first token of a sequence sees no valid slots. An additive `-inf` mask then produces `NaN`, while a boolean mask causes this kernel to return the unweighted average of all values—a silent error.
- **Forgetting `n_cached` in the mask.** Writing `j <= i` instead of `j <= n_cached + i` reuses a square prefill mask for a cached continuation.
- **Passing `mask=None` when `causal=False`.** Disabling causality removes the causal triangle, not the boundary between valid tokens and unused buffer slots.
- **Writing manual attention.** Computing `matmul` and `softmax` by hand fails the kernel-call check and uses 30× more memory at prefill sizes.

## Complexity

Time complexity is $O(B \cdot N_q \cdot T \cdot S \cdot D)$. Extra memory beyond the output is $O(1)$ for the fused kernel, compared to $O(B \cdot N_q \cdot T \cdot S)$ for the hand-written score matrix. An explicit boolean mask adds $T \cdot S$ bytes.

## In the wild

`mlx_lm.models.base.create_attention_mask` implements the full-buffer branch: `None` for a single token, `"causal"` for multiple tokens, and an explicit array only for special layouts like sliding windows. Because `mlx-lm`'s default KV cache grows on demand, the buffer passed to attention is always full. The padded-buffer layout appears when you preallocate a fixed-size cache so a compiled decode step keeps static shapes (as in `mlx-compile-step` later in this track), where `n_cached` becomes an MLX array and the mask comparison runs inside the compiled graph.
