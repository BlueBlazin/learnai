## The idea

`mx.fast.scaled_dot_product_attention` computes `softmax(scale * q @ k.T + mask) @ v`
as one operation. Three of its arguments carry all the decisions:

**`scale`** is required and has no default: the kernel multiplies by whatever it is
given. $1/\sqrt{D}$ is `q.shape[-1] ** -0.5`.

**Heads.** `k` and `v` are passed with their own number of heads. With fewer
key/value heads than query heads, the kernel gives each key/value head to a block of
consecutive query heads, which is the grouped-query convention (`mx.repeat` along the
head axis, not `mx.tile`).

**`mask`** selects, for each query, the slots it may read:

| layout | what each new token may read | mask |
|---|---|---|
| full buffer, one new token, or `causal=False` | every slot | `None` |
| full buffer, several new tokens, causal | everything up to its own slot | `"causal"` |
| unused capacity after the valid slots | valid slots only (up to its own, if causal) | an array |

The string deserves a careful reading. `"causal"` is not "a square lower-triangular
mask": with `T` queries and `S > T` keys it is aligned at the lower right, so the last
query sees all `S` keys and query $i$ sees the first $S - T + 1 + i$. For a cache that
is exactly full, $S - T$ *is* `n_cached`, and that is the correct mask for new tokens
after a cache. (An older reading, that the string only fits a square prefill, is not
how MLX 0.32.4 behaves; it was verified for this problem by comparing against explicit
masks.) The alignment is with the end of the *buffer*, though, not with the end of the
*valid tokens*. As soon as the buffer has slots that hold no token, the string hands
them to the queries.

For that case the mask has to be an array. A boolean one is the easy choice:

$$
\text{mask}[i, j] = \left(j \le \texttt{n\_cached} + i\right) ,
$$

`True` meaning "may attend". It broadcasts over batch and heads, costs `T * S` bytes,
and has no float dtype to get wrong.

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

Slicing the buffers to their valid part (`k[:, :, :used]`) and then using the string
is also correct and is accepted; it copies the valid keys and values on every call,
and its shapes change with `n_cached`, which a compiled decode step cannot have.

## What the kernel buys

Measured on this machine (M1 Max, MLX 0.32.4), float32, 8 heads, $D = 64$, peak
memory above the inputs, and the best of five timings:

| shapes | fused kernel | `matmul`, `softmax`, `matmul` |
|---|---|---|
| prefill, $T = S = 2048$ | 3.6 ms, 4 MiB | 3.4 ms, 136 MiB |
| $T = 512$ new tokens, $S = 4096$ | 1 to 3 ms, 1 MiB | 1.6 ms, 66 MiB |
| one token, $S = 4096$ | 0.26 ms, 0.1 MiB | 0.47 ms, 0.1 MiB |

The time is about the same. The memory is not: the hand-written version stores the
scores (8 × 2048 × 2048 × 4 bytes = 128 MiB) and the kernel never does, so its peak
is the 4 MiB of the output. The form of the mask (string, boolean, additive) made no
measurable difference to either number. In float16 the hand-written peak is 68 MiB
and the kernel's 2 MiB; the kernel computes its softmax in float32 internally, as
its documentation notes, and returns float16. For head
dimensions without a specialized Metal kernel (it exists for 64, 96, 128, 192, 256,
and 512) the same call falls back to composed operations with the same semantics.

## Common mistakes

- **`"causal"` for every causal call.** Right for a growing cache, wrong for a
  preallocated buffer: with 5 cached tokens, 3 new ones, and 16 slots, the last new
  token reads all 16 slots.
- **A boolean mask with the opposite meaning.** PyTorch's `masked_fill(mask, -inf)`
  and `key_padding_mask` use `True` for "hide". Here `True` is "may attend"
  (as in `torch.nn.functional.scaled_dot_product_attention`).
- **Leaving out the scale**, or passing `1.0`. Nothing fails; the softmax is sharper
  by a factor of $\sqrt{D}$ in the exponent.
- **An additive mask of the wrong dtype.** `mx.where(visible, 0.0, -inf)` is float32
  (two Python scalars have no dtype to adopt). With float16 inputs the kernel raises
  `Mask type must promote to output type float16`. Create it with
  `dtype=q.dtype`, or use a boolean mask.
- **Expanding the key/value heads with `mx.tile`.** Tiling gives query head $h$ the
  key/value head $h \bmod N_{kv}$; the convention is $\lfloor h / (N_q/N_{kv}) \rfloor$.
  Expanding at all is unnecessary: it copies the cache once per group.
- **`<` instead of `<=`.** A token must be able to attend to itself. For the first
  token of a sequence the mask then hides every slot: with an additive `-inf` mask
  the row is NaN, and with a boolean mask this kernel returns the plain average of
  all values, a wrong number that looks fine.
- **Forgetting the cache in the mask**: `j <= i` instead of `j <= n_cached + i`, the
  square mask of a prefill reused for a continuation.
- **No mask when `causal=False`.** "Not causal" removes the triangle, not the
  distinction between valid and unused slots.
- **Hand-written attention.** Correct numbers, 30 times the memory at prefill sizes.

## Complexity

$O(B \cdot N_q \cdot T \cdot S \cdot D)$ time. Memory beyond the output is constant
for the fused kernel, against $O(B \cdot N_q \cdot T \cdot S)$ for the scores of the
hand-written version. The explicit mask adds $T \cdot S$ bytes.

## In the wild

`mlx_lm.models.base.create_attention_mask` makes exactly the full-buffer choice:
`None` for a single token, `"causal"` otherwise, and an explicit array only when a
model needs something else (a sliding window, or a cache that builds its own mask).
Its caches grow, and what they hand to attention is always exactly full. The layout
with unused capacity is the one you build yourself when you preallocate the cache so
that a step's shapes never change, which `mlx-compile-step` does later in this
track; there `n_cached` is an array, and the comparison above becomes part of the
compiled graph.
