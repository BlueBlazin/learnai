## The idea

The kernel computes $\operatorname{softmax}(q k^\top / \sqrt{D} + \text{bias})\, v$
and leaves one decision to the caller: which entries of the bias are $-\infty$. It
offers two ways to say so.

**`is_causal=True`** is a promise about the layout, not a mask you supply: the
documentation defines it as `torch.ones(L, S).tril()`, a triangle anchored at the top
left, so query $i$ sees keys $0$ to $i$. For a prefill, where queries and keys are
the same $T$ tokens, that is causal attention, and because the kernel knows the
pattern it does not need a mask tensor at all. With a cache the queries are the
*last* $T$ of the $S$ keys. Query $i$ sits at slot $\texttt{n\_cached} + i$ and must
see everything up to there; the top-left triangle gives it $i + 1$ slots instead. For
a single decode step ($T = 1$) that is slot 0 alone.

**`attn_mask`** is the general form. As a boolean tensor, `True` means the pair
takes part in attention, and the tensor only has to broadcast to `(B, N_q, T, S)`.
The causal part is a `(T, S)` comparison of slot index with shifted row index; the
padding part is per row of the batch and per slot, so it enters as `(B, 1, 1, S)`.
Their conjunction is `(B, 1, T, S)`, one byte per entry, shared by all heads.

The two cannot be combined: PyTorch 2.15 raises "Explicit attn_mask should not be
set when is_causal=True". So the function has exactly one case for `is_causal`
(no cache, no padding) and one mask for everything else.

The padding mask arrives in the convention of `nn.MultiheadAttention`, where `True`
marks a key to ignore. The documentation of `scaled_dot_product_attention` carries a
note that its boolean semantics are the inverse, and that masks migrated from
`MultiheadAttention` must be inverted. Hence the `~`.

Grouped-query attention needs no arithmetic either. `enable_gqa=True` makes the
kernel pair query head $h$ with key/value head $\lfloor h / g \rfloor$, the
`repeat_interleave` order, without the caller copying the cache $g$ times.

## Reference solution (PyTorch)

```python
def attend(q, k, v, n_cached, key_padding):
    steps, slots = q.shape[-2], k.shape[-2]

    if n_cached == 0 and key_padding is None:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

    rows = torch.arange(steps, device=q.device)
    cols = torch.arange(slots, device=q.device)
    mask = cols[None, :] <= n_cached + rows[:, None]  # (T, S)
    if key_padding is not None:
        mask = mask & ~key_padding[:, None, None, :]  # (B, 1, T, S)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
```

The mask is created on `q.device`. `torch.arange(steps)` alone would be a CPU tensor;
next to CUDA operands the kernel call raises.

## Common mistakes

- **`is_causal=True` with a cache.** The triangle is anchored at slot 0. The new
  tokens see the first few cached slots and nothing recent, themselves included.
- **`is_causal=True` for every prefill.** With `key_padding` given, the padding
  slots are attended to.
- **`attn_mask` and `is_causal=True` together**, hoping the kernel intersects them.
  It raises.
- **Padding with the wrong polarity.** `mask & key_padding` keeps exactly the slots
  that should be hidden.
- **`<` for `<=`.** A token must attend to its own slot; with `<` the first token of
  a sequence has nothing to attend to.
- **Tiling the key/value heads** with `k.repeat(1, g, 1, 1)`. The heads come out as
  0, 1, 0, 1; grouped-query attention needs 0, 0, 1, 1.
- **Repeating the heads yourself** with `repeat_interleave`. The numbers are right,
  and every layer copies its cache $g$ times per call.
- **An additive float mask.** It works, costs four bytes per entry in float32
  instead of one, and has to follow the dtype of `q`.
- **Always building the mask.** Correct numbers, and a `(T, T)` tensor built and
  applied on every prefill where the kernel could have used its causal fast path.
- **Creating the mask on the default device.** Invisible on a CPU, an exception on a
  GPU.
- **Computing attention by hand.** `softmax(q @ k.T) @ v` materializes the
  `(B, N_q, T, S)` scores; the fused kernels exist to avoid exactly that.

## Complexity

$O(B\, N_q\, T\, S\, D)$ time. The mask is $O(T S)$ booleans, or $O(B\, T\, S)$ with
padding; the square prefill builds none.

## In the wild

Model code written for PyTorch 2 follows this shape: `is_causal` for training and
prefill, an explicit mask or no mask at all for decode steps (a single new token may
see every slot, so `T = 1` without padding needs no mask, which the tests accept). According to
the PyTorch 2.15 documentation the function dispatches on CUDA to FlashAttention-2,
memory-efficient attention, or a C++ implementation of the formula, chosen from the
inputs, and grouped-query attention is marked experimental there. On other devices
the C++ implementation is always used, which is what the CPU cases ran against; which
fused kernel accepts which mask form on a GPU is a property of the kernels that was
not verified by running them.
