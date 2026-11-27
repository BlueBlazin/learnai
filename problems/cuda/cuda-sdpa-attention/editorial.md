## The idea

`scaled_dot_product_attention` computes $\operatorname{softmax}(q k^\top / \sqrt{D} + \text{bias})\, v$ and lets the caller specify which entries receive $-\infty$ bias in two ways.

**`is_causal=True`** tells the kernel to use a fixed lower-triangular pattern without allocating a mask tensor. PyTorch defines it as `torch.ones(T, S).tril()`, anchored at the top-left corner, so query $i$ sees slots $0$ through $i$. For a square prefill without padding (`n_cached == 0`), queries and keys are the same $T$ tokens, so this pattern is exact. With a KV cache, however, the $T$ queries are the *last* $T$ of the $S$ keys. Query $i$ sits at slot $\texttt{n\_cached} + i$ and must see every slot up to that position, whereas the top-left triangle gives it only $i + 1$ slots. For a single decode step ($T = 1$), `is_causal=True` restricts attention to slot 0 alone.

**`attn_mask`** handles every other layout. As a boolean tensor, `True` means a query-key pair participates in attention, and the tensor only needs to broadcast to `(B, N_q, T, S)`. The causal condition compares column indices `0..S-1` against shifted row indices `n_cached..n_cached+T-1`, producing a `(T, S)` boolean tensor. Padding varies per sequence and per slot, so it broadcasts as `(B, 1, 1, S)`. Combining them with `&` produces a `(B, 1, T, S)` boolean mask (one byte per entry) shared across all heads.

PyTorch 2.15 does not allow combining both arguments and raises `"Explicit attn_mask should not be set when is_causal=True"`. Therefore, `attend` uses `is_causal=True` only when `n_cached == 0` and `key_padding is None`, and builds an explicit boolean `attn_mask` in all other cases.

`key_padding` follows `nn.MultiheadAttention`, where `True` marks a slot to ignore. Because `scaled_dot_product_attention` uses the opposite convention (`True` means keep), we invert `key_padding` with `~`.

For grouped-query attention with group size $g = N_q / N_{kv}$, passing `enable_gqa=True` instructs the kernel to pair query head $h$ with key/value head $\lfloor h / g \rfloor$ (the `repeat_interleave` order) without copying the KV cache $g$ times.

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

The index tensors `rows` and `cols` are created on `q.device`. Calling `torch.arange(steps)` without `device=` creates a CPU tensor, which raises a device-mismatch error when `q` is on CUDA or `meta`.

## Common mistakes

- **Using `is_causal=True` with a cache.** The triangle is anchored at slot 0, so new tokens see only the first few cached slots and miss recent tokens and themselves.
- **Using `is_causal=True` for every prefill.** When `key_padding` is provided, padding slots remain visible.
- **Passing both `attn_mask` and `is_causal=True`.** The kernel does not intersect them and raises an error.
- **Wrong padding polarity.** Writing `mask & key_padding` keeps only the padding slots instead of hiding them.
- **Using `<` instead of `<=`.** A token must be able to attend to its own slot. With `<`, the first token of a sequence has no visible slots.
- **Tiling key/value heads with `k.repeat(1, g, 1, 1)`.** Tiling produces head order `0, 1, 0, 1`, whereas grouped-query attention requires `0, 0, 1, 1`.
- **Expanding heads manually with `repeat_interleave`.** The numerical output matches, but every layer copies its KV cache $g$ times per call and violates the rule to pass `k` and `v` as-is.
- **Passing an additive float mask.** A float32 mask uses four bytes per entry instead of one, must match `q.dtype`, and violates the boolean-mask rule.
- **Always building a mask.** This skips the `is_causal=True` fast path on square prefills without padding and allocates an unnecessary `(T, T)` tensor.
- **Creating the mask on the default device.** This passes on CPU tests but fails on GPU or `meta` tensors.
- **Computing attention manually with `softmax(q @ k.T) @ v`.** That materializes the `(B, N_q, T, S)` score matrix, defeating the purpose of the fused kernel.

## Complexity

$O(B\, N_q\, T\, S\, D)$ time. The explicit boolean mask uses $O(T S)$ bytes without padding and $O(B\, T\, S)$ bytes with padding. Square prefills without padding build no mask.

## In the wild

PyTorch 2 inference code uses `is_causal=True` for training and unpadded prefill. For cached decoding, it passes an explicit mask (or no mask when $T = 1$ without padding, since a single decode token sees all slots). According to the PyTorch 2.15 documentation, on CUDA `scaled_dot_product_attention` dispatches to FlashAttention-2, memory-efficient attention, or a C++ fallback depending on the inputs, and `enable_gqa` is marked experimental. On CPU, PyTorch always uses the C++ implementation, which is what the CPU tests ran against. Which fused CUDA kernel accepts which mask shape on a GPU was not verified by running them.
