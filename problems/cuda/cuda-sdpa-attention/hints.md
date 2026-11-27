`is_causal=True` means query `i` sees slots `0` through `i`. For which value of `n_cached` does that match the problem rule? And when `key_padding` is given, which of the kernel's two mask arguments can still express both causality and padding?

---

The `is_causal=True` fast path fits only when there is no cache (`n_cached == 0`) and no padding (`key_padding is None`). Handle that case first with no mask tensor. For every other case, build a boolean `attn_mask` where `True` means *may attend*.

For the causal part, slot `j` is visible to new token `i` when `j <= n_cached + i`. Build both index ranges with `torch.arange(..., device=q.device)`. Because `key_padding` uses the opposite convention (`True` means hide), invert it with `~` and reshape it to `(B, 1, 1, S)` so it broadcasts over heads and queries.

For grouped-query attention, do not repeat `k` and `v`. Pass `enable_gqa=True` to the kernel instead.

---

```python
rows = torch.arange(T, device=q.device)
cols = torch.arange(S, device=q.device)
mask = cols[None, :] <= n_cached + rows[:, None]  # (T, S)
if key_padding is not None:
    mask = mask & ~key_padding[:, None, None, :]  # (B, 1, T, S)
out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
```

When `n_cached == 0 and key_padding is None`, call `F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)`.
