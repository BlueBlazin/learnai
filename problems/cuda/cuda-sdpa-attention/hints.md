`is_causal=True` means "query `i` sees slots `0` to `i`". For which values of
`n_cached` is that the rule of this problem? And when `key_padding` is given, which
of the kernel's two mask arguments can still express what you need?

---

Only one layout fits `is_causal=True`: no cache and no padding. Handle it first, with
no mask at all. For everything else build one boolean mask in which `True` means
*may attend*. The causal part compares slot numbers with row numbers shifted by the
cache: slot `j` is visible to new token `i` when `j <= n_cached + i`; build both
ranges with `torch.arange(..., device=q.device)`. `key_padding` says the opposite of
what the kernel wants (`True` = hide), so invert it, and give it the axes
`(B, 1, 1, S)` so that it broadcasts over heads and queries.

For the heads, do not repeat `k` and `v`: pass `enable_gqa=True`.

---

```python
rows = torch.arange(T, device=q.device)
cols = torch.arange(S, device=q.device)
mask = cols[None, :] <= n_cached + rows[:, None]  # (T, S)
if key_padding is not None:
    mask = mask & ~key_padding[:, None, None, :]  # (B, 1, T, S)
out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
```

and, when `n_cached == 0 and key_padding is None`,
`F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)`.
