Start from a standard pre-norm decoder block with grouped-query attention and list the six things Granite changes: two multipliers, one attention scale, one RoPE pairing layout, one RoPE base, and one fused MLP matrix. Identify which line of your block each detail affects. None of these six details changes a tensor shape, so missing one will not raise a shape error.

---

Build the block in this order and verify shapes at each step:

1. Build the prefix `p = f"model.layers.{layer}."` (including the trailing dot) and compute `h = ops.rms_norm(x, weights[p + "input_layernorm.weight"], 1e-5)`.
2. Project `q`, `k`, and `v`, reshape to `(B, T, -1, 64)`, and transpose the head axis to position 1. Using `-1` infers 16 heads for `q_proj` and 4 for `k_proj` and `v_proj` automatically.
3. Implement `rope(z, positions)` for shape `(B, n, T, 64)` using 32 frequencies $10^{7 \cdot (-i/32)}$ and angles `positions[:, None] * freq` of shape `(T, 32)`. Both the first 32 features and the last 32 features use those same 32 angles.
4. Repeat each key/value head $G$ times along the head axis (`repeat_interleave` / `repeat`, not `tile`). Compute `scores = q @ k.transpose(-1, -2) / 64`, mask above the diagonal with `-inf`, apply `ops.softmax`, multiply by `v`, merge heads, and project with `o_proj`.
5. Add `0.263 *` the attention output to `x`. Normalize again, multiply by `input_linear`, split the last axis in half, compute `silu(first) * second`, project with `output_linear`, and add `0.263 *` the MLP output to `x`.

---

Here is the half-split rotation in PyTorch:

```python
half = 32
inv_freq = 1e7 ** (-torch.arange(half, dtype=torch.float32) / half)
angles = positions.to(torch.float32)[:, None] * inv_freq  # (T, 32)
cos = torch.cat([angles.cos(), angles.cos()], dim=-1)  # (T, 64)
sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
rotated = torch.cat([-z[..., half:], z[..., :half]], dim=-1)  # (a, b) -> (-b, a)
return z * cos + rotated * sin
```

`z * cos + rotated * sin` computes $(a \cos - b \sin,\; b \cos + a \sin)$ for all 32 pairs at once. Scale attention scores with `(q @ k.transpose(-1, -2)) * (1 / 64)`, and update the residual stream with `x = x + 0.263 * attn_out` and `x = x + 0.263 * mlp_out`, scaling only the branch and never `x`.
