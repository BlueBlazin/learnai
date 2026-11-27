The cache state after any call depends only on the total number of appended tokens. The newest `keep_recent` tokens stay in `float32`, and all older tokens are quantized. Think about combining everything currently in `float32` (the old float window plus the new chunk) and asking how many of its oldest tokens must leave.

---

Concatenate the old float window and the new chunk along the sequence axis (`dim=2`). If the combined tensor has `n` tokens, the oldest `leaving = max(0, n - keep_recent)` tokens are quantized in this call and the remaining tokens become the new float window. Note that `leaving` may reach into the new chunk or be `0`. Slicing `[:, :, :0]` yields a length-0 tensor that quantizes to length-0 codes, scales, and biases, so you can concatenate without special-casing empty slices. Existing quantized tensors are only concatenated, never recomputed.

---

To quantize a tensor `x` of shape `(B, H, S, D)`:

```
groups = x.reshape(B, H, S, D // group_size, group_size)
lo, hi = groups.min(-1), groups.max(-1)                 # (B, H, S, D // group_size)
scale  = where(hi > lo, (hi - lo) / (2**bits - 1), 1)
codes  = clamp(round((groups - lo[..., None]) / scale[..., None]), 0, 2**bits - 1)
```

Cast `codes` to `uint8` and reshape back to `(B, H, S, D)`, with `bias = lo`. To read the cache back, reshape `codes` to `(B, H, S, G, D // G)` where `G = scale.shape[-1]`, cast to `float32`, multiply by `scale[..., None]`, add `bias[..., None]`, reshape back to `(B, H, S, D)`, and concatenate the float window after it. Write `D // G` explicitly instead of `-1` so the reshape still works when `S` is `0`.
