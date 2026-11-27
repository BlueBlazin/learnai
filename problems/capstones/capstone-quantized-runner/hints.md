The float weights exist only inside `quantize_checkpoint`. Every later stage has access only to three tensors per matrix: packed uint8 bytes, per-group scales, and per-group biases. Which two helper functions, written once against those three tensors, let the `decoder-forward` pass run almost unchanged?

---

Build four small helpers:
1. `quantize_groups(w, bits, group_size)` from `groupwise-4bit`.
2. `pack` and `unpack` (for 4 bits, pack `low + 16 * high`, and unpack with `% 16` and `// 16`, then stack and reshape).
3. `qlinear(ckpt, name, x, group_size)`, which loads `name`, `...scales`, and `...biases`, infers `bits` from their shapes, unpacks the codes, and applies the `quantized-matmul` identity.
4. `embed`, which indexes the three tensors by `tokens` first and dequantizes only the selected rows.

Make `step_q` the single real forward pass (`forward_q` calls `step_q` with `cache=None`). Use `positions = S + arange(T)`, mask with `arange(S + T)[None, :] <= positions[:, None]`, and apply RoPE to keys before concatenating them to the cache. Have `generate_q` call `step_q` once for the full prompt and then once per generated token. For stage 4, generalize the quantizer to take a per-tensor bit width and sum the stored bits directly from the checkpoint tensors.

---

The core operations of each stage:

```python
# stage 1, per matrix
codes, scale, bias = quantize_groups(w, bits, group_size)
ckpt[name] = codes if bits == 8 else codes.reshape(out, -1, 2)[..., 0] + 16 * codes.reshape(out, -1, 2)[..., 1]
ckpt[name[:-6] + "scales"], ckpt[name[:-6] + "biases"] = scale, bias

# stage 2, a linear layer on codes (after unpacking to float32 `codes`)
xg = x.reshape(*x.shape[:-1], n_groups, group_size)
cg = codes.reshape(out, n_groups, group_size)
y = (einsum("...gk,ogk->...og", xg, cg) * scale).sum(-1) + xg.sum(-1) @ bias.T

# stage 3, inside a layer
q, k = ops.rope(q, positions, base), ops.rope(k, positions, base)  # positions = S + arange(T)
if cache is not None:
    k, v = concat((cache[i][0], k), axis=2), concat((cache[i][1], v), axis=2)
new_cache.append((k, v))
attended = ops.attention(q, k, v, visible)

# stage 4
stored_bits = sum(elements * (8 if uint8 else 32) for every tensor)
parameters = sum(scales.size * group_size for every matrix) + sum(gain.size for every gain)
```
