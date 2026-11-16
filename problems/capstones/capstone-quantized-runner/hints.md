The float weights exist only inside `quantize_checkpoint`. For every later stage,
ask what you can compute from three tensors per matrix: a row of bytes, and one
scale and one bias per group. Which two helper functions, written once, would let
the forward pass of `decoder-forward` run almost unchanged?

---

Write four helpers. `quantize_groups(w, bits, group_size)` from `groupwise-4bit`;
`pack` and `unpack` (for 4 bits: `low + 16 * high`, and back with `% 16` and
`// 16`, interleaved with a `stack` and a `reshape`); `qlinear(ckpt, name, x,
group_size)`, which reads `name`, `...scales`, `...biases`, derives the bits from
the shapes, unpacks, and applies the identity of `quantized-matmul`; and `embed`,
which indexes the three tensors with the token ids first and dequantizes only those
rows. Then make `step_q` the one real forward pass (`forward_q` is `step_q` with
`cache=None`): positions are `S + arange(T)`, the mask is `arange(S + T)[None, :]
<= positions[:, None]`, keys are rotated before they are concatenated to the cache.
`generate_q` calls `step_q` with the whole prompt once and then with one token at a
time. For stage 4, pass a function from name to bits into the quantizer instead of
a single number, and count bits from what the checkpoint holds.

---

The core of each stage:

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
