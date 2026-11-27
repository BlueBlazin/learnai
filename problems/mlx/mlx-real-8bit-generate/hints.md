Inspect the checkpoint tensor names: `model.layers.0.self_attn.q_proj.weight`, `.scales`, and `.biases`. Which MLX layers store those exact three parameters and accept `group_size` and `bits` in their constructor? Replace every `nn.Linear` and `nn.Embedding` from `mlx-real-model-forward` with those layers, and think about what `load_weights(path, strict=True)` and `set_dtype(mx.float32)` do to `uint32` versus floating parameters.

---

Build the module tree with `nn.QuantizedLinear(in_dims, out_dims, bias=False, group_size=group_size, bits=bits)` and `nn.QuantizedEmbedding(vocab_size, d_model, group_size=group_size, bits=bits)`. Name the MLP projections `gate_proj`, `up_proj`, and `down_proj` under `mlp`, and always include `lm_head`.

Load with `model.load_weights(str(path / "model.safetensors"), strict=True)`, then call `model.set_dtype(mx.float32)` and `mx.eval(model.parameters())`. `set_dtype` converts only floating parameters, so the `uint32` packed weights stay untouched.

For the KV cache, pass each attention layer its past `(keys, values)`. Rotate the new queries and keys with `self.rope(x, offset=past_length)`, concatenate along the sequence axis, and return the updated pair. Pass `mask="causal"` whenever `T > 1` (the kernel aligns it to the last key) and `mask=None` when `T == 1`.

---

The generation loop matches `mlx-streaming-generate` with an `eos_id` check:

```python
def step(tokens, cache):
    logits, cache = model(tokens, cache)
    return mx.argmax(logits[0, -1]), cache


token, cache = step(mx.array([prompt_ids]), None)
mx.async_eval(token)
out = []
for index in range(max_new_tokens):
    if index + 1 < max_new_tokens:  # nothing follows the last token
        next_token, cache = step(token.reshape(1, 1), cache)
        mx.async_eval(next_token)
    out.append(token.item())  # the only read; the next step is already running
    if out[-1] == eos_id:
        break
    if index + 1 < max_new_tokens:
        token = next_token
return out
```

Make sure `model(tokens, None)` behaves the same as `model(tokens)`.
