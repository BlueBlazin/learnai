Look at the tensor names: `model.layers.0.self_attn.q_proj.weight`, `.scales`,
`.biases`. Which MLX layer has exactly those three parameters, and takes `group_size`
and `bits` in its constructor? If every `nn.Linear` and the `nn.Embedding` of the
previous problem's module are replaced by their quantized counterparts, what is left
to do so that `load_weights(path, strict=True)` succeeds and nothing is ever
dequantized?

---

Build the tree of `mlx-real-model-forward` with `nn.QuantizedLinear(in, out,
bias=False, group_size=g, bits=b)` and `nn.QuantizedEmbedding(vocab, d,
group_size=g, bits=b)`, the MLP as `gate_proj`, `up_proj`, `down_proj` under `mlp`,
and an `lm_head`. Load with `model.load_weights(str(path / "model.safetensors"),
strict=True)`, then `model.set_dtype(mx.float32)`: it converts floating parameters
only, so the uint32 codes are untouched. For the cache, have each attention layer
take its past `(keys, values)`, rotate the new queries and keys with
`self.rope(x, offset=past_length)`, concatenate, and return the new pair; pass
`mask="causal"` when more than one token is fed (the kernel aligns it to the last
key) and no mask for a single token.

---

The generation loop is `mlx-streaming-generate` with an EOS test:

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

`model(tokens, None)` must behave like `model(tokens)`.
