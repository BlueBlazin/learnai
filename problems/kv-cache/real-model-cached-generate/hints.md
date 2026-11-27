Consider one attention layer while decoding the token at position $n$. Only the keys and values from earlier tokens are needed from the past. The query, residual stream, MLP, layer norms, and output head all depend solely on the new token. Decide what each layer needs to store: keys before or after RoPE, and at the width of the 16 query heads or the 4 KV heads?

---

Write a single helper that runs $T$ new tokens after $n$ cached tokens, and build both entry points on top of it (`prefill` uses $n = 0, T = P$, and `decode_step` uses $T = 1$). Four changes turn the `real-model-logits` forward pass into a cached forward pass:

1. Use positions `n, n + 1, ..., n + T - 1` instead of `0, ..., T - 1`.
2. After rotating the new keys, concatenate them (and the new values) onto the layer's cached tensors without mutating the input cache.
3. Build a `(T, n + T)` causal mask so each new token at position `p` attends to positions `0..p`.
4. Pass only the last row through the final norm and output head.

Return the per-layer `(keys, values)` pairs as the new cache, creating new tensors via `cat` or `concatenate` rather than modifying the input cache in place.

---

In PyTorch, inside the layer loop with `cache[i] = (keys, values)` of shape `(n_kv_heads, n, head_dim)`:

```python
q = rope(heads(h @ w_q.T, cfg.n_heads))  # rotated for positions n .. n + T - 1
k = rope(heads(h @ w_k.T, cfg.n_kv_heads))  # rotated once, then cached
v = heads(h @ w_v.T, cfg.n_kv_heads)
if cache is not None:
    k = torch.cat([cache[i][0], k], dim=1)
    v = torch.cat([cache[i][1], v], dim=1)
new_cache.append((k, v))
scores = q @ k.repeat_interleave(group, dim=0).transpose(-1, -2) * cfg.attention_scale
```

Implement `generate` by calling `prefill`, then looping `max_new_tokens` times to append `token = int(logits.argmax())` and call `decode_step(weights, cfg, token, cache)` on every iteration except the last. If your tokens match but the speed test fails, your decode step is still processing more than one token. Check the shape of the tensor passed into `q_proj`.
