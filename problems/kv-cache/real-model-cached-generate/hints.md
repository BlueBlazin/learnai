Look at one attention layer while the token at position $n$ is being decoded. Which
of its inputs depend on earlier tokens? Only the keys and values it attends over. The
query, the residual stream, the MLP, the norm, and the head all belong to the new
token alone. So what does a layer have to remember, and in which form: before or
after RoPE, at the width of the 16 query heads or of the 4 KV heads?

---

Write one function that runs `T` new tokens after `n` cached ones, and build both
entry points on it: `prefill` is `n = 0`, `T = P`; `decode_step` is `T = 1`. The
differences from the forward pass of `real-model-logits` are four lines:

1. positions are `n, n + 1, ..., n + T - 1`, not `0, ..., T - 1`;
2. after rotating the new keys, append them (and the new values) to the layer's
   cached ones, without touching what was cached;
3. the mask is `(T, n + T)`: the new token at position `p` sees entries `0..p`;
4. only the last row goes through the final norm and the head.

Return the per-layer `(keys, values)` as the new cache. Build new tensors (`cat` /
`concatenate`); do not write into the ones you were given.

---

In PyTorch, inside the loop over layers, with `cache[i] = (keys, values)` of shape
`(n_kv_heads, n, head_dim)`:

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

and `generate` is `prefill`, then `token = int(logits.argmax())` and
`decode_step(weights, cfg, token, cache)` in a loop that skips the call after the last
token. If the tokens are right and the speed case fails, your step is reading
something whose size grows with the prompt: print the shape of what you pass through
`q_proj`.
