## The idea

Causality is what makes a cache possible. The key and the value a layer computes for
the token at position $p$ depend on tokens $0..p$ and on nothing after them, so they
are final the moment they are computed. Decoding token $n$ therefore needs, from the
past, exactly those keys and values, per layer. Everything else in the forward pass
(the query, the residual stream, the MLP, the final norm, the head) is about the new
token and has one row.

Three properties of this model decide what goes into the cache.

- **Rotated keys.** RoPE makes the score of a query at position $p$ and a key at
  position $s$ depend on $p - s$ only, and it does so by rotating each of them by its
  own position. The key's rotation depends on $s$ alone, so it is done once and
  stored. The new token is rotated for position $n$, the number of tokens in the
  cache.
- **KV-head width.** Grouped-query attention computes 4 key/value heads and lets each
  serve 4 query heads. The cache stores the 4; the expansion to 16 is a `repeat` at
  the moment of use. Storing the expanded form is correct and four times the memory.
- **A value, not an object.** Appending with `cat` creates new tensors and leaves the
  old cache valid, so a caller can branch. The copy costs a few megabytes per step,
  small next to the gigabyte of weights every step reads.

With one function for "run `T` tokens after `n` cached ones", `prefill` and
`decode_step` are its two special cases, and the mask is the one of
`kv-cache-append`: `(T, n + T)`, all cached entries visible, a causal triangle over
the new ones.

## Reference solution (PyTorch)

```python
def _forward(weights, cfg, tokens, cache):
    seq = tokens.shape[0]
    past = 0 if cache is None else cache[0][0].shape[1]
    half, group = cfg.head_dim // 2, cfg.n_heads // cfg.n_kv_heads
    positions = past + torch.arange(seq)
    inv_freq = cfg.rope_theta ** (-torch.arange(half, dtype=torch.float32) / half)
    angles = positions.to(torch.float32)[:, None] * inv_freq
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
    visible = torch.arange(past + seq)[None, :] <= positions[:, None]

    def rope(z):
        return z * cos + torch.cat([-z[..., half:], z[..., :half]], dim=-1) * sin

    def heads(z, n):
        return z.reshape(seq, n, cfg.head_dim).transpose(0, 1)

    embedding = weights["model.embed_tokens.weight"]
    x = embedding[tokens] * cfg.embedding_multiplier
    new_cache = []
    for i in range(cfg.n_layers):
        p = f"model.layers.{i}."
        h = ops.rms_norm(x, weights[p + "input_layernorm.weight"], cfg.norm_eps)
        q = rope(heads(h @ weights[p + "self_attn.q_proj.weight"].T, cfg.n_heads))
        k = rope(heads(h @ weights[p + "self_attn.k_proj.weight"].T, cfg.n_kv_heads))
        v = heads(h @ weights[p + "self_attn.v_proj.weight"].T, cfg.n_kv_heads)
        if cache is not None:
            k = torch.cat([cache[i][0], k], dim=1)
            v = torch.cat([cache[i][1], v], dim=1)
        new_cache.append((k, v))
        scores = q @ k.repeat_interleave(group, dim=0).transpose(-1, -2)
        scores = (scores * cfg.attention_scale).masked_fill(~visible, float("-inf"))
        mixed = ops.softmax(scores) @ v.repeat_interleave(group, dim=0)
        mixed = mixed.transpose(0, 1).reshape(seq, cfg.d_model)
        x = x + cfg.residual_multiplier * (mixed @ weights[p + "self_attn.o_proj.weight"].T)
        h = ops.rms_norm(x, weights[p + "post_attention_layernorm.weight"], cfg.norm_eps)
        both = h @ weights[p + "shared_mlp.input_linear.weight"].T
        hidden = torch.nn.functional.silu(both[..., : cfg.d_ff]) * both[..., cfg.d_ff :]
        x = x + cfg.residual_multiplier * (
            hidden @ weights[p + "shared_mlp.output_linear.weight"].T
        )
    last = ops.rms_norm(x[-1], weights["model.norm.weight"], cfg.norm_eps)
    head = embedding if cfg.tied_embeddings else weights["lm_head.weight"]
    return (head @ last) / cfg.logits_scaling, new_cache


def prefill(weights, cfg, prompt_ids):
    return _forward(weights, cfg, prompt_ids, None)


def decode_step(weights, cfg, token, cache):
    return _forward(weights, cfg, torch.tensor([token]), cache)


def generate(weights, cfg, prompt_ids, max_new_tokens):
    logits, cache = prefill(weights, cfg, prompt_ids)
    tokens = []
    for step in range(max_new_tokens):
        token = int(logits.argmax())
        tokens.append(token)
        if step + 1 < max_new_tokens:
            logits, cache = decode_step(weights, cfg, token, cache)
    return tokens
```

The MLX reference is a translation. The JAX reference differs in one respect: its
cache is `(keys, values, length)` with buffers whose capacity is a multiple of 64
slots, written with `.at[:, positions].set(...)` (which also returns new arrays) and
read through a mask that hides the unused slots. The arithmetic is identical; the
shapes stay fixed for 64 steps at a time, so JAX compiles the step once instead of
once per token.

## Common mistakes

- **Positions restart.** `arange(T)` instead of `n + arange(T)`: every decoded token
  is rotated as if it were the first token of the text. Prefill is right, the first
  decoded token is already wrong.
- **Keys rotated again.** Applying RoPE to the concatenated keys rotates the cached
  ones a second time, for positions they were already rotated for.
- **Unrotated keys in the cache.** Caching the projection output and rotating only
  the new key means every step attends over keys without positions.
- **No causal mask in prefill.** The last row of the prompt sees everything anyway,
  so the first logits look close; but the keys and values cached for earlier tokens
  were computed from hidden states that had seen the future, and every later step
  inherits them.
- **Logits of the wrong row.** Row 0 is the last row only when one token is fed.
- **Feeding the last prompt token again**, **returning the prompt with the reply**,
  and **stopping one token short** are the three off-by-one errors of the loop. The
  loop makes `max_new_tokens` tokens from `max_new_tokens - 1` decode steps.
- **A cache written in place.** Preallocated buffers shared by every cache the
  function returned are fast and correct in a straight line, and wrong the moment a
  caller continues from an older cache: the slot it is about to read was overwritten
  by another branch.
- **No cache.** Keeping the token ids and running the whole text again gives the
  right tokens at every step. After a 768-token prompt each step costs as much as the
  prefill, about nine times a cached step in PyTorch on a CPU.

## Complexity

Prefill: one pass over $P$ tokens, $O(P^2)$ in the attention scores. A decode step at
length $n$: one token through every layer, plus $O(n)$ per layer to attend over the
cache (and to copy it, in the appending form). Memory: $2 \cdot \texttt{n\_layers}
\cdot \texttt{n\_kv\_heads} \cdot \texttt{head\_dim}$ floats per token, 57 kB for
Granite 4.0 350M.

Measured on this model, a step costs about the same for a 12-token and a 768-token
context: on a CPU it is bound by reading the 1.4 GB of weights once per token, not by
the attention. That is also why the cache pays off less than folklore says for short
prompts (re-running a 40-token text is under twice the cost of a cached step) and
why the speed case uses a long one.

## In the wild

`mlx-lm`'s `KVCache` appends into buffers that grow 256 slots at a time, the same
idea as the JAX reference; `transformers`' `DynamicCache` concatenates like the
PyTorch one; `llama.cpp` reserves the whole context up front; vLLM keeps the cache in
fixed-size blocks shared by all requests (`paged-kv-allocator`). All of them store
rotated keys at KV-head width.
