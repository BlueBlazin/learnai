## The idea

Causal attention makes KV caching possible. The key and value that a layer computes for token $p$ depend only on tokens $0, \dots, p$ and never on future tokens, so they are final as soon as they are computed. Decoding a new token at position $n$ therefore needs only those saved keys and values from each layer. Everything else in the forward pass (the query, residual stream, MLP, final norm, and output head) operates solely on the single new token.

Three properties of Granite 4.0 350M determine how the cache is structured:

- **Store rotated keys.** RoPE makes the attention score between a query at position $p$ and a key at position $s$ depend only on $p - s$ by rotating each vector using its own position. Because a key's rotation depends only on $s$, rotate it once when the token is processed and store the rotated key. A new token is rotated for position $n$, the current cache length.
- **Store at KV-head width.** Grouped-query attention computes 4 key/value heads and shares each across 4 query heads. Store the 4 KV heads in the cache and expand them to 16 heads with `repeat_interleave` only when computing attention. Storing expanded heads gives the same numbers at four times the memory cost.
- **Treat the cache as an immutable value.** Concatenating new keys and values with `cat` allocates new tensors and leaves the input cache untouched, so callers can branch safely. Copying a few megabytes per step is negligible next to reading over a gigabyte of model weights.

Writing one shared `_forward` helper that processes $T$ new tokens after $n$ cached tokens handles both entry points cleanly. `prefill` calls it with $n = 0$ and $T = P$, while `decode_step` calls it with $T = 1$. The attention mask has shape `(T, n + T)`. All $n$ cached tokens are visible, and the $T$ new tokens follow a causal triangle.

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

The MLX reference is a direct translation. The JAX reference differs in one detail. Its cache stores `(keys, values, length)` inside buffers whose capacity grows in multiples of 64 slots. New entries are written with `.at[:, positions].set(...)` (which returns new arrays without mutating the old ones), and a mask hides unused slots. The math is identical, while keeping array shapes constant for 64 steps at a time avoids recompiling on every token.

## Common mistakes

- **Resetting positions to 0.** Using `arange(T)` instead of `n + arange(T)` rotates every decoded token as if it were the start of the prompt. `prefill` passes, and the first decoded token is already wrong.
- **Rotating cached keys a second time.** Applying RoPE after concatenating with the cache rotates previously cached keys again for positions they were already rotated for.
- **Storing unrotated keys.** Caching the raw `k_proj` output and rotating only the new key leaves all past keys without positional encoding.
- **Omitting the causal mask during prefill.** The last prompt token sees all earlier tokens anyway, so the initial logits may look plausible. However, earlier cached keys and values were computed from hidden states that attended to future tokens, corrupting every subsequent decode step.
- **Taking logits from row 0.** Row 0 equals the last row only when a single token is processed.
- **Loop off-by-one errors.** Re-feeding the last prompt token, returning the prompt along with generated tokens, or stopping one token short are all common loop bugs. Producing `max_new_tokens` tokens requires `prefill` followed by `max_new_tokens - 1` calls to `decode_step`.
- **Mutating the cache in place.** Preallocated buffers mutated across calls work on a single linear sequence, but fail as soon as a caller branches from an earlier cache because a second branch overwrites slots the first branch still expects.
- **Recomputing instead of caching.** Storing token IDs and re-running the entire sequence yields exact logits, but after a 768-token prompt each step costs as much as `prefill` (roughly nine times slower than a cached step in PyTorch on CPU).

## Complexity

`prefill` runs one forward pass over $P$ tokens, with $O(P^2)$ attention score computation per layer. A `decode_step` at cache length $n$ runs one token through every layer plus $O(n)$ attention work (and $O(n)$ cache copy in the concatenating version). Cache memory is $2 \cdot \texttt{n\_layers} \cdot \texttt{n\_kv\_heads} \cdot \texttt{head\_dim}$ floats per token, or 57 kB per token for Granite 4.0 350M.

On CPU, a decode step takes almost the same time at context length 12 as at context length 768. Runtime is dominated by reading the 1.4 GB of float32 model weights once per token rather than by attention over the cache. That is why caching gives only a modest speedup on short prompts (re-running 40 tokens is under twice as slow as a cached step) and why the speed test uses a 768-token prompt.

## In the wild

Production runtimes use the same core design with different memory allocation strategies. `mlx-lm`'s `KVCache` appends into buffers that grow in 256-slot chunks, matching the JAX reference approach. `transformers`' `DynamicCache` concatenates tensors like the PyTorch reference. `llama.cpp` preallocates the full context window up front, and vLLM stores the cache in fixed-size blocks shared across requests (`paged-kv-allocator`). All of them store rotated keys at KV-head width.
