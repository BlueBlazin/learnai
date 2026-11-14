## The idea

The whole model is nine facts. Six are inside the block and were the subject of
`real-model-block`; three are in the frame around the blocks:

| # | fact | where it comes from |
|---|---|---|
| 1 | the embedding output is multiplied by 12 | `embedding_multiplier` |
| 2 | each branch is multiplied by 0.263 before the residual addition | `residual_multiplier` |
| 3 | RMSNorm with $\epsilon = 10^{-5}$ inside the square root, no bias | `rms_norm_eps`, the code |
| 4 | 16 query heads on 4 key/value heads, no biases | tensor shapes |
| 5 | RoPE, base $10^7$, half-split pairs, on queries and keys | `rope_theta`, the code |
| 6 | attention scores times $1/64$ | `attention_multiplier` |
| 7 | fused SwiGLU input matrix, gate rows first | tensor names, the code |
| 8 | final norm, then the embedding matrix as the head, then divide by 4 | `tie_word_embeddings`, `logits_scaling` |
| 9 | weights stored as bfloat16, used as float32 | the safetensors header |

Three of the config numbers form a set. Granite is trained in the μP
parameterization, which keeps activations at the same scale whatever the width by
moving scale factors out of the weights and into fixed multipliers: one on the way in
(`embedding_multiplier`), one on each branch (`residual_multiplier`), one on the
attention logits (`attention_multiplier`), and one on the way out (`logits_scaling`).
To the training code they are hyperparameters. To a port they are part of the
weights: leave one out and you are running a different function.

Notice what does **not** matter. The final norm makes the logits invariant to the
scale of the last hidden state, so a mistake that only rescales the stream at the end
is invisible; the mistakes that survive are the ones that change the *mix*: the
embedding multiplier changes how strongly the token identity competes with what the
blocks add (each block adds branches of a fixed scale to a stream that starts 12
times larger), and the residual multiplier changes it again 56 times over.

And facts 8 and 9 are about absences. There is no `lm_head.weight` in the file; the
config says why. There is no float32 in the file either, and the kit's `Weights`
converts for you; a loader that keeps the storage type computes in 8 bits of
mantissa, and the logits drift by tenths.

## Reference solution (PyTorch)

```python
def granite_logits(weights, cfg, tokens):
    batch, seq = tokens.shape
    half = cfg.head_dim // 2
    inv_freq = cfg.rope_theta ** (-torch.arange(half, dtype=torch.float32) / half)
    angles = torch.arange(seq, dtype=torch.float32)[:, None] * inv_freq
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
    causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
    group = cfg.n_heads // cfg.n_kv_heads

    def rope(z):
        rotated = torch.cat([-z[..., half:], z[..., :half]], dim=-1)
        return z * cos + rotated * sin

    def heads(z, n):
        return z.reshape(batch, seq, n, cfg.head_dim).transpose(1, 2)

    embedding = weights["model.embed_tokens.weight"]
    x = embedding[tokens] * cfg.embedding_multiplier
    for i in range(cfg.n_layers):
        p = f"model.layers.{i}."
        h = ops.rms_norm(x, weights[p + "input_layernorm.weight"], cfg.norm_eps)
        q = rope(heads(h @ weights[p + "self_attn.q_proj.weight"].T, cfg.n_heads))
        k = rope(heads(h @ weights[p + "self_attn.k_proj.weight"].T, cfg.n_kv_heads))
        v = heads(h @ weights[p + "self_attn.v_proj.weight"].T, cfg.n_kv_heads)
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        scores = (q @ k.transpose(-1, -2)) * cfg.attention_scale
        scores = scores.masked_fill(~causal, float("-inf"))
        mixed = (ops.softmax(scores) @ v).transpose(1, 2).reshape(batch, seq, cfg.d_model)
        x = x + cfg.residual_multiplier * (mixed @ weights[p + "self_attn.o_proj.weight"].T)

        h = ops.rms_norm(x, weights[p + "post_attention_layernorm.weight"], cfg.norm_eps)
        both = h @ weights[p + "shared_mlp.input_linear.weight"].T
        hidden = torch.nn.functional.silu(both[..., : cfg.d_ff]) * both[..., cfg.d_ff :]
        x = x + cfg.residual_multiplier * (
            hidden @ weights[p + "shared_mlp.output_linear.weight"].T
        )
    x = ops.rms_norm(x, weights["model.norm.weight"], cfg.norm_eps)
    head = embedding if cfg.tied_embeddings else weights["lm_head.weight"]
    return (x @ head.T) / cfg.logits_scaling
```

The JAX and MLX versions are the same lines with `transpose(0, 2, 1, 3)`,
`repeat(..., axis=1)`, and `where(causal, scores, -inf)`. This agrees with
`transformers` to about $4 \cdot 10^{-5}$ on logits of magnitude 30 in all three
frameworks.

## Common mistakes

- **No embedding multiplier.** The blocks then add branches of the usual size to a
  stream 12 times too small, so context overwhelms token identity from the first
  layer on.
- **No logits divisor**, or multiplying by it. The argmax is unchanged, which is why
  greedy decoding hides this bug. Sampling does not: every temperature is off by a
  factor of 4 (or 16), and perplexity is wrong.
- **No final norm.** The head sees a stream with entries in the thousands.
- **The attention scale, the residual multiplier, the RoPE layout, the gate and up
  halves**: the mistakes of `real-model-block`, now 28 times each.
- **Literals instead of `cfg`.** `12`, `0.263`, and `1/64` are this checkpoint's
  numbers. Granite 4.2 3B has the same code path with all multipliers at 1 and a
  separate head; a function with the 350M's numbers baked in loads it and is wrong.
- **Always tying**, or **trusting the dict**. `weights.get("lm_head.weight",
  embedding)` looks robust and encodes the wrong rule: whether the head is tied is a
  property of the model, stated in the config. A file can hold a tied model's matrix
  under both names, or a stale head left behind by a conversion script.
- **Layers in name order.** Iterating over the sorted names of the checkpoint, or
  over `sorted(set(layer prefixes))`, gives 0, 1, 10, 11, ..., 19, 2, 20, ...: correct
  for every model with at most ten layers, which includes every toy model you will
  test with. The safetensors header of this very checkpoint lists its tensors in that
  order. Loop over `range(cfg.n_layers)`.
- **Staying in bfloat16.** Casting the stream to the checkpoint's dtype, or never
  leaving it, costs about 0.4 in the logits on this model: enough to flip near-ties
  and to fail a 1e-3 comparison everywhere. Reduced precision is a deployment choice
  to make after the float32 port matches, with its own measured tolerance.
- **No causal mask**: the last position is still right, as in `decoder-forward`.

## Complexity

Per layer $O(B\,T\,D\,(D + F))$ for the matrices and $O(B\,H\,T^2 d)$ for attention;
the head is $O(B\,T\,D\,V)$. For 12 tokens of the real model that is about 3 billion
multiply-adds for the blocks and 1.2 billion for the head: a tenth of a second on a
laptop CPU. The head costs as much as ten blocks, and during generation only its last
row is needed (`last-position-head`).

## In the wild

Every runtime that supports Granite carries these facts in its own words:
`transformers` in `GraniteMoeHybridModel`, `mlx-lm` in its `granite` model file,
llama.cpp in the GGUF metadata keys its converter writes for the multipliers. When
one of them gains support for a new architecture, the pull request almost always
contains a table like the one in the constraints: a prompt, the reference logits, the
port's logits, the largest difference. The next problems build on this function: a
KV cache turns the quadratic recomputation into one new row per token
(`real-model-cached-generate`), and an 8-bit checkpoint is judged by how far its
logits move from these (`quantize-real-weights`).
