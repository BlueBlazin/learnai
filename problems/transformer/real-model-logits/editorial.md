## The idea

Granite 4.0 350M is defined by nine architectural facts. Six live inside each block
(covered in `real-model-block`), and three form the frame around the blocks:

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

Four of the config numbers work together as a set. Granite is trained with μP
(Maximal Update Parameterization), which keeps activation scales stable across model
widths by pulling scale factors out of the weights and into fixed multipliers:
`embedding_multiplier` on the input, `residual_multiplier` on each block branch,
`attention_multiplier` on the attention scores, and `logits_scaling` on the output.
During training these act as hyperparameters, but during inference they are part of
the model function itself. Leaving any one out produces a different model.

Notice also which scaling errors survive to the output. Because the final RMSNorm
normalizes the last hidden state, a mistake that only rescales the stream at the very
end disappears. Mistakes that change the relative mix of terms do survive. The
embedding multiplier sets how strongly token identity competes with the block updates
(starting the stream 12 times larger), and the residual multiplier adjusts that ratio
again across all 56 sublayer additions.

Facts 8 and 9 are easy to miss because they concern what is absent from the file.
There is no `lm_head.weight` tensor in the checkpoint, and `tie_word_embeddings` in
`config.json` explains why. There are no float32 tensors in the file either. The kit's
`Weights` loader converts bfloat16 weights to float32 for you, whereas a loader that
keeps bfloat16 computes with only 7 mantissa bits and drifts in the logits by tenths.

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

The JAX and MLX implementations follow the same steps using `transpose(0, 2, 1, 3)`,
`repeat(..., axis=1)`, and `where(causal, scores, -inf)`. Across all three frameworks,
this matches `transformers` to about $4 \cdot 10^{-5}$ on logits of magnitude 30.

## Common mistakes

- **Omitting `embedding_multiplier`.** Without it, the blocks add updates of normal
  magnitude to a stream that is 12 times too small, so context overwhelms token
  identity from the first layer onward.
- **Omitting `logits_scaling`, or multiplying instead of dividing.** Greedy decoding
  hides this bug because positive scaling preserves the argmax. However, sampling
  temperatures are off by a factor of 4 (or 16), and perplexity is wrong.
- **Omitting the final RMSNorm.** The output head then multiplies an unnormalized
  residual stream with entries in the thousands.
- **Block-level bugs repeated across all 28 layers.** Getting the attention scale,
  residual multiplier, RoPE half-split layout, or SwiGLU gate/up split wrong (the same
  pitfalls as in `real-model-block`) compounds across every layer.
- **Hardcoding literals instead of reading `cfg`.** Constants like `12`, `0.263`, and
  `1/64` belong to this specific checkpoint. Other models on the same code path, such
  as Granite 4.2 3B with unit multipliers and an untied head, will load and silently
  produce wrong logits if those numbers are hardcoded.
- **Always tying the head, or inferring tying from `weights`.** Writing
  `weights.get("lm_head.weight", embedding)` encodes the wrong rule. Whether the head
  is tied is declared by `cfg.tied_embeddings`. A checkpoint or wrapper can contain a
  duplicate or stale `lm_head.weight` even when the model is tied.
- **Iterating layers in lexicographical order.** Sorting checkpoint tensor names or
  layer prefixes produces `0, 1, 10, 11, ..., 19, 2, 20, ...`. This passes every toy
  model with ten or fewer layers, but fails on 28 layers. The safetensors header of
  this checkpoint lists its tensors in lexicographical order, so always loop over
  `range(cfg.n_layers)`.
- **Computing in bfloat16.** Keeping the activations or weights in bfloat16 shifts the
  logits by roughly 0.4 on this model, which flips near-ties and fails the `1e-3`
  tolerance everywhere. Reduced precision is a deployment trade-off to evaluate only
  after the float32 port matches.
- **Missing the causal mask.** As in `decoder-forward`, the final position still looks
  correct without a causal mask, while earlier positions leak future tokens.

## Complexity

Let $B$ be the batch size, $T$ the sequence length, $D = \texttt{d\_model}$,
$F = \texttt{d\_ff}$, $H = \texttt{n\_heads}$, $d = \texttt{head\_dim}$, and
$V = \texttt{vocab\_size}$. Each transformer layer costs $O(B T D (D + F))$ for its
linear projections and $O(B H T^2 d)$ for attention. The output head costs
$O(B T D V)$.

On a 12-token prompt for Granite 4.0 350M, the 28 blocks take about 3 billion
multiply-adds and the output head takes about 1.2 billion, running in roughly a tenth
of a second on a laptop CPU. The head alone costs as much as ten transformer blocks,
which is why generation usually computes only the last position's logits
(`last-position-head`).

## In the wild

Every inference runtime that supports Granite encodes these same conventions in its
own format: `transformers` in `GraniteMoeHybridModel`, `mlx-lm` in its `granite`
model module, and `llama.cpp` in the GGUF metadata keys written by its converter. When
a runtime adds support for a new architecture, the pull request almost always includes
a verification table comparing a prompt's reference logits, ported logits, and maximum
error.

The next problems build directly on this forward pass. Adding a KV cache replaces
quadratic recomputation with one new row per generated token
(`real-model-cached-generate`), and quantizing the checkpoint to 8 bits is judged by
how far its logits move from these float32 values (`quantize-real-weights`).
