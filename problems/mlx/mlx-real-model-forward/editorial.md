## The idea

This problem has two independent requirements: matching the checkpoint's parameter tree and assembling the forward pass from MLX's fused kernels.

**Matching the parameter tree.** In MLX, an `nn.Module` behaves like a nested dictionary. Assigning a submodule to `self.name` creates a string key, a Python list of submodules creates integer keys, and an array attribute becomes a parameter. `tree_flatten(model.parameters())` walks this hierarchy and joins keys with dots.

If those dotted paths match the checkpoint's tensor names, `load_weights(path, strict=True)` handles loading on its own. It verifies that the name sets and shapes match, then swaps in the checkpoint arrays. The checkpoint header defines the exact tree:

```
model.embed_tokens.weight
model.layers.{0..27}.input_layernorm.weight
model.layers.{0..27}.self_attn.{q,k,v,o}_proj.weight
model.layers.{0..27}.post_attention_layernorm.weight
model.layers.{0..27}.shared_mlp.{input,output}_linear.weight
model.norm.weight
```

For Granite 4.0 350M, that is 226 tensors (8 per layer plus the embedding and final norm). Because the embeddings are tied, `lm_head.weight` is absent from the file and must also be absent from the module, or strict loading will fail. Non-array attributes such as Python floats (multipliers, head counts) and parameter-free modules like `nn.RoPE` do not appear in `parameters()`.

**Fused kernels and config constants.** Three MLX primitives cover the core operations:

- `nn.RMSNorm(d_model, eps=cfg.norm_eps)` dispatches to `mx.fast.rms_norm`.
- `nn.RoPE(head_dim, traditional=False, base=cfg.rope_theta)` dispatches to `mx.fast.rope`. Setting `traditional=False` selects the half-split layout used by this checkpoint, and passing `base=cfg.rope_theta` overrides the default of 10,000 with Granite's 10,000,000.
- `mx.fast.scaled_dot_product_attention(q, k, v, scale=cfg.attention_scale, mask="causal")` handles grouped-query attention natively when `k` and `v` have fewer heads than `q`, so the 4 key/value heads can be passed directly. The `scale` argument takes `cfg.attention_scale` ($1/64 = 1/\texttt{head\_dim}$ here, not $1/\sqrt{\texttt{head\_dim}}$).

The remaining steps apply the config constants from `real-model-logits`: multiply embeddings by `embedding_multiplier`, multiply both residual branches by `residual_multiplier`, split the fused MLP projection with `gate` first and `up` second, reuse the embedding matrix when tied, and divide the final logits by `logits_scaling`.

## Reference solution (MLX)

```python
class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.scale = cfg.n_heads, cfg.n_kv_heads, cfg.attention_scale
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        self.rope = nn.RoPE(cfg.head_dim, traditional=False, base=cfg.rope_theta)

    def __call__(self, x):
        batch, seq, _ = x.shape

        def heads(z, n):
            return z.reshape(batch, seq, n, -1).transpose(0, 2, 1, 3)

        q = self.rope(heads(self.q_proj(x), self.n_heads))
        k = self.rope(heads(self.k_proj(x), self.n_kv_heads))
        v = heads(self.v_proj(x), self.n_kv_heads)
        mixed = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        return self.o_proj(mixed.transpose(0, 2, 1, 3).reshape(batch, seq, -1))


class SharedMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_linear = nn.Linear(cfg.d_model, 2 * cfg.d_ff, bias=False)
        self.output_linear = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def __call__(self, x):
        gate, up = mx.split(self.input_linear(x), 2, axis=-1)
        return self.output_linear(nn.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.residual = cfg.residual_multiplier
        self.input_layernorm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.shared_mlp = SharedMLP(cfg)

    def __call__(self, x):
        x = x + self.residual * self.self_attn(self.input_layernorm(x))
        return x + self.residual * self.shared_mlp(self.post_attention_layernorm(x))


class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = [Block(cfg) for _ in range(cfg.n_layers)]
        self.norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)


class Granite(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding_multiplier = cfg.embedding_multiplier
        self.logits_scaling = cfg.logits_scaling
        self.model = Backbone(cfg)
        if not cfg.tied_embeddings:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def __call__(self, tokens):
        x = self.model.embed_tokens(tokens) * self.embedding_multiplier
        for layer in self.model.layers:
            x = layer(x)
        x = self.model.norm(x)
        if "lm_head" in self:
            logits = self.lm_head(x)
        else:
            logits = self.model.embed_tokens.as_linear(x)
        return logits / self.logits_scaling
```

Constructing `Granite` does not allocate real weight memory. `nn.Linear` and `nn.Embedding` create lazy random arrays that `load_weights` replaces before evaluation, and `mx.load` reads the 705 MB file lazily when parameters are first evaluated.

## Which 16 bits

The checkpoint is stored in `bfloat16`, so a dtype-preserving module can run it directly as stored. Measured on an M1 Max (MLX 0.32.4) against `transformers` in `float32`, on logits of magnitude 25:

| dtype | parameters | largest logit error, four 12-token prompts |
|---|---|---|
| float32 | 1.31 GiB | 0.0001 |
| bfloat16 | 0.66 GiB | 0.52 |
| float16 | 0.66 GiB | 0.09 on two prompts, **NaN** on the other two |

`float16` has more mantissa bits than `bfloat16` (10 vs. 7), so it is more accurate when values stay in range. It fails on this model because of its limited dynamic range. In layer 2, the MLP produces a massive activation where the gate reaches 122 and the up projection reaches 591 on the capital prompt. Their product exceeds 65,504 (the maximum finite `float16`), producing `inf` in the residual stream and `NaN` at the next RMSNorm. `bfloat16` keeps `float32`'s exponent range and avoids overflow, which is why `mlx-lm` runs checkpoints in `bfloat16` and why the 16-bit real-model test uses `bfloat16`. The small random models stay well within `float16` range.

Preserving dtype requires multiplying by Python floats rather than MLX arrays. `x * 12.0` preserves `x`'s dtype, whereas `x * mx.array(12.0)` promotes a 16-bit `x` to `float32` for every subsequent layer.

## What the fused kernels buy

Comparing `mx.fast.scaled_dot_product_attention` against hand-written attention (`repeat`, `matmul`, `where`, `softmax`, `matmul`) on the real model in `float32`:

| | 12 tokens | 1024 tokens | peak memory at 1024 tokens |
|---|---|---|---|
| `mx.fast.scaled_dot_product_attention` | 14 ms | 129 ms | 1.68 GiB |
| hand-written attention | 14 ms | 157 ms | 1.88 GiB |

At 12 tokens the timings are identical. At 1024 tokens, the fused kernel saves about 20% runtime and 200 MiB of peak memory by avoiding the `(16, 1024, 1024)` score matrix and repeated KV tensors in every layer. The fused kernel supports head dimensions 64, 96, 128, 192, 256, and 512, and silently falls back to unfused operations for other sizes.

## Common mistakes

- **Wrong module hierarchy.** Placing `embed_tokens`, `layers`, and `norm` directly on `Granite` produces `layers.0...` instead of `model.layers.0...`, causing strict loading to reject all 226 tensors.
- **Using `1 / sqrt(head_dim)` for attention scale.** That is 8 times larger than Granite's `1 / 64`.
- **Omitting `base=cfg.rope_theta` in `nn.RoPE`.** The default base is 10,000. Text may still look plausible, but the logits diverge.
- **Using `traditional=True` in `nn.RoPE`.** That rotates adjacent pairs `(2i, 2i + 1)` instead of the half-split pairs `(i, i + head_dim // 2)` used during training.
- **Dropping multipliers.** Forgetting `embedding_multiplier` (12), `residual_multiplier` (0.263), or `logits_scaling` (4) breaks numerical agreement.
- **Swapping gate and up.** The first half of `input_linear` is the SiLU gate and the second half is the up projection.
- **Hardcoding RMSNorm epsilon.** `nn.RMSNorm`'s default `1e-5` matches Granite 4.0 350M, but fails on test configs with a different `norm_eps`.
- **Expanding KV heads with `mx.tile`.** With group size $g = \texttt{n\_heads} / \texttt{n\_kv\_heads}$, query head $h$ maps to KV head $\lfloor h / g \rfloor$, whereas `mx.tile` cycles `(0, 1, 0, 1)`. Pass the KV heads unexpanded to `scaled_dot_product_attention`.
- **Using the embedding as the head when untied.** When `cfg.tied_embeddings` is `False`, the model must project through `lm_head` instead of `embed_tokens.as_linear`.
- **Omitting `mask="causal"`.** Without the mask, tokens attend to future positions.
- **Wrapping scalar constants in `mx.array`.** A `float32` scalar array upcasts 16-bit activations to `float32`.
- **Writing unfused attention or importing `mlx_lm`.** Both violate the exercise rules and fail the graph or source checks.

## Complexity

One forward pass processes $B \cdot T$ tokens through `n_layers` blocks, with $O(T^2)$ attention work per head and layer inside the fused kernel. Parameters occupy 2 bytes each in 16-bit dtypes or 4 bytes each in `float32`.

## In the wild

`mlx_lm/models/granitemoehybrid.py` is this same module extended with KV caching and optional mixture-of-experts and Mamba layers (unused in this dense checkpoint). `mlx_lm.utils.load_model` constructs the module from `config.json`, calls `model.sanitize(weights)` to drop or rename any non-matching tensors, and then calls `load_weights`. The next problem applies the same pattern to a 3B model stored with quantized weights.
