## The idea

Two things have to be right, and they are checked separately.

**The tree is the loader.** `mlx.nn.Module` is a dict. Assigning a module to
`self.name` adds a key; a Python list of modules adds integer keys; an array is a
parameter. `tree_flatten(model.parameters())` walks it and joins the keys with dots.
If those dotted paths are the checkpoint's tensor names, `load_weights(path,
strict=True)` is the entire loader: it reads the names from the file, checks that the
two sets are equal and that every shape matches, and swaps the arrays in. So the
constructor is written by reading the checkpoint's header, not the paper:

```
model.embed_tokens.weight
model.layers.{0..27}.input_layernorm.weight
model.layers.{0..27}.self_attn.{q,k,v,o}_proj.weight
model.layers.{0..27}.post_attention_layernorm.weight
model.layers.{0..27}.shared_mlp.{input,output}_linear.weight
model.norm.weight
```

226 tensors, eight per layer plus two. There is no `lm_head.weight` because the head
is tied, and therefore the module must not have that parameter either: strict
loading fails on a parameter the file does not provide. Things that are not arrays
(the multipliers, head counts, an `nn.RoPE` instance, which has no parameters) can
be attributes anywhere without adding a name.

**The forward pass is three kernels and the config.**

- `nn.RMSNorm(d_model, eps=cfg.norm_eps)` calls `mx.fast.rms_norm`.
- `nn.RoPE(head_dim, traditional=False, base=cfg.rope_theta)` calls `mx.fast.rope`.
  `traditional=False` is the half-split layout this checkpoint uses; the default base
  is 10,000, and Granite's is 10,000,000.
- `mx.fast.scaled_dot_product_attention(q, k, v, scale=cfg.attention_scale,
  mask="causal")` does grouped-query attention itself when `k` and `v` have fewer
  heads than `q`, so the 4 key/value heads are passed as 4. The scale is whatever
  the config says: $1/64$ here, which is $1/\texttt{head\_dim}$ and not
  $1/\sqrt{\texttt{head\_dim}}$.

Everything else is the list from `real-model-logits`: the embedding times 12, both
branches times 0.263, the fused gate-and-up matrix with the gate rows first, the
embedding matrix as the head, the logits divided by 4.

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

Constructing the module allocates nothing that matters: `nn.Linear` and
`nn.Embedding` create lazy random arrays, and `load_weights` replaces them before
anything is evaluated. `mx.load` is lazy too, so the 705 MB are read when the
parameters are first evaluated.

## Which 16 bits

The checkpoint is bfloat16, and a model that computes in the dtype of its parameters
can run it as stored. Measured on this machine (M1 Max, MLX 0.32.4), against
`transformers` in float32, on logits of magnitude 25:

| dtype | parameters | largest logit error, four 12-token prompts |
|---|---|---|
| float32 | 1.31 GiB | 0.0001 |
| bfloat16 | 0.66 GiB | 0.52 |
| float16 | 0.66 GiB | 0.09 on two prompts, **NaN** on the other two |

float16 has more mantissa than bfloat16 (10 bits against 7) and is more accurate
when it works. It fails here because it has less range. In layer 2 the MLP of this
model produces one of the "massive activations" real transformers have: on the
capital prompt the gate reaches 122 and the up projection 591, and their product
passes 65,504, the largest float16. One `inf` in the residual stream is NaN one
RMSNorm later. bfloat16 keeps float32's exponent and survives, which is why `mlx-lm`
runs checkpoints in bfloat16, and why the 16-bit case on the real model is bfloat16.
The small random models stay far from the limit and run in float16 without trouble.

The dtype follows from one habit: multiply by Python numbers, not by arrays. `x *
12.0` keeps `x`'s dtype; `x * mx.array(12.0)` promotes a 16-bit `x` to float32, and
every layer after it computes in float32.

## What the fused kernels buy

On the real model in float32, with the attention kernel and with the same attention
written as `repeat`, `matmul`, `where`, `softmax`, `matmul`:

| | 12 tokens | 1024 tokens | peak memory at 1024 tokens |
|---|---|---|---|
| `mx.fast.scaled_dot_product_attention` | 14 ms | 129 ms | 1.68 GiB |
| hand-written attention | 14 ms | 157 ms | 1.88 GiB |

Nothing at prompt length 12, about 20% of the time and 200 MiB at 1024, where the
hand-written version materializes a `(16, 1024, 1024)` score matrix and the repeated
keys and values in every layer. The kernel exists for head dimensions 64, 96, 128,
192, 256, and 512; for others the same call silently expands into the hand-written
operations.

## Common mistakes

- **A tree that does not spell the names.** Putting `embed_tokens`, `layers`, and
  `norm` on the top-level module gives `layers.0...` where the file says
  `model.layers.0...`; a strict load reports all 226 tensors as unexpected.
- **`1 / sqrt(head_dim)` as the scale.** The textbook value, 8 times Granite's.
- **`nn.RoPE(head_dim)`.** The default base is 10,000. The text still reads
  well, which makes this one hard to see by eye; only the logits give it away.
- **`traditional=True`.** Rotates pairs `(2i, 2i + 1)`; the checkpoint was trained
  with pairs `(i, i + 32)`.
- **Missing multipliers**: the embedding's 12, the residual's 0.263, the logits' 4.
- **Gate and up swapped** in the fused projection.
- **A default epsilon.** `nn.RMSNorm(d)` uses `1e-5`, which happens to be Granite's;
  a config with another value gets the wrong norm.
- **Tiling the key/value heads** (`mx.tile`) before the kernel. `tile` cycles the
  heads (0, 1, 0, 1), but query head $h$ belongs to KV head $\lfloor h / g \rfloor$;
  and the kernel wants them unexpanded.
- **The embedding as the head of an untied model.** The loaded `lm_head.weight` is
  ignored.
- **No mask**, so every position sees the future.
- **Array constants** that promote a 16-bit model to float32.
- **Hand-written attention**, or importing the model from `mlx_lm`.

## Complexity

One pass over $B \cdot T$ tokens through `n_layers` blocks; attention is
$O(T^2)$ per head and layer inside the kernel. The parameters take 2 or 4 bytes
each.

## In the wild

`mlx_lm/models/granitemoehybrid.py` is this module with the model's optional parts
(mixture-of-experts and Mamba layers, which this checkpoint has none of) and a KV
cache. `mlx_lm.utils.load_model` builds it from `config.json`, calls
`model.sanitize(weights)` to drop or rename tensors that do not fit the tree, and
then `load_weights`. The next problem does the same for a 3B model whose linear
layers are stored quantized.
