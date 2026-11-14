## The idea

The checkpoint tells you how to build the model. Its tensor names are the module
tree, as in `mlx-real-model-forward`, and its tensor *triples* are the layer type: a
layer stored as `weight` (uint32), `scales`, and `biases` is an `nn.QuantizedLinear`
(or an `nn.QuantizedEmbedding`), whose parameters have exactly those names and whose
forward pass is `mx.quantized_matmul` on the packed codes. So the whole loader is:

1. read `config.json`, including `quantization: {group_size, bits}`;
2. build the tree with quantized layers of that group size and bit width;
3. `load_weights(path, strict=True)`: names and shapes are checked, arrays swapped in;
4. convert the floating parameters to float32 and evaluate.

At no point is there a float matrix. The constructor's own initial values are lazy
and never evaluated; `mx.load` is lazy; the codes go from the file into uint32
parameters and stay there. `set_dtype(mx.float32)` touches floating parameters only,
which are the per-group scales and biases (2 of every 66 stored values) and the norm
gains.

The forward pass differs from the small model's in what the config says (multipliers
of 1, separate gate and up matrices, an untied head) and in the cache.

**The cache.** Each attention layer keeps the rotated keys and the values of
everything seen. With a past of $C$ tokens, the new queries and keys are rotated at
positions $C, C + 1, \dots$ (`rope(x, offset=C)`), appended, and the queries attend
over all $C + T$ keys. `mx.fast.scaled_dot_product_attention`'s `"causal"` mask is
aligned to the *last* key (the last query sees everything, the one before it all but
one), which is exactly right for $T$ new tokens after a prefix. A single new token
needs no mask.

**The loop** is the one of `mlx-streaming-generate`: build step $t + 1$ from the
token *array* and hand it to `mx.async_eval` before `.item()` blocks on token $t$.
An EOS can only be recognized after the read, by which time the next step is already
running; it is abandoned, which costs one forward pass per generation and is what
`mlx_lm.generate_step` does too.

## Reference solution (MLX)

```python
class Attention(nn.Module):
    def __init__(self, cfg, quant):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.scale = cfg.n_heads, cfg.n_kv_heads, cfg.attention_scale
        self.q_proj = nn.QuantizedLinear(cfg.d_model, cfg.n_heads * cfg.head_dim, False, **quant)
        self.k_proj = nn.QuantizedLinear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, False, **quant)
        self.v_proj = nn.QuantizedLinear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, False, **quant)
        self.o_proj = nn.QuantizedLinear(cfg.n_heads * cfg.head_dim, cfg.d_model, False, **quant)
        self.rope = nn.RoPE(cfg.head_dim, traditional=False, base=cfg.rope_theta)

    def __call__(self, x, past):
        batch, seq, _ = x.shape

        def heads(z, n):
            return z.reshape(batch, seq, n, -1).transpose(0, 2, 1, 3)

        offset = 0 if past is None else past[0].shape[2]
        q = self.rope(heads(self.q_proj(x), self.n_heads), offset=offset)
        k = self.rope(heads(self.k_proj(x), self.n_kv_heads), offset=offset)
        v = heads(self.v_proj(x), self.n_kv_heads)
        if past is not None:
            k = mx.concatenate([past[0], k], axis=2)
            v = mx.concatenate([past[1], v], axis=2)
        mask = "causal" if seq > 1 else None
        mixed = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.o_proj(mixed.transpose(0, 2, 1, 3).reshape(batch, seq, -1)), (k, v)


class MLP(nn.Module):
    def __init__(self, cfg, quant):
        super().__init__()
        self.gate_proj = nn.QuantizedLinear(cfg.d_model, cfg.d_ff, False, **quant)
        self.up_proj = nn.QuantizedLinear(cfg.d_model, cfg.d_ff, False, **quant)
        self.down_proj = nn.QuantizedLinear(cfg.d_ff, cfg.d_model, False, **quant)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg, quant):
        super().__init__()
        self.residual = cfg.residual_multiplier
        self.input_layernorm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.self_attn = Attention(cfg, quant)
        self.post_attention_layernorm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.mlp = MLP(cfg, quant)

    def __call__(self, x, past):
        attended, present = self.self_attn(self.input_layernorm(x), past)
        x = x + self.residual * attended
        return x + self.residual * self.mlp(self.post_attention_layernorm(x)), present


class Backbone(nn.Module):
    def __init__(self, cfg, quant):
        super().__init__()
        self.embed_tokens = nn.QuantizedEmbedding(cfg.vocab_size, cfg.d_model, **quant)
        self.layers = [Block(cfg, quant) for _ in range(cfg.n_layers)]
        self.norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)


class Granite(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        bits, group_size = cfg.quantization
        quant = {"group_size": group_size, "bits": bits}
        self.embedding_multiplier = cfg.embedding_multiplier
        self.logits_scaling = cfg.logits_scaling
        self.model = Backbone(cfg, quant)
        self.lm_head = nn.QuantizedLinear(cfg.d_model, cfg.vocab_size, False, **quant)

    def __call__(self, tokens, cache=None):
        x = self.model.embed_tokens(tokens) * self.embedding_multiplier
        new_cache = []
        for index, layer in enumerate(self.model.layers):
            x, present = layer(x, None if cache is None else cache[index])
            new_cache.append(present)
        logits = self.lm_head(self.model.norm(x)) / self.logits_scaling
        return logits, new_cache


def load_model(path):
    directory = Path(path)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    model = Granite(RealConfig.from_json(config))
    model.load_weights(str(directory / "model.safetensors"), strict=True)
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    return model


def generate(model, prompt_ids, max_new_tokens, eos_id=None):
    def step(tokens, cache):
        logits, cache = model(tokens, cache)
        return mx.argmax(logits[0, -1]), cache

    token, cache = step(mx.array([prompt_ids]), None)
    mx.async_eval(token)
    out = []
    for index in range(max_new_tokens):
        if index + 1 < max_new_tokens:
            next_token, cache = step(token.reshape(1, 1), cache)
            mx.async_eval(next_token)
        out.append(token.item())
        if out[-1] == eos_id:
            break
        if index + 1 < max_new_tokens:
            token = next_token
    return out
```

`nn.QuantizedEmbedding` dequantizes only the rows it looks up (`weight[tokens]`,
`scales[tokens]`, `biases[tokens]`), which is the one `mx.dequantize` in the graph.
The reference's logits equal `mlx-lm`'s to the last bit on all three real prompts:
the same kernels in the same order.

## What it costs to get it wrong

Measured on this machine (M1 Max, 64 GB, MLX 0.32.4) on the real checkpoint, the
12-token capital prompt:

| implementation | parameters held | peak, load + 8 tokens | speed over 48 tokens |
|---|---|---|---|
| the reference: packed codes, cache, async | 3.84 GiB | 3.97 GiB | 68.5 tok/s |
| the same with `mx.eval`, token read before the next step | 3.84 GiB | 3.97 GiB | 61.2 tok/s |
| scales and biases left in bfloat16 (as `mlx-lm` runs it) | 3.62 GiB | 3.73 GiB | 71.0 tok/s |
| no KV cache: the whole sequence per token | 3.84 GiB | 4.04 GiB | 12.4 tok/s |
| packed parameters, `mx.dequantize` in every call | 3.84 GiB | 6.36 GiB | 10.4 tok/s |
| float32 `nn.Linear` layers, dequantized at load | 13.63 GiB | 14.54 GiB | 22.7 tok/s |

The last row is the reason the problem exists. The float32 model computes the same
tokens. It holds 3.5 times the memory, and it is three times *slower*, with no
unpacking to do: a decoder step reads every weight once, so its speed is the memory
bandwidth divided by the bytes of weights, and float32 weights are four times the
bytes. Dequantizing in every call is worse still, because the float matrices are
written and then read on every token.

Loading takes 0.2 seconds when the file is in the page cache: `mx.load` maps names to
lazy arrays, and evaluating the parameters is one read of 3.9 GB.

## Common mistakes

- **Dequantizing at load** into `nn.Linear` layers: 13.6 GiB for this model, and the
  parameter names no longer match the file.
- **Dequantizing in the forward pass** (`x @ mx.dequantize(...).T`): the parameters
  look right, and every token allocates the float matrices.
- **The embedding as the head.** Right for the small Granite model, which is tied.
  This one has an `lm_head`, and the config says `tie_word_embeddings: false`.
- **Bit width and group size from habit.** A layer built for 8 bits in groups of 64
  cannot load a 4-bit checkpoint: the shapes of `weight` and `scales` differ.
- **Multipliers dropped because they are 1 here.** They are in the config; another
  Granite checkpoint has 12, 0.263, and 4.
- **`silu` on the wrong projection**: the gate is activated, the up projection
  multiplies.
- **No conversion to float32.** The model then computes in bfloat16: correct enough
  to generate the same tokens on clear-cut steps, off by up to 0.4 in the logits,
  and not what the fixture or a float32 comparison expects.
- **RoPE without the offset** on cached steps: every new token is rotated as if it
  were the first.
- **A causal mask only when there is no cache**: a multi-token continuation then
  sees its own future.
- **No cache in `generate`**, a blocking `mx.eval`, or a read before the next step
  is scheduled: right tokens, 5 times or 10% slower.
- **An EOS that does not stop**, or a step scheduled after the token at the limit.

## Complexity

Loading is one pass over the file. A decoding step is 281 quantized
multiplications over 3.66 billion codes, about 3.6 GiB of memory traffic, plus
attention over the cache. The cache grows by `2 * n_layers * n_kv_heads * head_dim`
floats per token: 41,000 for this model, 160 kB in float32.

## In the wild

`mlx_lm.utils.load_model` does these four steps for every architecture: it builds
the float module tree, calls `nn.quantize(model, group_size, bits,
class_predicate=...)` so that exactly the modules whose `scales` are in the file
become quantized, and then `load_weights`. Per-layer settings in
`config["quantization"]` allow mixed bit widths (the Granite model's
`quant_predicate` keeps `o_proj`, `down_proj`, and the head at 8 bits when the rest
is lower). The chat prompts in the tests were rendered with the model's own template,
which opens the assistant turn with `<think>`: the first generated tokens are the
model's reasoning, and with `enable_thinking=False` the template closes the block
itself and the model answers directly.
