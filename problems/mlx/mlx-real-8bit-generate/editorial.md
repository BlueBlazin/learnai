## The idea

As in `mlx-real-model-forward`, the checkpoint's tensor names define the module tree. Here, the tensor triples also define the layer type. Any layer stored as `weight` (`uint32`), `scales`, and `biases` corresponds to `nn.QuantizedLinear` or `nn.QuantizedEmbedding`, which expose those exact parameter names and multiply directly from packed codes via `mx.quantized_matmul`.

Loading therefore takes four steps:

1. Read `config.json`, including `quantization` (`bits` and `group_size`).
2. Build the module tree using `nn.QuantizedLinear` and `nn.QuantizedEmbedding` with that bit width and group size.
3. Call `load_weights(path, strict=True)` to verify names and shapes and bind the checkpoint arrays.
4. Convert floating parameters to `float32` with `set_dtype(mx.float32)` and evaluate `model.parameters()`.

No dense float weight matrix is ever materialized. The initial weights created by the constructors are lazy and discarded before evaluation, and `mx.load` is lazy as well. `set_dtype(mx.float32)` converts only floating parameters (`scales`, `biases`, and norm gains), leaving the `uint32` packed codes untouched.

Compared to Granite 4.0 350M, the forward pass adds a KV cache, uses separate `gate_proj` and `up_proj` matrices, and always projects through `lm_head`.

**KV cache.** Each attention layer stores the rotated keys and values from previous calls. When the cache already holds $C$ tokens and the current call receives $T$ new tokens, rotate the new queries and keys with `rope(x, offset=C)`, concatenate the new keys and values onto the cached ones, and attend over all $C + T$ positions. `mx.fast.scaled_dot_product_attention`'s `"causal"` mask aligns to the last key (the last query sees all keys, the second-to-last sees all but one), which handles any $T > 1$ continuation automatically. For a single token ($T = 1$), no mask is needed.

**Asynchronous generation loop.** As in `mlx-streaming-generate`, pass the unevaluated token array into step $t + 1$ and call `mx.async_eval` before calling `.item()` on token $t$. Because `eos_id` is only known after `.item()` returns, an early stop leaves one already-scheduled step unused. Discarding that extra step matches `mlx_lm.generate_step`.

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

`nn.QuantizedEmbedding` dequantizes only the rows indexed by `tokens` (`weight[tokens]`, `scales[tokens]`, `biases[tokens]`), which accounts for the single `mx.dequantize` in the graph. Because this implementation invokes the same kernels in the same order as `mlx-lm`, its logits match `mlx-lm` bit-for-bit on all three real prompts.

## What it costs to get it wrong

Measured on an M1 Max (64 GB, MLX 0.32.4) on the real checkpoint using the 12-token capital prompt:

| implementation | parameters held | peak, load + 8 tokens | speed over 48 tokens |
|---|---|---|---|
| the reference: packed codes, cache, async | 3.84 GiB | 3.97 GiB | 68.5 tok/s |
| the same with `mx.eval`, token read before the next step | 3.84 GiB | 3.97 GiB | 61.2 tok/s |
| scales and biases left in bfloat16 (as `mlx-lm` runs it) | 3.62 GiB | 3.73 GiB | 71.0 tok/s |
| no KV cache: the whole sequence per token | 3.84 GiB | 4.04 GiB | 12.4 tok/s |
| packed parameters, `mx.dequantize` in every call | 3.84 GiB | 6.36 GiB | 10.4 tok/s |
| float32 `nn.Linear` layers, dequantized at load | 13.63 GiB | 14.54 GiB | 22.7 tok/s |

Dequantizing into `float32` at load time produces identical tokens while using 3.5 times more memory and running 3 times slower. Each decoding step is memory-bandwidth bound and reads every weight once, so moving 4 times as many weight bytes directly cuts throughput. Calling `mx.dequantize` inside each forward pass is even slower because it writes and re-reads every float matrix on every token.

When the checkpoint is in the OS page cache, loading takes about 0.2 seconds. `mx.load` creates lazy array mappings, and `mx.eval(model.parameters())` streams the 3.9 GB file in one pass.

## Common mistakes

- **Dequantizing at load into `nn.Linear`.** Uses 13.6 GiB for the 3B model and breaks parameter name matching with the checkpoint.
- **Dequantizing inside the forward pass (`x @ mx.dequantize(...).T`).** Parameter names match, but every token allocates and writes full float matrices.
- **Using the embedding as the output head.** Granite 4.2 3B has `tie_word_embeddings: false` and an explicit `lm_head`.
- **Hardcoding `bits=8` and `group_size=64`.** Fails on checkpoints with other quantization settings (such as 4-bit, group size 32) because `weight` and `scales` shapes change.
- **Dropping config multipliers.** Even though Granite 4.2 3B sets them to 1.0, other test configs use non-unit values.
- **Applying `silu` to `up_proj` instead of `gate_proj`.** The gate branch is activated with SiLU and then multiplied by `up_proj(x)`.
- **Skipping `set_dtype(mx.float32)`.** Leaving scales, biases, and norms in `bfloat16` shifts logits by up to 0.4 and fails the `float32` tolerance check.
- **Forgetting `offset` in `nn.RoPE` on cached steps.** Without `offset=C`, every cached step rotates new tokens as position 0.
- **Applying `mask="causal"` only when `cache is None`.** A multi-token continuation (`seq > 1`) with a non-empty cache then attends to future tokens.
- **Skipping the KV cache, calling `mx.eval`, or reading before scheduling.** Produces the right tokens while running 10% to 5x slower and failing the `EvalSpy` checks.
- **Missing the `eos_id` stop or scheduling past `max_new_tokens`.** Violates the exact call-count and evaluation-order checks.

## Complexity

Loading reads the checkpoint file once. Each decoding step runs 281 quantized matrix multiplications across 3.66 billion packed weights (about 3.6 GiB of weight traffic) plus attention over the cached keys and values. The KV cache grows by `2 * n_layers * n_kv_heads * head_dim` floats per token (40,960 floats, or 160 KB in `float32`, for Granite 4.2 3B).

## In the wild

`mlx_lm.utils.load_model` follows the same four steps across architectures. It builds the float module tree, calls `nn.quantize(model, group_size, bits, class_predicate=...)` so only layers with `scales` in the file are quantized, and then calls `load_weights`. Per-layer entries in `config["quantization"]` also support mixed bit widths (for example, keeping `o_proj`, `down_proj`, and `lm_head` at 8 bits when other layers use lower precision). The chat prompts in the tests use Granite's chat template, which opens the assistant turn with `<think>` so the first tokens are reasoning tokens. Setting `enable_thinking=False` closes the block immediately so the model answers directly.
