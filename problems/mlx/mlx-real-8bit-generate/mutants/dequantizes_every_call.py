# BUG: keeps the packed codes but every linear layer rebuilds its float matrix with mx.dequantize on each call and multiplies by it; the logits are right, and every token allocates and reads the float weights of the whole model.
import json
from pathlib import Path

import mlx.core as mx

from learnai.kit import RealConfig
from mlx import nn

# Every linear layer and the embedding is an nn.QuantizedLinear / nn.QuantizedEmbedding:
# its parameters are `weight` (packed uint32 codes), `scales`, and `biases`, the three
# tensors the checkpoint stores for it, and its forward pass is mx.quantized_matmul on
# the packed codes. The float matrix never exists.


class FloatingLinear(nn.QuantizedLinear):
    def __call__(self, x: mx.array) -> mx.array:
        weight = mx.dequantize(self.weight, self.scales, self.biases, self.group_size, self.bits)
        return x @ weight.T


class Attention(nn.Module):
    def __init__(self, cfg: RealConfig, quant: dict):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.scale = cfg.n_heads, cfg.n_kv_heads, cfg.attention_scale
        self.q_proj = FloatingLinear(cfg.d_model, cfg.n_heads * cfg.head_dim, False, **quant)
        self.k_proj = FloatingLinear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, False, **quant)
        self.v_proj = FloatingLinear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, False, **quant)
        self.o_proj = FloatingLinear(cfg.n_heads * cfg.head_dim, cfg.d_model, False, **quant)
        self.rope = nn.RoPE(cfg.head_dim, traditional=False, base=cfg.rope_theta)

    def __call__(self, x: mx.array, past: tuple | None) -> tuple[mx.array, tuple]:
        batch, seq, _ = x.shape

        def heads(z: mx.array, n: int) -> mx.array:
            return z.reshape(batch, seq, n, -1).transpose(0, 2, 1, 3)

        # New tokens continue at the position after the cached ones.
        offset = 0 if past is None else past[0].shape[2]
        q = self.rope(heads(self.q_proj(x), self.n_heads), offset=offset)
        k = self.rope(heads(self.k_proj(x), self.n_kv_heads), offset=offset)
        v = heads(self.v_proj(x), self.n_kv_heads)
        if past is not None:
            k = mx.concatenate([past[0], k], axis=2)
            v = mx.concatenate([past[1], v], axis=2)
        # "causal" is lower-right aligned: the last query sees every key, the one before
        # it all but the last. A single new token needs no mask at all.
        mask = "causal" if seq > 1 else None
        mixed = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.o_proj(mixed.transpose(0, 2, 1, 3).reshape(batch, seq, -1)), (k, v)


class MLP(nn.Module):
    def __init__(self, cfg: RealConfig, quant: dict):
        super().__init__()
        self.gate_proj = FloatingLinear(cfg.d_model, cfg.d_ff, False, **quant)
        self.up_proj = FloatingLinear(cfg.d_model, cfg.d_ff, False, **quant)
        self.down_proj = FloatingLinear(cfg.d_ff, cfg.d_model, False, **quant)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: RealConfig, quant: dict):
        super().__init__()
        self.residual = cfg.residual_multiplier
        self.input_layernorm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.self_attn = Attention(cfg, quant)
        self.post_attention_layernorm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.mlp = MLP(cfg, quant)

    def __call__(self, x: mx.array, past: tuple | None) -> tuple[mx.array, tuple]:
        attended, present = self.self_attn(self.input_layernorm(x), past)
        x = x + self.residual * attended
        return x + self.residual * self.mlp(self.post_attention_layernorm(x)), present


class Backbone(nn.Module):
    def __init__(self, cfg: RealConfig, quant: dict):
        super().__init__()
        self.embed_tokens = nn.QuantizedEmbedding(cfg.vocab_size, cfg.d_model, **quant)
        self.layers = [Block(cfg, quant) for _ in range(cfg.n_layers)]
        self.norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)


class Granite(nn.Module):
    def __init__(self, cfg: RealConfig):
        super().__init__()
        bits, group_size = cfg.quantization
        quant = {"group_size": group_size, "bits": bits}
        self.embedding_multiplier = cfg.embedding_multiplier
        self.logits_scaling = cfg.logits_scaling
        self.model = Backbone(cfg, quant)
        self.lm_head = FloatingLinear(cfg.d_model, cfg.vocab_size, False, **quant)

    def __call__(self, tokens: mx.array, cache: list | None = None) -> tuple[mx.array, list]:
        x = self.model.embed_tokens(tokens) * self.embedding_multiplier
        new_cache = []
        for index, layer in enumerate(self.model.layers):
            x, present = layer(x, None if cache is None else cache[index])
            new_cache.append(present)
        logits = self.lm_head(self.model.norm(x)) / self.logits_scaling
        return logits, new_cache


def load_model(path: str) -> nn.Module:
    directory = Path(path)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    model = Granite(RealConfig.from_json(config))
    # mx.load is lazy: names and shapes now, bytes when the parameters are evaluated.
    # The packed codes go in as uint32 and stay uint32.
    model.load_weights(str(directory / "model.safetensors"), strict=True)
    # Scales, biases, and norm gains are bfloat16 in the file. Converting them (and
    # nothing else: uint32 is not a floating type) makes the model compute in float32.
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    return model


def generate(
    model, prompt_ids: list[int], max_new_tokens: int, eos_id: int | None = None
) -> list[int]:
    def step(tokens: mx.array, cache: list | None) -> tuple[mx.array, list]:
        logits, cache = model(tokens, cache)
        return mx.argmax(logits[0, -1]), cache

    token, cache = step(mx.array([prompt_ids]), None)
    mx.async_eval(token)  # the cache is an ancestor of the token
    out: list[int] = []
    for index in range(max_new_tokens):
        if index + 1 < max_new_tokens:
            # Build and schedule the next step from the token array before reading it.
            next_token, cache = step(token.reshape(1, 1), cache)
            mx.async_eval(next_token)
        out.append(token.item())  # waits for this step only; the next one is running
        if out[-1] == eos_id:
            break  # the step scheduled above is abandoned
        if index + 1 < max_new_tokens:
            token = next_token
    return out
