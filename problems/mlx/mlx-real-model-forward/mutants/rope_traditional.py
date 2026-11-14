# BUG: uses the traditional RoPE layout, which rotates adjacent feature pairs (2i, 2i + 1); the checkpoint was trained with the half-split layout (i, i + head_dim / 2).
import mlx.core as mx

from mlx import nn
from whetstone.kit import RealConfig

# Attribute names are checkpoint names: `model.layers.3.self_attn.q_proj.weight` in the
# file is `self.model.layers[3].self_attn.q_proj.weight` here, so `load_weights` needs
# no renaming. Python floats and ints (the multipliers, the head counts) are plain
# attributes, not parameters, and they do not change the dtype of what they multiply.


class Attention(nn.Module):
    def __init__(self, cfg: RealConfig):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.scale = cfg.n_heads, cfg.n_kv_heads, cfg.attention_scale
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        # Half-split pairs (feature i with i + head_dim / 2), base from the config.
        self.rope = nn.RoPE(cfg.head_dim, traditional=True, base=cfg.rope_theta)

    def __call__(self, x: mx.array) -> mx.array:
        batch, seq, _ = x.shape

        def heads(z: mx.array, n: int) -> mx.array:
            return z.reshape(batch, seq, n, -1).transpose(0, 2, 1, 3)  # (B, n, T, head_dim)

        q = self.rope(heads(self.q_proj(x), self.n_heads))
        k = self.rope(heads(self.k_proj(x), self.n_kv_heads))
        v = heads(self.v_proj(x), self.n_kv_heads)
        # The fused kernel does grouped-query attention itself: 4 KV heads are passed
        # as 4, not repeated to 16. The scale is 1 / head_dim, straight from the config.
        mixed = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        return self.o_proj(mixed.transpose(0, 2, 1, 3).reshape(batch, seq, -1))


class SharedMLP(nn.Module):
    def __init__(self, cfg: RealConfig):
        super().__init__()
        self.input_linear = nn.Linear(cfg.d_model, 2 * cfg.d_ff, bias=False)
        self.output_linear = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = mx.split(self.input_linear(x), 2, axis=-1)  # gate first, then up
        return self.output_linear(nn.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg: RealConfig):
        super().__init__()
        self.residual = cfg.residual_multiplier
        self.input_layernorm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.shared_mlp = SharedMLP(cfg)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.residual * self.self_attn(self.input_layernorm(x))
        return x + self.residual * self.shared_mlp(self.post_attention_layernorm(x))


class Backbone(nn.Module):
    def __init__(self, cfg: RealConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = [Block(cfg) for _ in range(cfg.n_layers)]
        self.norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)


class Granite(nn.Module):
    def __init__(self, cfg: RealConfig):
        super().__init__()
        self.embedding_multiplier = cfg.embedding_multiplier
        self.logits_scaling = cfg.logits_scaling
        self.model = Backbone(cfg)
        if not cfg.tied_embeddings:
            # Only an untied checkpoint has `lm_head.weight`; a tied model must not
            # declare the parameter, or a strict load fails.
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def __call__(self, tokens: mx.array) -> mx.array:
        x = self.model.embed_tokens(tokens) * self.embedding_multiplier
        for layer in self.model.layers:
            x = layer(x)
        x = self.model.norm(x)
        if "lm_head" in self:
            logits = self.lm_head(x)
        else:
            logits = self.model.embed_tokens.as_linear(x)  # the embedding matrix is the head
        return logits / self.logits_scaling
