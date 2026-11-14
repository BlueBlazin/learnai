# BUG: pairs feature i with feature i + head_dim / 2 (the rotate_half layout of Hugging Face checkpoints) instead of rotating the adjacent pairs (2i, 2i + 1) that this model uses.
import torch

from whetstone.kit import DecoderConfig, ops


def rope_half_split(x: torch.Tensor, positions: torch.Tensor, base: float) -> torch.Tensor:
    half = x.shape[-1] // 2
    theta = base ** (-torch.arange(half, dtype=torch.float32) / half)
    angles = positions.to(torch.float32)[:, None] * theta  # (T, head_dim / 2)
    cos, sin = torch.cos(angles), torch.sin(angles)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


def forward(params: dict[str, torch.Tensor], cfg: DecoderConfig, tokens: torch.Tensor) -> torch.Tensor:
    batch, seq = tokens.shape
    positions = torch.arange(seq)  # every row starts at position 0
    causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool))  # query t sees keys <= t

    def heads(t: torch.Tensor, n: int) -> torch.Tensor:
        # (B, T, n * head_dim) -> (B, n, T, head_dim)
        return t.reshape(batch, seq, n, cfg.head_dim).transpose(1, 2)

    x = params["model.embed_tokens.weight"][tokens]  # (B, T, d_model); no scaling
    for i in range(cfg.n_layers):
        p = f"model.layers.{i}."
        # Attention sublayer: x + attn(norm(x)).
        h = ops.rms_norm(x, params[p + "input_layernorm.weight"], cfg.norm_eps)
        q = heads(h @ params[p + "self_attn.q_proj.weight"].T, cfg.n_heads)
        k = heads(h @ params[p + "self_attn.k_proj.weight"].T, cfg.n_kv_heads)
        v = heads(h @ params[p + "self_attn.v_proj.weight"].T, cfg.n_kv_heads)
        # Position enters here and only here: rotate queries and keys, per head, not values.
        q = rope_half_split(q, positions, cfg.rope_base)
        k = rope_half_split(k, positions, cfg.rope_base)
        attended = ops.attention(q, k, v, causal)  # grouped-query: (B, n_heads, T, head_dim)
        attended = attended.transpose(1, 2).reshape(batch, seq, cfg.d_model)
        x = x + attended @ params[p + "self_attn.o_proj.weight"].T
        # MLP sublayer: x + mlp(norm(x)).
        h = ops.rms_norm(x, params[p + "post_attention_layernorm.weight"], cfg.norm_eps)
        x = x + ops.swiglu(
            h,
            params[p + "mlp.gate_proj.weight"],
            params[p + "mlp.up_proj.weight"],
            params[p + "mlp.down_proj.weight"],
        )
    # The stream was never normalized on the way, so it is normalized once before the head.
    x = ops.rms_norm(x, params["model.norm.weight"], cfg.norm_eps)
    # A tied checkpoint has no lm_head.weight: the head is the embedding matrix.
    head = params["model.embed_tokens.weight" if cfg.tie_embeddings else "lm_head.weight"]
    return x @ head.T
