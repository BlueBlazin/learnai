# BUG: takes the first half of the fused projection as up and the second as gate; the gate rows come first.
import torch

from learnai.kit import RealConfig, ops


def granite_logits(
    weights: dict[str, torch.Tensor], cfg: RealConfig, tokens: torch.Tensor
) -> torch.Tensor:
    batch, seq = tokens.shape
    half = cfg.head_dim // 2
    # RoPE tables, shared by every layer: feature i of a head pairs with feature i + half.
    inv_freq = cfg.rope_theta ** (-torch.arange(half, dtype=torch.float32) / half)
    angles = torch.arange(seq, dtype=torch.float32)[:, None] * inv_freq  # (T, half)
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
    causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool))  # token t sees tokens <= t
    group = cfg.n_heads // cfg.n_kv_heads

    def rope(z: torch.Tensor) -> torch.Tensor:
        rotated = torch.cat([-z[..., half:], z[..., :half]], dim=-1)  # (a, b) -> (-b, a)
        return z * cos + rotated * sin

    def heads(z: torch.Tensor, n: int) -> torch.Tensor:
        return z.reshape(batch, seq, n, cfg.head_dim).transpose(1, 2)  # (B, n, T, head_dim)

    embedding = weights["model.embed_tokens.weight"]
    x = embedding[tokens] * cfg.embedding_multiplier  # 1. embed, times 12
    for i in range(cfg.n_layers):  # 2. the blocks, in numeric order
        p = f"model.layers.{i}."
        h = ops.rms_norm(x, weights[p + "input_layernorm.weight"], cfg.norm_eps)
        q = rope(heads(h @ weights[p + "self_attn.q_proj.weight"].T, cfg.n_heads))
        k = rope(heads(h @ weights[p + "self_attn.k_proj.weight"].T, cfg.n_kv_heads))
        v = heads(h @ weights[p + "self_attn.v_proj.weight"].T, cfg.n_kv_heads)
        k = k.repeat_interleave(group, dim=1)  # each KV head serves `group` query heads
        v = v.repeat_interleave(group, dim=1)
        scores = (q @ k.transpose(-1, -2)) * cfg.attention_scale  # 1 / head_dim, not its root
        scores = scores.masked_fill(~causal, float("-inf"))
        mixed = (ops.softmax(scores) @ v).transpose(1, 2).reshape(batch, seq, cfg.d_model)
        x = x + cfg.residual_multiplier * (mixed @ weights[p + "self_attn.o_proj.weight"].T)

        h = ops.rms_norm(x, weights[p + "post_attention_layernorm.weight"], cfg.norm_eps)
        both = h @ weights[p + "shared_mlp.input_linear.weight"].T  # gate and up in one matmul
        hidden = torch.nn.functional.silu(both[..., cfg.d_ff :]) * both[..., : cfg.d_ff]
        mlp = hidden @ weights[p + "shared_mlp.output_linear.weight"].T
        x = x + cfg.residual_multiplier * mlp
    x = ops.rms_norm(x, weights["model.norm.weight"], cfg.norm_eps)  # 3. the final norm
    # 4. A tied checkpoint has no lm_head.weight: the head is the embedding matrix.
    head = embedding if cfg.tied_embeddings else weights["lm_head.weight"]
    return (x @ head.T) / cfg.logits_scaling  # 5. logits, divided by 4
