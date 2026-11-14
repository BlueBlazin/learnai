# BUG: applies RoPE to the projected queries and keys before splitting them into heads, so the frequencies are spread over the full projection width instead of repeating within each head of width head_dim.
import torch

from whetstone.kit import DecoderConfig, ops


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
        q = h @ params[p + "self_attn.q_proj.weight"].T
        k = h @ params[p + "self_attn.k_proj.weight"].T
        v = heads(h @ params[p + "self_attn.v_proj.weight"].T, cfg.n_kv_heads)
        q = heads(ops.rope(q, positions, cfg.rope_base), cfg.n_heads)
        k = heads(ops.rope(k, positions, cfg.rope_base), cfg.n_kv_heads)
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
