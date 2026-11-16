# BUG: puts the padding condition on the query axis instead of the key axis: padding queries are blocked, but real queries still attend to padding keys.
import torch

from learnai.kit import DecoderConfig, layer_params, ops


def decoder_block(
    cfg: DecoderConfig, block: dict[str, torch.Tensor], x: torch.Tensor, mask: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """One pre-norm block of the decoder (provided; you do not need to change it).

    block: the layer's weights without the "model.layers.{i}." prefix.
    x: float32, shape (B, T, d_model). The residual stream.
    mask: bool, broadcastable to (B, n_heads, T, T). True where query position i (third
        axis) may attend to key position j (last axis). A query that may attend to
        nothing gets a zero attention output, not NaN.
    positions: integer, shape (T,) or (B, T). The RoPE position of every token.
    Returns float32, shape (B, T, d_model).
    """
    batch, seq, _ = x.shape

    def heads(t: torch.Tensor, n: int) -> torch.Tensor:
        return t.reshape(batch, seq, n, cfg.head_dim).transpose(1, 2)

    h = ops.rms_norm(x, block["input_layernorm.weight"], cfg.norm_eps)
    q = heads(h @ block["self_attn.q_proj.weight"].T, cfg.n_heads)
    k = heads(h @ block["self_attn.k_proj.weight"].T, cfg.n_kv_heads)
    v = heads(h @ block["self_attn.v_proj.weight"].T, cfg.n_kv_heads)
    q, k = ops.rope(q, positions, cfg.rope_base), ops.rope(k, positions, cfg.rope_base)
    attended = ops.attention(q, k, v, mask)
    attended = attended.transpose(1, 2).reshape(batch, seq, cfg.d_model)
    x = x + attended @ block["self_attn.o_proj.weight"].T
    h = ops.rms_norm(x, block["post_attention_layernorm.weight"], cfg.norm_eps)
    return x + ops.swiglu(
        h,
        block["mlp.gate_proj.weight"],
        block["mlp.up_proj.weight"],
        block["mlp.down_proj.weight"],
    )


def batched_forward(
    params: dict[str, torch.Tensor], cfg: DecoderConfig, seqs: list[list[int]], pad_id: int
) -> torch.Tensor:
    longest = max(len(seq) for seq in seqs)
    # Left-pad: the last real token of every row lands in the last column.
    tokens = torch.tensor([[pad_id] * (longest - len(seq)) + list(seq) for seq in seqs])
    pads = torch.tensor([longest - len(seq) for seq in seqs])[:, None]  # (B, 1)
    column = torch.arange(longest)[None, :]  # (1, T)
    # Padding is known from the lengths, never from the token values: pad_id may also
    # occur as a real token.
    real = column >= pads  # (B, T)
    causal = column[0][None, :] <= column[0][:, None]  # (T, T)
    mask = causal[None, None, :, :] & real[:, None, :, None]  # (B, 1, T, T)
    # Each row counts positions from its own first real token. (Padding columns get
    # negative positions; nothing ever attends to them.)
    positions = column - pads  # (B, T)

    x = params["model.embed_tokens.weight"][tokens]  # (B, T, d_model)
    for i in range(cfg.n_layers):
        x = decoder_block(cfg, layer_params(params, i), x, mask, positions)
    # Only the last column is needed, so normalize and project that column alone.
    x = ops.rms_norm(x[:, -1], params["model.norm.weight"], cfg.norm_eps)
    head = params["model.embed_tokens.weight" if cfg.tie_embeddings else "lm_head.weight"]
    return x @ head.T
