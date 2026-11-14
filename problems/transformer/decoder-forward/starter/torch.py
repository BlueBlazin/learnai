import torch

from whetstone.kit import DecoderConfig, ops


def forward(params: dict[str, torch.Tensor], cfg: DecoderConfig, tokens: torch.Tensor) -> torch.Tensor:
    """The full forward pass of a decoder-only transformer: token ids in, logits out.

    params: checkpoint-style names to float32 tensors; linear weights are (out, in). For layer i:
        "model.embed_tokens.weight"                         (vocab_size, d_model)
        "model.layers.{i}.input_layernorm.weight"           (d_model,)
        "model.layers.{i}.self_attn.q_proj.weight"          (n_heads * head_dim, d_model)
        "model.layers.{i}.self_attn.k_proj.weight"          (n_kv_heads * head_dim, d_model)
        "model.layers.{i}.self_attn.v_proj.weight"          (n_kv_heads * head_dim, d_model)
        "model.layers.{i}.self_attn.o_proj.weight"          (d_model, n_heads * head_dim)
        "model.layers.{i}.post_attention_layernorm.weight"  (d_model,)
        "model.layers.{i}.mlp.gate_proj.weight"             (d_mlp, d_model)
        "model.layers.{i}.mlp.up_proj.weight"               (d_mlp, d_model)
        "model.layers.{i}.mlp.down_proj.weight"             (d_model, d_mlp)
        "model.norm.weight"                                 (d_model,)
        "lm_head.weight"                                    (vocab_size, d_model), only if untied
    cfg: vocab_size, d_model, n_layers, n_heads, n_kv_heads, d_mlp, rope_base, norm_eps,
        tie_embeddings, and the derived head_dim = d_model // n_heads.
    tokens: integer, shape (B, T). Token ids in [0, vocab_size).
    Building blocks: ops.rms_norm(x, weight, eps), ops.rope(x, positions, base),
    ops.attention(q, k, v, mask), ops.swiglu(x, w_gate, w_up, w_down).
    Returns float32 logits, shape (B, T, vocab_size).
    """
    raise NotImplementedError
