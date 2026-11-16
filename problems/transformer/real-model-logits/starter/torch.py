import torch

from learnai.kit import RealConfig, ops


def granite_logits(
    weights: dict[str, torch.Tensor], cfg: RealConfig, tokens: torch.Tensor
) -> torch.Tensor:
    """Next-token logits of a Granite 4.0 model, from its checkpoint tensors.

    weights: mapping from checkpoint names to float32 tensors, (out, in) for matrices:
        "model.embed_tokens.weight"                          (vocab_size, d_model)
        "model.layers.{i}.input_layernorm.weight"            (d_model,)
        "model.layers.{i}.self_attn.q_proj.weight"           (n_heads * head_dim, d_model)
        "model.layers.{i}.self_attn.k_proj.weight"           (n_kv_heads * head_dim, d_model)
        "model.layers.{i}.self_attn.v_proj.weight"           (n_kv_heads * head_dim, d_model)
        "model.layers.{i}.self_attn.o_proj.weight"           (d_model, n_heads * head_dim)
        "model.layers.{i}.post_attention_layernorm.weight"   (d_model,)
        "model.layers.{i}.shared_mlp.input_linear.weight"    (2 * d_ff, d_model): gate, then up
        "model.layers.{i}.shared_mlp.output_linear.weight"   (d_model, d_ff)
        "model.norm.weight"                                  (d_model,)
        "lm_head.weight"                                     (vocab_size, d_model), untied only
    cfg: n_layers, d_model, n_heads, n_kv_heads, head_dim, d_ff, vocab_size, norm_eps,
        rope_theta, attention_scale, embedding_multiplier, residual_multiplier,
        logits_scaling, tied_embeddings.
    tokens: integer, shape (B, T). Token ids in [0, vocab_size); positions are 0..T-1.
    Building blocks: ops.rms_norm(x, weight, eps), ops.softmax(x).
    Returns float32 logits, shape (B, T, vocab_size).
    """
    raise NotImplementedError
