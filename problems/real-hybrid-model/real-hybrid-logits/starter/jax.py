import jax
import jax.numpy as jnp

from learnai.kit import HybridConfig, ops


def qwen_logits(weights: dict[str, jax.Array], cfg: HybridConfig, tokens: jax.Array) -> jax.Array:
    """Next-token logits of a Qwen3.5 text decoder, from its checkpoint tensors.

    weights: mapping from checkpoint names to float32 tensors, (out, in) for matrices.
      Under "model.language_model.":
        "embed_tokens.weight"   (vocab_size, d_model)   also the output head
        "norm.weight"           (d_model,)              stored as gain - 1
      and with p = "model.language_model.layers.{i}.", for every layer
        p + "input_layernorm.weight", p + "post_attention_layernorm.weight"   (d_model,)  gain - 1
        p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"   (d_mlp, d_model)
        p + "mlp.down_proj.weight"                             (d_model, d_mlp)
      for a "linear_attention" layer the nine p + "linear_attn." tensors of real-gdn-block
        (in_proj_qkv, conv1d, in_proj_a, in_proj_b, A_log, dt_bias, in_proj_z, norm, out_proj),
      for a "full_attention" layer
        p + "self_attn.q_proj.weight"   (n_heads * 2 * head_dim, d_model)   per head: query, gate
        p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"   (n_kv_heads * head_dim, d_model)
        p + "self_attn.o_proj.weight"   (d_model, n_heads * head_dim)
        p + "self_attn.q_norm.weight", p + "self_attn.k_norm.weight"   (head_dim,)   gain - 1
      The mapping may hold other tensors; a text decoder does not read them.
    cfg: layers (the mixer of each layer), n_layers, d_model, vocab_size, d_mlp, norm_eps,
      n_heads, n_kv_heads, head_dim, rope_base, partial_rotary_factor, rotary_dim,
      n_key_heads, n_value_heads, key_head_dim, value_head_dim, conv_kernel.
    tokens: integer, shape (T,). Token ids in [0, vocab_size); positions are 0 .. T - 1.
    Building blocks: ops.rms_norm(x, weight, eps), ops.swiglu(x, w_gate, w_up, w_down),
      ops.partial_rope(x, positions, base, factor), ops.attention(q, k, v, mask),
      ops.l2norm(x), ops.softplus(x), ops.causal_conv1d(x, w, state).
    Returns float32 logits, shape (T, vocab_size).
    """
    raise NotImplementedError
