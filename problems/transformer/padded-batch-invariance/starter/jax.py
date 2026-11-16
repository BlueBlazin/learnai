import jax
import jax.numpy as jnp

from learnai.kit import DecoderConfig, ops


def decoder_block(
    cfg: DecoderConfig, block: dict[str, jax.Array], x: jax.Array, mask: jax.Array, positions: jax.Array
) -> jax.Array:
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

    def heads(t: jax.Array, n: int) -> jax.Array:
        return t.reshape(batch, seq, n, cfg.head_dim).transpose(0, 2, 1, 3)

    h = ops.rms_norm(x, block["input_layernorm.weight"], cfg.norm_eps)
    q = heads(h @ block["self_attn.q_proj.weight"].T, cfg.n_heads)
    k = heads(h @ block["self_attn.k_proj.weight"].T, cfg.n_kv_heads)
    v = heads(h @ block["self_attn.v_proj.weight"].T, cfg.n_kv_heads)
    q, k = ops.rope(q, positions, cfg.rope_base), ops.rope(k, positions, cfg.rope_base)
    attended = ops.attention(q, k, v, mask)
    attended = attended.transpose(0, 2, 1, 3).reshape(batch, seq, cfg.d_model)
    x = x + attended @ block["self_attn.o_proj.weight"].T
    h = ops.rms_norm(x, block["post_attention_layernorm.weight"], cfg.norm_eps)
    return x + ops.swiglu(
        h,
        block["mlp.gate_proj.weight"],
        block["mlp.up_proj.weight"],
        block["mlp.down_proj.weight"],
    )


def batched_forward(
    params: dict[str, jax.Array], cfg: DecoderConfig, seqs: list[list[int]], pad_id: int
) -> jax.Array:
    """Next-token logits for a batch of sequences of different lengths, in one pass.

    params: the decoder's weights, keyed as in decoder-forward ("model.embed_tokens.weight",
        "model.layers.{i}.<name>", "model.norm.weight", and "lm_head.weight" if untied).
    cfg: the DecoderConfig (n_layers, norm_eps, tie_embeddings, ...).
    seqs: B lists of token ids, each of length >= 1; the lengths may differ.
    pad_id: the token id to pad with. It may also occur in seqs as a real token.
    Returns float32, shape (B, vocab_size). Row b holds the logits after the last token
    of seqs[b], equal to what the model computes for seqs[b] alone.
    """
    raise NotImplementedError
