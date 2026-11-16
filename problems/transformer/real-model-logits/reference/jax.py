import jax
import jax.numpy as jnp

from learnai.kit import RealConfig, ops


def granite_logits(weights: dict[str, jax.Array], cfg: RealConfig, tokens: jax.Array) -> jax.Array:
    batch, seq = tokens.shape
    half = cfg.head_dim // 2
    # RoPE tables, shared by every layer: feature i of a head pairs with feature i + half.
    inv_freq = cfg.rope_theta ** (-jnp.arange(half, dtype=jnp.float32) / half)
    angles = jnp.arange(seq, dtype=jnp.float32)[:, None] * inv_freq  # (T, half)
    cos = jnp.concatenate([jnp.cos(angles), jnp.cos(angles)], axis=-1)
    sin = jnp.concatenate([jnp.sin(angles), jnp.sin(angles)], axis=-1)
    causal = jnp.tril(jnp.ones((seq, seq), dtype=bool))  # token t sees tokens <= t
    group = cfg.n_heads // cfg.n_kv_heads

    def rope(z: jax.Array) -> jax.Array:
        rotated = jnp.concatenate([-z[..., half:], z[..., :half]], axis=-1)  # (a, b) -> (-b, a)
        return z * cos + rotated * sin

    def heads(z: jax.Array, n: int) -> jax.Array:
        return z.reshape(batch, seq, n, cfg.head_dim).transpose(0, 2, 1, 3)  # (B, n, T, head_dim)

    embedding = weights["model.embed_tokens.weight"]
    x = embedding[tokens] * cfg.embedding_multiplier  # 1. embed, times 12
    for i in range(cfg.n_layers):  # 2. the blocks, in numeric order
        p = f"model.layers.{i}."
        h = ops.rms_norm(x, weights[p + "input_layernorm.weight"], cfg.norm_eps)
        q = rope(heads(h @ weights[p + "self_attn.q_proj.weight"].T, cfg.n_heads))
        k = rope(heads(h @ weights[p + "self_attn.k_proj.weight"].T, cfg.n_kv_heads))
        v = heads(h @ weights[p + "self_attn.v_proj.weight"].T, cfg.n_kv_heads)
        k = jnp.repeat(k, group, axis=1)  # each KV head serves `group` query heads
        v = jnp.repeat(v, group, axis=1)
        scores = (q @ k.transpose(0, 1, 3, 2)) * cfg.attention_scale  # 1 / head_dim, not its root
        scores = jnp.where(causal, scores, -jnp.inf)
        mixed = (ops.softmax(scores) @ v).transpose(0, 2, 1, 3).reshape(batch, seq, cfg.d_model)
        x = x + cfg.residual_multiplier * (mixed @ weights[p + "self_attn.o_proj.weight"].T)

        h = ops.rms_norm(x, weights[p + "post_attention_layernorm.weight"], cfg.norm_eps)
        both = h @ weights[p + "shared_mlp.input_linear.weight"].T  # gate and up in one matmul
        hidden = jax.nn.silu(both[..., : cfg.d_ff]) * both[..., cfg.d_ff :]
        mlp = hidden @ weights[p + "shared_mlp.output_linear.weight"].T
        x = x + cfg.residual_multiplier * mlp
    x = ops.rms_norm(x, weights["model.norm.weight"], cfg.norm_eps)  # 3. the final norm
    # 4. A tied checkpoint has no lm_head.weight: the head is the embedding matrix.
    head = embedding if cfg.tied_embeddings else weights["lm_head.weight"]
    return (x @ head.T) / cfg.logits_scaling  # 5. logits, divided by 4
