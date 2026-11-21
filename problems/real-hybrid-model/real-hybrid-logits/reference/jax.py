import jax
import jax.numpy as jnp

from learnai.kit import HybridConfig, ops

ROOT = "model.language_model."


def linear(u: jax.Array, w: jax.Array) -> jax.Array:
    """u @ w.T for an (out, in) weight, multiplied from the weight's side. Eager JAX
    materializes `w.T`: a copy of every matrix per call, 3 GB for the real model."""
    return (w @ u.T).T


def delta_mixer(h: jax.Array, weights: dict[str, jax.Array], prefix: str, cfg: HybridConfig):
    """The Gated DeltaNet mixer of `real-gdn-block` on a whole sequence, from zero state."""
    steps = h.shape[0]
    heads, d_k, d_v = cfg.n_value_heads, cfg.key_head_dim, cfg.value_head_dim
    key_dim = heads * d_k
    projected = linear(h, weights[prefix + "in_proj_qkv.weight"])
    mixed, _ = ops.causal_conv1d(projected, weights[prefix + "conv1d.weight"])
    mixed = jax.nn.silu(mixed)
    q = ops.l2norm(mixed[:, :key_dim].reshape(steps, heads, d_k)) * d_k**-0.5
    k = ops.l2norm(mixed[:, key_dim : 2 * key_dim].reshape(steps, heads, d_k))
    v = mixed[:, 2 * key_dim :].reshape(steps, heads, d_v)
    a = linear(h, weights[prefix + "in_proj_a.weight"]) + weights[prefix + "dt_bias"]
    g = jnp.exp(-jnp.exp(weights[prefix + "A_log"]) * ops.softplus(a))
    beta = jax.nn.sigmoid(linear(h, weights[prefix + "in_proj_b.weight"]))
    memory = jnp.zeros((heads, d_v, d_k), dtype=jnp.float32)
    # Eager JAX pays for every index expression, so the per-token slices are taken once,
    # by iterating over the arrays, and keys and queries are made columns beforehand.
    g, beta = g[:, :, None, None], beta[:, :, None]
    k_row, k_col, q_col = k[:, :, None, :], k[:, :, :, None], q[:, :, :, None]
    outputs = []
    for g_t, beta_t, v_t, kr_t, kc_t, qc_t in zip(g, beta, v, k_row, k_col, q_col, strict=True):
        memory = memory * g_t  # decay
        error = v_t - jnp.squeeze(memory @ kc_t, -1)
        memory = memory + jnp.expand_dims(beta_t * error, -1) * kr_t  # write the correction
        outputs.append(jnp.squeeze(memory @ qc_t, -1))  # read
    z = linear(h, weights[prefix + "in_proj_z.weight"]).reshape(steps, heads, d_v)
    # The one norm weight that is a plain gain.
    y = ops.rms_norm(jnp.stack(outputs), weights[prefix + "norm.weight"], cfg.norm_eps)
    merged = (y * jax.nn.silu(z)).reshape(steps, heads * d_v)
    return linear(merged, weights[prefix + "out_proj.weight"])


def gated_attention(
    h: jax.Array,
    weights: dict[str, jax.Array],
    prefix: str,
    cfg: HybridConfig,
    positions: jax.Array,
    causal: jax.Array,
) -> jax.Array:
    steps, heads, d = h.shape[0], cfg.n_heads, cfg.head_dim
    # Each head owns 2 * d consecutive outputs of q_proj: its query, then its gate.
    both = linear(h, weights[prefix + "q_proj.weight"]).reshape(steps, heads, 2 * d)
    q, gate = both[..., :d], both[..., d:]
    k = linear(h, weights[prefix + "k_proj.weight"]).reshape(steps, cfg.n_kv_heads, d)
    v = linear(h, weights[prefix + "v_proj.weight"]).reshape(steps, cfg.n_kv_heads, d)
    # QK norm over the head, stored as gain - 1 like the block norms; then heads first.
    q = ops.rms_norm(q, 1.0 + weights[prefix + "q_norm.weight"], cfg.norm_eps).swapaxes(0, 1)
    k = ops.rms_norm(k, 1.0 + weights[prefix + "k_norm.weight"], cfg.norm_eps).swapaxes(0, 1)
    v = v.swapaxes(0, 1)
    # Only the first int(d * factor) channels rotate, half-split inside them.
    q = ops.partial_rope(q, positions, cfg.rope_base, cfg.partial_rotary_factor)
    k = ops.partial_rope(k, positions, cfg.rope_base, cfg.partial_rotary_factor)
    # Grouped-query attention, scores / sqrt(d), causal.
    attended = ops.attention(q, k, v, causal).swapaxes(0, 1).reshape(steps, heads * d)
    gated = attended * jax.nn.sigmoid(gate.reshape(steps, heads * d))
    return linear(gated, weights[prefix + "o_proj.weight"])


def qwen_logits(
    weights: dict[str, jax.Array], cfg: HybridConfig, tokens: jax.Array
) -> jax.Array:
    def norm(u: jax.Array, name: str) -> jax.Array:
        return ops.rms_norm(u, 1.0 + weights[name], cfg.norm_eps)  # stored as gain - 1

    steps = tokens.shape[0]
    positions = jnp.arange(steps)
    causal = positions[None, :] <= positions[:, None]  # row t sees columns 0 .. t
    table = weights[ROOT + "embed_tokens.weight"]
    x = table[tokens]  # no scale, no position embedding
    for i, kind in enumerate(cfg.layers):  # the schedule is the config's, in numeric order
        p = f"{ROOT}layers.{i}."
        h = norm(x, p + "input_layernorm.weight")
        if kind == "linear_attention":
            x = x + delta_mixer(h, weights, p + "linear_attn.", cfg)
        else:
            x = x + gated_attention(h, weights, p + "self_attn.", cfg, positions, causal)
        h = norm(x, p + "post_attention_layernorm.weight")
        # SwiGLU, written out for the same reason (ops.swiglu transposes its three weights).
        hidden = jax.nn.silu(linear(h, weights[p + "mlp.gate_proj.weight"]))
        hidden = hidden * linear(h, weights[p + "mlp.up_proj.weight"])
        x = x + linear(hidden, weights[p + "mlp.down_proj.weight"])
    x = norm(x, ROOT + "norm.weight")
    return linear(x, table)  # there is no lm_head tensor: the head is the embedding matrix
