import mlx.core as mx

from learnai.kit import HybridConfig, ops
from mlx import nn

ROOT = "model.language_model."


def delta_mixer(h: mx.array, weights: dict[str, mx.array], prefix: str, cfg: HybridConfig):
    """The Gated DeltaNet mixer of `real-gdn-block` on a whole sequence, from zero state."""
    steps = h.shape[0]
    heads, d_k, d_v = cfg.n_value_heads, cfg.key_head_dim, cfg.value_head_dim
    key_dim = heads * d_k
    projected = h @ weights[prefix + "in_proj_qkv.weight"].T
    mixed, _ = ops.causal_conv1d(projected, weights[prefix + "conv1d.weight"])
    mixed = nn.silu(mixed)
    q = ops.l2norm(mixed[:, :key_dim].reshape(steps, heads, d_k)) * d_k**-0.5
    k = ops.l2norm(mixed[:, key_dim : 2 * key_dim].reshape(steps, heads, d_k))
    v = mixed[:, 2 * key_dim :].reshape(steps, heads, d_v)
    a = h @ weights[prefix + "in_proj_a.weight"].T + weights[prefix + "dt_bias"]
    g = mx.exp(-mx.exp(weights[prefix + "A_log"]) * ops.softplus(a))
    beta = mx.sigmoid(h @ weights[prefix + "in_proj_b.weight"].T)
    memory = mx.zeros((heads, d_v, d_k), dtype=mx.float32)
    outputs = []
    for t in range(steps):  # decay, write the correction, read
        memory = memory * g[t, :, None, None]
        error = v[t] - mx.einsum("hvk,hk->hv", memory, k[t])
        memory = memory + (beta[t, :, None] * error)[:, :, None] * k[t, :, None, :]
        outputs.append(mx.einsum("hvk,hk->hv", memory, q[t]))
    z = (h @ weights[prefix + "in_proj_z.weight"].T).reshape(steps, heads, d_v)
    # The one norm weight that is a plain gain.
    y = ops.rms_norm(mx.stack(outputs), weights[prefix + "norm.weight"], cfg.norm_eps)
    return (y * nn.silu(z)).reshape(steps, heads * d_v) @ weights[prefix + "out_proj.weight"].T


def gated_attention(
    h: mx.array,
    weights: dict[str, mx.array],
    prefix: str,
    cfg: HybridConfig,
    positions: mx.array,
    causal: mx.array,
) -> mx.array:
    steps, heads, d = h.shape[0], cfg.n_heads, cfg.head_dim
    # Each head owns 2 * d consecutive outputs of q_proj: its query, then its gate.
    both = (h @ weights[prefix + "q_proj.weight"].T).reshape(steps, heads, 2 * d)
    q, gate = both[..., :d], both[..., d:]
    k = (h @ weights[prefix + "k_proj.weight"].T).reshape(steps, cfg.n_kv_heads, d)
    v = (h @ weights[prefix + "v_proj.weight"].T).reshape(steps, cfg.n_kv_heads, d)
    # QK norm over the head, stored as gain - 1 like the block norms; then heads first.
    q = ops.rms_norm(q, 1.0 + weights[prefix + "q_norm.weight"], cfg.norm_eps).swapaxes(0, 1)
    k = ops.rms_norm(k, 1.0 + weights[prefix + "k_norm.weight"], cfg.norm_eps).swapaxes(0, 1)
    v = v.swapaxes(0, 1)
    # Only the first int(d * factor) channels rotate, half-split inside them.
    q = ops.partial_rope(q, positions, cfg.rope_base, cfg.partial_rotary_factor)
    k = ops.partial_rope(k, positions, cfg.rope_base, cfg.partial_rotary_factor)
    # Grouped-query attention, scores / sqrt(d), causal.
    attended = ops.attention(q, k, v, causal).swapaxes(0, 1).reshape(steps, heads * d)
    gated = attended * mx.sigmoid(gate.reshape(steps, heads * d))
    return gated @ weights[prefix + "o_proj.weight"].T


def qwen_logits(
    weights: dict[str, mx.array], cfg: HybridConfig, tokens: mx.array
) -> mx.array:
    def norm(u: mx.array, name: str) -> mx.array:
        return ops.rms_norm(u, 1.0 + weights[name], cfg.norm_eps)  # stored as gain - 1

    steps = tokens.shape[0]
    positions = mx.arange(steps)
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
        x = x + ops.swiglu(
            h,
            weights[p + "mlp.gate_proj.weight"],
            weights[p + "mlp.up_proj.weight"],
            weights[p + "mlp.down_proj.weight"],
        )
    x = norm(x, ROOT + "norm.weight")
    return x @ table.T  # there is no lm_head tensor: the head is the embedding matrix
