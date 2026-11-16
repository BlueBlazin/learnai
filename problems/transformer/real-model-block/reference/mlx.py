import mlx.core as mx

from learnai.kit import ops
from mlx import nn

HEAD_DIM = 64
NORM_EPS = 1e-5
ROPE_THETA = 1e7
ATTENTION_SCALE = 1 / 64  # 1 / head_dim, not 1 / sqrt(head_dim)
RESIDUAL = 0.263  # each branch is scaled by this before it joins the stream


def rope(x: mx.array, positions: mx.array) -> mx.array:
    """Rotate (B, heads, T, 64) by position; feature i pairs with feature i + 32."""
    half = HEAD_DIM // 2
    inv_freq = ROPE_THETA ** (-mx.arange(half).astype(mx.float32) / half)
    angles = positions.astype(mx.float32)[:, None] * inv_freq  # (T, 32)
    cos = mx.concatenate([mx.cos(angles), mx.cos(angles)], axis=-1)  # same angle, both halves
    sin = mx.concatenate([mx.sin(angles), mx.sin(angles)], axis=-1)
    rotated = mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)  # (a, b) -> (-b, a)
    return x * cos + rotated * sin


def granite_block(
    x: mx.array, weights: dict[str, mx.array], layer: int, positions: mx.array
) -> mx.array:
    p = f"model.layers.{layer}."  # the trailing dot keeps layer 1 apart from layers 10 to 19
    batch, seq, width = x.shape

    def heads(h: mx.array, name: str) -> mx.array:
        # (B, T, D) -> (B, n, T, 64); n comes from the matrix: 16 for q, 4 for k and v
        out = h @ weights[p + f"self_attn.{name}.weight"].T
        return out.reshape(batch, seq, -1, HEAD_DIM).transpose(0, 2, 1, 3)

    # Attention branch: x + 0.263 * attn(norm(x)).
    h = ops.rms_norm(x, weights[p + "input_layernorm.weight"], NORM_EPS)
    q = rope(heads(h, "q_proj"), positions)
    k = rope(heads(h, "k_proj"), positions)
    v = heads(h, "v_proj")  # values are not rotated
    group = q.shape[1] // k.shape[1]  # query heads per KV head
    k = mx.repeat(k, group, axis=1)  # KV head j serves query heads j*g .. j*g + g - 1
    v = mx.repeat(v, group, axis=1)
    scores = (q @ k.transpose(0, 1, 3, 2)) * ATTENTION_SCALE
    causal = mx.tril(mx.ones((seq, seq))) > 0  # token t sees tokens <= t
    scores = mx.where(causal, scores, -mx.inf)
    mixed = ops.softmax(scores) @ v  # (B, H, T, 64)
    mixed = mixed.transpose(0, 2, 1, 3).reshape(batch, seq, width)
    x = x + RESIDUAL * (mixed @ weights[p + "self_attn.o_proj.weight"].T)

    # MLP branch: x + 0.263 * mlp(norm(x)). One matrix computes gate and up together.
    h = ops.rms_norm(x, weights[p + "post_attention_layernorm.weight"], NORM_EPS)
    both = h @ weights[p + "shared_mlp.input_linear.weight"].T  # (B, T, 2 * F)
    d_ff = both.shape[-1] // 2
    gate, up = both[..., :d_ff], both[..., d_ff:]  # gate first
    hidden = nn.silu(gate) * up
    return x + RESIDUAL * (hidden @ weights[p + "shared_mlp.output_linear.weight"].T)
