# BUG: takes the first half of the fused projection as up and the second as gate; the gate rows come first.
import torch

from whetstone.kit import ops

HEAD_DIM = 64
NORM_EPS = 1e-5
ROPE_THETA = 1e7
ATTENTION_SCALE = 1 / 64  # 1 / head_dim, not 1 / sqrt(head_dim)
RESIDUAL = 0.263  # each branch is scaled by this before it joins the stream


def rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Rotate (B, heads, T, 64) by position; feature i pairs with feature i + 32."""
    half = HEAD_DIM // 2
    inv_freq = ROPE_THETA ** (-torch.arange(half, dtype=torch.float32) / half)
    angles = positions.to(torch.float32)[:, None] * inv_freq  # (T, 32)
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)  # the same angle for both halves
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)  # (a, b) -> (-b, a)
    return x * cos + rotated * sin


def granite_block(
    x: torch.Tensor, weights: dict[str, torch.Tensor], layer: int, positions: torch.Tensor
) -> torch.Tensor:
    p = f"model.layers.{layer}."  # the trailing dot keeps layer 1 apart from layers 10 to 19
    batch, seq, width = x.shape

    def heads(h: torch.Tensor, name: str) -> torch.Tensor:
        # (B, T, D) -> (B, n, T, 64); n comes from the matrix: 16 for q, 4 for k and v
        out = h @ weights[p + f"self_attn.{name}.weight"].T
        return out.reshape(batch, seq, -1, HEAD_DIM).transpose(1, 2)

    # Attention branch: x + 0.263 * attn(norm(x)).
    h = ops.rms_norm(x, weights[p + "input_layernorm.weight"], NORM_EPS)
    q = rope(heads(h, "q_proj"), positions)
    k = rope(heads(h, "k_proj"), positions)
    v = heads(h, "v_proj")  # values are not rotated
    group = q.shape[1] // k.shape[1]  # query heads per KV head
    k = k.repeat_interleave(group, dim=1)  # KV head j serves query heads j*g .. j*g + g - 1
    v = v.repeat_interleave(group, dim=1)
    scores = (q @ k.transpose(-1, -2)) * ATTENTION_SCALE
    causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool))  # token t sees tokens <= t
    scores = scores.masked_fill(~causal, float("-inf"))
    mixed = ops.softmax(scores) @ v  # (B, H, T, 64)
    mixed = mixed.transpose(1, 2).reshape(batch, seq, width)
    x = x + RESIDUAL * (mixed @ weights[p + "self_attn.o_proj.weight"].T)

    # MLP branch: x + 0.263 * mlp(norm(x)). One matrix computes gate and up together.
    h = ops.rms_norm(x, weights[p + "post_attention_layernorm.weight"], NORM_EPS)
    both = h @ weights[p + "shared_mlp.input_linear.weight"].T  # (B, T, 2 * F)
    d_ff = both.shape[-1] // 2
    up, gate = both[..., :d_ff], both[..., d_ff:]
    hidden = torch.nn.functional.silu(gate) * up
    return x + RESIDUAL * (hidden @ weights[p + "shared_mlp.output_linear.weight"].T)
