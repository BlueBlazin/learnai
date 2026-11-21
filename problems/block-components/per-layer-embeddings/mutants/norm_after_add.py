# BUG: the norm is applied to the stream plus the branch instead of to the branch alone, so the stream itself is normalized.
import math

import torch

from learnai.kit import ops

EPS = 1e-6  # rms_norm_eps of Gemma 4, for every norm in this problem


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def ple_inputs(
    tokens: torch.Tensor,
    h0: torch.Tensor,
    table: torch.Tensor,
    w_model: torch.Tensor,
    norm_gain: torch.Tensor,
    n_layers: int,
) -> torch.Tensor:
    steps, hidden = h0.shape
    d_ple = table.shape[-1] // n_layers
    # Token part: one lookup of a packed row, layer i in columns i * d_ple .. (i + 1) * d_ple.
    token_part = (table[tokens] * math.sqrt(d_ple)).reshape(steps, n_layers, d_ple)
    # Context part: a projection of the layer-0 input, cut the same way, then normalized
    # per layer over its d_ple channels.
    context = (h0 @ w_model.T) * hidden**-0.5
    context = ops.rms_norm(context.reshape(steps, n_layers, d_ple), norm_gain, eps=EPS)
    return (context + token_part) * 2.0**-0.5


def ple_inject(
    h: torch.Tensor,
    ple_i: torch.Tensor,
    w_gate: torch.Tensor,
    w_proj: torch.Tensor,
    post_gain: torch.Tensor,
) -> torch.Tensor:
    # Down to d_ple, activation, gate by this layer's embedding, back up, norm, add.
    gated = gelu_tanh(h @ w_gate.T) * ple_i
    return ops.rms_norm(h + gated @ w_proj.T, post_gain, eps=EPS)


def ple_layer(
    params: dict[str, torch.Tensor],
    x: torch.Tensor,
    ple: torch.Tensor,
    i: int,
    mask: torch.Tensor,
) -> torch.Tensor:
    # Attention sublayer: norm in, norm out, add.
    u = ops.rms_norm(x, params["input_layernorm.weight"], eps=EPS)
    q = u @ params["self_attn.q_proj.weight"].T
    k = u @ params["self_attn.k_proj.weight"].T
    v = u @ params["self_attn.v_proj.weight"].T
    attn = ops.attention(q, k, v, mask) @ params["self_attn.o_proj.weight"].T
    a = x + ops.rms_norm(attn, params["post_attention_layernorm.weight"], eps=EPS)

    # MLP sublayer: norm in, norm out, add.
    u = ops.rms_norm(a, params["pre_feedforward_layernorm.weight"], eps=EPS)
    gate = gelu_tanh(u @ params["mlp.gate_proj.weight"].T)
    mlp = (gate * (u @ params["mlp.up_proj.weight"].T)) @ params["mlp.down_proj.weight"].T
    m = a + ops.rms_norm(mlp, params["post_feedforward_layernorm.weight"], eps=EPS)

    # The per-layer embedding enters last, after both residual adds, with layer i's slice.
    y = ple_inject(
        m,
        ple[:, i],
        params["per_layer_input_gate.weight"],
        params["per_layer_projection.weight"],
        params["post_per_layer_input_norm.weight"],
    )
    return y * params["layer_scalar"]
