# BUG: passes up_proj where SwiGLU expects gate_proj and the reverse, so the activation is applied to the wrong half: down(silu(up(x)) * gate(x)).
import torch
import torch.nn.functional as F

from learnai.kit import ops

NORM_EPS = 1e-6


def delta_mixer(
    h: torch.Tensor,
    weights: dict[str, torch.Tensor],
    prefix: str,
    state: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """The Gated DeltaNet mixer (`linear_attn`) on the normed rows `h`, (T, D)."""
    steps = h.shape[0]
    a_log, gain = weights[prefix + "A_log"], weights[prefix + "norm.weight"]
    heads, d_v = a_log.shape[0], gain.shape[0]  # as many key heads as value heads
    value_dim = heads * d_v
    key_dim = (weights[prefix + "in_proj_qkv.weight"].shape[0] - value_dim) // 2
    d_k = key_dim // heads
    conv_state, memory = (None, None) if state is None else state
    if memory is None:
        memory = torch.zeros(heads, d_v, d_k)

    # Project, mix each channel with its last K - 1 projected inputs, SiLU, split.
    projected = h @ weights[prefix + "in_proj_qkv.weight"].T
    mixed, conv_state = ops.causal_conv1d(projected, weights[prefix + "conv1d.weight"], conv_state)
    mixed = F.silu(mixed)
    q = ops.l2norm(mixed[:, :key_dim].reshape(steps, heads, d_k)) * d_k**-0.5
    k = ops.l2norm(mixed[:, key_dim : 2 * key_dim].reshape(steps, heads, d_k))
    v = mixed[:, 2 * key_dim :].reshape(steps, heads, d_v)

    # Gates per token and head: the decay g in (0, 1) and the write strength beta.
    a = h @ weights[prefix + "in_proj_a.weight"].T + weights[prefix + "dt_bias"]
    g = torch.exp(-torch.exp(a_log) * ops.softplus(a))
    beta = torch.sigmoid(h @ weights[prefix + "in_proj_b.weight"].T)

    # The gated delta rule, token by token: decay, write the correction, read.
    outputs = []
    for t in range(steps):
        memory = memory * g[t, :, None, None]
        error = v[t] - torch.einsum("hvk,hk->hv", memory, k[t])
        memory = memory + (beta[t, :, None] * error)[:, :, None] * k[t, :, None, :]
        outputs.append(torch.einsum("hvk,hk->hv", memory, q[t]))
    y = torch.stack(outputs)  # (T, heads, d_v)

    # The gated norm: this weight is a plain gain, used as stored. Norm, then gate.
    z = (h @ weights[prefix + "in_proj_z.weight"].T).reshape(steps, heads, d_v)
    y = ops.rms_norm(y, gain, NORM_EPS) * F.silu(z)
    out = y.reshape(steps, value_dim) @ weights[prefix + "out_proj.weight"].T
    return out, (conv_state, memory)


def qwen_gdn_block(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    layer: int,
    state: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    p = f"model.language_model.layers.{layer}."  # the dot keeps layer 1 apart from 10 to 19
    # The block norms store gain - 1: add the one back. Epsilon is 1e-6, not ops' default.
    h = ops.rms_norm(x, 1.0 + weights[p + "input_layernorm.weight"], NORM_EPS)
    mixed, state = delta_mixer(h, weights, p + "linear_attn.", state)
    x = x + mixed
    h = ops.rms_norm(x, 1.0 + weights[p + "post_attention_layernorm.weight"], NORM_EPS)
    x = x + ops.swiglu(
        h,
        weights[p + "mlp.up_proj.weight"],
        weights[p + "mlp.gate_proj.weight"],
        weights[p + "mlp.down_proj.weight"],
    )
    return x, state
