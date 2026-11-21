# BUG: the output gate is silu(gate), the gate of the delta layer's norm, instead of sigmoid(gate).
import torch
import torch.nn.functional as F

from learnai.kit import HybridConfig, ops

DELTA = (
    "in_proj_qkv.weight", "in_proj_z.weight", "in_proj_a.weight", "in_proj_b.weight",
    "conv1d.weight", "A_log", "dt_bias", "norm.weight", "out_proj.weight",
)  # fmt: skip


def layer_types(n_layers: int, interval: int) -> list[str]:
    # Layers are counted from 1 here: the last layer of every group of `interval` attends.
    return [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(n_layers)
    ]


def gated_attention(
    params: dict[str, torch.Tensor],
    prefix: str,
    cfg: HybridConfig,
    u: torch.Tensor,
    past: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Gated attention on T new rows `u`, after the `past[0].shape[1]` cached tokens."""
    new, heads, kv_heads, d = u.shape[0], cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    seen = 0 if past is None else past[0].shape[1]

    # Each head owns 2 * d consecutive outputs of q_proj: its query, then its gate.
    both = (u @ params[prefix + "q_proj.weight"].T).reshape(new, heads, 2 * d)
    q, gate = both[..., :d], both[..., d:]
    k = (u @ params[prefix + "k_proj.weight"].T).reshape(new, kv_heads, d)
    v = (u @ params[prefix + "v_proj.weight"].T).reshape(new, kv_heads, d)

    # Norm over the head dimension, then heads first, then the rotation at the
    # positions these tokens have in the whole sequence.
    q = ops.rms_norm(q, params[prefix + "q_norm.weight"], cfg.norm_eps).transpose(0, 1)
    k = ops.rms_norm(k, params[prefix + "k_norm.weight"], cfg.norm_eps).transpose(0, 1)
    v = v.transpose(0, 1)
    positions = torch.arange(seen, seen + new)
    q = ops.partial_rope(q, positions, cfg.rope_base, cfg.partial_rotary_factor)
    k = ops.partial_rope(k, positions, cfg.rope_base, cfg.partial_rotary_factor)

    # The cache holds finished keys: normalized and rotated once, never again.
    if past is not None:
        k = torch.cat((past[0], k), dim=1)
        v = torch.cat((past[1], v), dim=1)

    # Row t is the token at position seen + t: it sees keys 0 .. seen + t.
    mask = torch.arange(seen + new)[None, :] <= positions[:, None]
    attended = ops.attention(q, k, v, mask).transpose(0, 1).reshape(new, heads * d)
    gated = attended * F.silu(gate.reshape(new, heads * d))
    return gated @ params[prefix + "o_proj.weight"].T, (k, v)


def hybrid_forward(
    params: dict[str, torch.Tensor],
    cfg: HybridConfig,
    tokens: torch.Tensor,
    state: list[tuple[torch.Tensor, torch.Tensor]] | None,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    past = [None] * cfg.n_layers if state is None else state
    new_state = []  # a new list: the caller's is left as it was
    x = params["model.embed_tokens.weight"][tokens]
    for i, kind in enumerate(cfg.layers):
        layer = f"model.layers.{i}."
        u = ops.rms_norm(x, params[layer + "input_layernorm.weight"], cfg.norm_eps)
        if kind == "linear_attention":
            delta = {name: params[f"{layer}linear_attn.{name}"] for name in DELTA}
            mixed, kept = ops.gated_delta_layer(u, delta, past[i], cfg.n_key_heads, cfg.norm_eps)
        else:
            mixed, kept = gated_attention(params, layer + "self_attn.", cfg, u, past[i])
        new_state.append(kept)
        x = x + mixed
        u = ops.rms_norm(x, params[layer + "post_attention_layernorm.weight"], cfg.norm_eps)
        x = x + ops.swiglu(
            u,
            params[layer + "mlp.gate_proj.weight"],
            params[layer + "mlp.up_proj.weight"],
            params[layer + "mlp.down_proj.weight"],
        )
    x = ops.rms_norm(x, params["model.norm.weight"], cfg.norm_eps)
    return x @ params["model.embed_tokens.weight"].T, new_state
