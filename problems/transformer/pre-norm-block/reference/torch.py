import torch

from whetstone.kit import ops


def block(x: torch.Tensor, params: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    # Attention sublayer. It reads a normalized copy of the stream; the stream itself
    # is never normalized, only added to.
    h = ops.rms_norm(x, params["input_layernorm.weight"])
    q = h @ params["self_attn.q_proj.weight"].T
    k = h @ params["self_attn.k_proj.weight"].T
    v = h @ params["self_attn.v_proj.weight"].T
    x = x + ops.attention(q, k, v, mask) @ params["self_attn.o_proj.weight"].T

    # MLP sublayer. It reads the stream *after* attention was added, through its own
    # norm, and adds to that same stream.
    h = ops.rms_norm(x, params["post_attention_layernorm.weight"])
    return x + ops.swiglu(
        h,
        params["mlp.gate_proj.weight"],
        params["mlp.up_proj.weight"],
        params["mlp.down_proj.weight"],
    )
