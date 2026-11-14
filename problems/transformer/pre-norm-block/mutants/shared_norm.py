# BUG: uses the gain of input_layernorm for both sublayers; the MLP must read the stream through post_attention_layernorm.
import torch

from whetstone.kit import ops


def attn(h: torch.Tensor, params: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    q = h @ params["self_attn.q_proj.weight"].T
    k = h @ params["self_attn.k_proj.weight"].T
    v = h @ params["self_attn.v_proj.weight"].T
    return ops.attention(q, k, v, mask) @ params["self_attn.o_proj.weight"].T


def mlp(h: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
    return ops.swiglu(
        h,
        params["mlp.gate_proj.weight"],
        params["mlp.up_proj.weight"],
        params["mlp.down_proj.weight"],
    )


def block(x: torch.Tensor, params: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    x = x + attn(ops.rms_norm(x, params["input_layernorm.weight"]), params, mask)
    return x + mlp(ops.rms_norm(x, params["input_layernorm.weight"]), params)
