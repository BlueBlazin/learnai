# BUG: overwrites the stream with its normalized copy (x = norm(x); x = x + f(x)), so the residual carries the normalized values instead of the untouched input.
import torch

from learnai.kit import ops


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
    x = ops.rms_norm(x, params["input_layernorm.weight"])
    x = x + attn(x, params, mask)
    x = ops.rms_norm(x, params["post_attention_layernorm.weight"])
    return x + mlp(x, params)
