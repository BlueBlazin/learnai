# BUG: the second residual connection adds the MLP output to the block input instead of to the stream after attention, so the attention output is dropped from the result.
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
    h = x + attn(ops.rms_norm(x, params["input_layernorm.weight"]), params, mask)
    return x + mlp(ops.rms_norm(h, params["post_attention_layernorm.weight"]), params)
