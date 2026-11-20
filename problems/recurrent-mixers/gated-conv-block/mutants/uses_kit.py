# BUG: calls the kit's ops.causal_conv1d, which the statement rules out.
import torch

from learnai.kit import ops


def conv_block(
    x: torch.Tensor,
    w_in: torch.Tensor,
    w_conv: torch.Tensor,
    w_out: torch.Tensor,
    state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, C, h = torch.chunk(x @ w_in.T, 3, dim=-1)
    conv, state = ops.causal_conv1d(B * h, w_conv, state)
    return (C * conv) @ w_out.T, state
