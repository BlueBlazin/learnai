# BUG: gates with sigmoid(g) instead of silu(g) = g * sigmoid(g); that is the original gated linear unit (GLU), not SwiGLU.
import torch


def swiglu(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    gate = x @ w_gate.T
    up = x @ w_up.T
    return (torch.sigmoid(gate) * up) @ w_down.T
