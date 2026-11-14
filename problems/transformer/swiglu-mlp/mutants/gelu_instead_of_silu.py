# BUG: gates with GELU instead of SiLU; that is GeGLU, the feed-forward of Gemma, not SwiGLU.
import torch
import torch.nn.functional as F


def swiglu(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    gate = x @ w_gate.T
    up = x @ w_up.T
    return (F.gelu(gate) * up) @ w_down.T
