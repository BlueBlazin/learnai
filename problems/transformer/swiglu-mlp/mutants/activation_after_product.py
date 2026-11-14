# BUG: applies SiLU to the product of the two branches instead of to the gate branch alone.
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
    return F.silu(gate * up) @ w_down.T
