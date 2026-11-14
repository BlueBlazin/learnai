# BUG: applies SiLU to the up branch and leaves the gate branch linear; the two projections are not interchangeable in a checkpoint.
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
    return (gate * F.silu(up)) @ w_down.T
