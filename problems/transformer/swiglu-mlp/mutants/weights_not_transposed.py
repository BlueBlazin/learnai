# BUG: applies the projections as x @ W instead of x @ W^T, treating the (out, in) weights as (in, out); it only runs at all when d = h = d_out.
import torch
import torch.nn.functional as F


def swiglu(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    gate = x @ w_gate
    up = x @ w_up
    return (F.silu(gate) * up) @ w_down
