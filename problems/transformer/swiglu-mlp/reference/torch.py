import torch
import torch.nn.functional as F


def swiglu(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    # Weights are (out, in), so a projection is x @ W^T: (..., d) -> (..., h).
    gate = x @ w_gate.T
    up = x @ w_up.T
    # Only the gate branch goes through SiLU; the up branch stays linear.
    hidden = F.silu(gate) * up
    return hidden @ w_down.T  # (..., h) -> (..., d_out)
