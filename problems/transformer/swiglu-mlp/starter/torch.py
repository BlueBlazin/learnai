import torch
import torch.nn.functional as F


def swiglu(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """SwiGLU feed-forward layer: (silu(x W_gate^T) * (x W_up^T)) W_down^T.

    x: float32, shape (..., d). One to three dimensions; the last axis is the feature axis.
    w_gate, w_up: float32, shape (h, d). Gate and up projections, (out, in) layout.
    w_down: float32, shape (d_out, h). Down projection, (out, in) layout.
    Returns float32, shape (..., d_out).
    """
    raise NotImplementedError
