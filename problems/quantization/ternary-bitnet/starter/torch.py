import torch


def ternary_quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize w to {-1, 0, 1} with one scale, the mean absolute value of w.

    w: float32 tensor of shape (out, in).
    Returns (t, scale): an int8 tensor of shape (out, in) with values -1, 0, or 1, and
    a 0-d float32 tensor (1 if every weight is 0). round() rounds half to even.
    """
    raise NotImplementedError


def ternary_matmul(x: torch.Tensor, t: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """x @ (scale * t).T, computed by adding and subtracting selected inputs.

    x: float32 tensor of shape (..., in). t: int8 tensor of shape (out, in) with values
    -1, 0, or 1. scale: 0-d float32 tensor.
    Returns a float32 tensor of shape (..., out).
    """
    raise NotImplementedError
