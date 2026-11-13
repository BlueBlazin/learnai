import torch

QMIN, QMAX = -128, 127


def quant_params(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale and zero-point of the affine int8 grid that covers [min(x), max(x)] and 0.

    x: float32 tensor of any shape with 1 to 4 dimensions (one grid for the whole tensor).
    Returns (scale, zero_point): a 0-d float32 tensor with scale > 0, and a 0-d integer
    tensor with QMIN <= zero_point <= QMAX.
    """
    raise NotImplementedError


def quantize(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """Map x to int8 codes: clamp(round(x / scale) + zero_point, QMIN, QMAX).

    x: float32 tensor of any shape. scale: 0-d float32. zero_point: 0-d integer.
    Returns an int8 tensor of the same shape as x.
    """
    raise NotImplementedError


def dequantize(q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """Map int8 codes back to real values: (q - zero_point) * scale.

    q: integer tensor (int8 in practice) with values in [QMIN, QMAX], any shape.
    Returns a float32 tensor of the same shape as q.
    """
    raise NotImplementedError
