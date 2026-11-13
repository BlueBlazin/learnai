# BUG: floors the zero-point instead of rounding it to the nearest integer, shifting the whole grid by one code for about half of all ranges.
import torch

QMIN, QMAX = -128, 127


def quant_params(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    zero = torch.zeros((), dtype=x.dtype)
    lo = torch.minimum(x.min(), zero)
    hi = torch.maximum(x.max(), zero)
    scale = torch.where(hi > lo, (hi - lo) / (QMAX - QMIN), torch.ones_like(hi))
    zero_point = torch.clamp(torch.floor(QMIN - lo / scale), QMIN, QMAX)
    return scale, zero_point.to(torch.int32)


def quantize(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    q = torch.round(x / scale) + zero_point
    return torch.clamp(q, QMIN, QMAX).to(torch.int8)


def dequantize(q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    return (q.to(torch.int32) - zero_point).to(torch.float32) * scale
