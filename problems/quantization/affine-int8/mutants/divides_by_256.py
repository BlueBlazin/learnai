# BUG: divides the range by 256 (the number of codes) instead of 255 (the number of steps), so max(x) no longer maps to code 127.
import torch

QMIN, QMAX = -128, 127


def quant_params(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    zero = torch.zeros((), dtype=x.dtype)
    lo = torch.minimum(x.min(), zero)
    hi = torch.maximum(x.max(), zero)
    scale = torch.where(hi > lo, (hi - lo) / (QMAX - QMIN + 1), torch.ones_like(hi))
    zero_point = torch.clamp(QMIN - torch.round(lo / scale), QMIN, QMAX)
    return scale, zero_point.to(torch.int32)


def quantize(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    q = torch.round(x / scale) + zero_point
    return torch.clamp(q, QMIN, QMAX).to(torch.int8)


def dequantize(q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    return (q.to(torch.int32) - zero_point).to(torch.float32) * scale
