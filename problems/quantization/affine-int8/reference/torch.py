import torch

QMIN, QMAX = -128, 127


def quant_params(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Widen the range to include 0, so that 0.0 lands exactly on a grid point.
    zero = torch.zeros((), dtype=x.dtype)
    lo = torch.minimum(x.min(), zero)
    hi = torch.maximum(x.max(), zero)
    # 256 codes have 255 steps between them. An all-zero tensor gets scale 1.
    scale = torch.where(hi > lo, (hi - lo) / (QMAX - QMIN), torch.ones_like(hi))
    # lo must map to QMIN: round(lo / scale) + zero_point = QMIN.
    zero_point = torch.clamp(QMIN - torch.round(lo / scale), QMIN, QMAX)
    return scale, zero_point.to(torch.int32)


def quantize(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    q = torch.round(x / scale) + zero_point
    # Clamp while still in float: casting an out-of-range value to int8 wraps around.
    return torch.clamp(q, QMIN, QMAX).to(torch.int8)


def dequantize(q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    # Widen before subtracting: q - zero_point spans [-255, 255], which overflows int8.
    return (q.to(torch.int32) - zero_point).to(torch.float32) * scale
