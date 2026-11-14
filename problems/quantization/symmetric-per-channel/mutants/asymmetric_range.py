# BUG: uses the full two's-complement range [-2**(bits-1), 2**(bits-1) - 1]: the scale divides by 2**(bits-1), so the largest positive weight needs a code one above the maximum and is clipped.
import torch


def quantize_weight(w: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    max_abs = w.abs().amax(dim=1)
    scale = torch.where(max_abs > 0, max_abs / -qmin, torch.ones_like(max_abs))
    q = torch.clamp(torch.round(w / scale[:, None]), qmin, qmax)
    return q.to(torch.int8), scale


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale[:, None]
