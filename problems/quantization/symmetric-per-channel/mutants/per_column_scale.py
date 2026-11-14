# BUG: reduces over axis 0, giving one scale per input column instead of one per output row; for a square matrix the shapes still fit.
import torch


def quantize_weight(w: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = 2 ** (bits - 1) - 1
    max_abs = w.abs().amax(dim=0)
    scale = torch.where(max_abs > 0, max_abs / qmax, torch.ones_like(max_abs))
    q = torch.clamp(torch.round(w / scale[None, :]), -qmax, qmax)
    return q.to(torch.int8), scale


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale[:, None]
