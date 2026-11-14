# BUG: dequantize multiplies q of shape (out, in) by scale of shape (out,) without adding an axis, so the scales line up with the columns; it fails for most shapes and is silently wrong for square ones.
import torch


def quantize_weight(w: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = 2 ** (bits - 1) - 1
    max_abs = w.abs().amax(dim=1)
    scale = torch.where(max_abs > 0, max_abs / qmax, torch.ones_like(max_abs))
    q = torch.clamp(torch.round(w / scale[:, None]), -qmax, qmax)
    return q.to(torch.int8), scale


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale
