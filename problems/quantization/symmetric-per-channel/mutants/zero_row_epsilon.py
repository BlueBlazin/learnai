# BUG: guards the division by adding 1e-8 to every scale instead of special-casing the all-zero row: that row gets scale 1e-8 instead of 1, and rows of tiny weights get a visibly wrong scale and codes.
import torch


def quantize_weight(w: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = 2 ** (bits - 1) - 1
    max_abs = w.abs().amax(dim=1)
    scale = max_abs / qmax + 1e-8
    q = torch.clamp(torch.round(w / scale[:, None]), -qmax, qmax)
    return q.to(torch.int8), scale


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale[:, None]
