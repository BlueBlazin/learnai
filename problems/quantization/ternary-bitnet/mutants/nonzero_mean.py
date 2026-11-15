# BUG: the mean magnitude is taken over the non-zero entries only; a matrix with exact zeros gets a larger scale and a different border for every weight.
import torch


def ternary_quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gamma = w.abs().sum() / torch.count_nonzero(w).clamp(min=1)
    scale = torch.where(gamma > 0, gamma, torch.ones_like(gamma))
    t = torch.clamp(torch.round(w / scale), -1, 1)
    return t.to(torch.int8), scale


def ternary_matmul(x: torch.Tensor, t: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    xe = x.unsqueeze(-2)
    zero = torch.zeros_like(xe)
    plus = torch.where(t == 1, xe, zero).sum(dim=-1)
    minus = torch.where(t == -1, xe, zero).sum(dim=-1)
    return scale * (plus - minus)
