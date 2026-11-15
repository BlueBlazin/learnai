# BUG: rounds without clamping to [-1, 1], so weights beyond 1.5 scales get the codes 2, 3, ... which are not ternary.
import torch


def ternary_quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gamma = w.abs().mean()
    scale = torch.where(gamma > 0, gamma, torch.ones_like(gamma))
    t = torch.round(w / scale)
    return t.to(torch.int8), scale


def ternary_matmul(x: torch.Tensor, t: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    xe = x.unsqueeze(-2)
    zero = torch.zeros_like(xe)
    plus = torch.where(t == 1, xe, zero).sum(dim=-1)
    minus = torch.where(t == -1, xe, zero).sum(dim=-1)
    return scale * (plus - minus)
