# BUG: rescales x but adds eps unchanged to the mean square of the rescaled vector; eps must be divided by the square of the scale too.
import math

import torch


def rms_norm_f16(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    root_eps = math.sqrt(eps)
    c = torch.clamp(x.abs().amax(dim=-1, keepdim=True), min=root_eps)
    u = x / c
    return u / torch.sqrt(u.square().mean(dim=-1, keepdim=True) + eps) * weight
