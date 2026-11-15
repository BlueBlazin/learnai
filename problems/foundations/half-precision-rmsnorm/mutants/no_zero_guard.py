# BUG: rescales by max|x| with no lower bound: an all-zero vector divides 0 by 0 and comes out as NaN, and for a tiny vector (sqrt(eps) / c)^2 overflows.
import math

import torch


def rms_norm_f16(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    root_eps = math.sqrt(eps)
    c = x.abs().amax(dim=-1, keepdim=True)
    u = x / c
    r = root_eps / c
    return u / torch.sqrt(u.square().mean(dim=-1, keepdim=True) + r * r) * weight
