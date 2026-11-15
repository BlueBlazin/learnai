# BUG: rescales by the mean of |x| instead of the maximum: an entry can then be up to d times the scale, and its square overflows float16 once d exceeds 255.
import math

import torch


def rms_norm_f16(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    root_eps = math.sqrt(eps)
    c = torch.clamp(x.abs().mean(dim=-1, keepdim=True), min=root_eps)
    u = x / c
    r = root_eps / c
    return u / torch.sqrt(u.square().mean(dim=-1, keepdim=True) + r * r) * weight
