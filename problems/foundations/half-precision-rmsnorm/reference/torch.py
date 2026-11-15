import math

import torch


def rms_norm_f16(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # Divide each vector by c = max(max|x|, sqrt(eps)) before squaring anything:
    #   x / sqrt(mean(x^2) + eps) == u / sqrt(mean(u^2) + (sqrt(eps) / c)^2),  u = x / c.
    # Every |u| is at most 1 and sqrt(eps) / c is at most 1, and one of the two terms
    # under the root is at least 1 / d, so nothing overflows, nothing that matters
    # underflows, and an all-zero vector needs no special case (c = sqrt(eps) > 0).
    root_eps = math.sqrt(eps)
    c = torch.clamp(x.abs().amax(dim=-1, keepdim=True), min=root_eps)
    u = x / c
    r = root_eps / c
    return u / torch.sqrt(u.square().mean(dim=-1, keepdim=True) + r * r) * weight
