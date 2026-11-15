# BUG: guards the logarithms with an epsilon of 1e-10 instead of clamping to the float32 limits; in float32 1 - 1e-10 rounds to 1, so u = 1 still gives +inf noise and NaN on a masked entry.
import torch

U_MIN = 1.1754944e-38
U_MAX = 0.99999994


def gumbel_sample(logits: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor:
    eps = 1e-10
    gumbel = -torch.log(-torch.log(torch.clamp(uniforms, eps, 1.0 - eps)))
    return torch.argmax(logits + gumbel, dim=-1)
