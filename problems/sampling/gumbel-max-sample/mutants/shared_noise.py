# BUG: uses the first row of uniforms for every row, so the rows of a batch are not sampled independently.
import torch

U_MIN = 1.1754944e-38
U_MAX = 0.99999994


def gumbel_sample(logits: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor:
    u = torch.clamp(uniforms[:1], U_MIN, U_MAX)
    gumbel = -torch.log(-torch.log(u))
    return torch.argmax(logits + gumbel, dim=-1)
