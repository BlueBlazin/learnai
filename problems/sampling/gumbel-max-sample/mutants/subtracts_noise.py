# BUG: subtracts the Gumbel noise (logits + log(-log(u))); the negated noise is not Gumbel distributed and the winners do not follow the softmax.
import torch

U_MIN = 1.1754944e-38
U_MAX = 0.99999994


def gumbel_sample(logits: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor:
    u = torch.clamp(uniforms, U_MIN, U_MAX)
    gumbel = -torch.log(-torch.log(u))
    return torch.argmax(logits - gumbel, dim=-1)
