# BUG: ignores the supplied uniforms and draws its own, which the statement rules out; the output cannot be reproduced or compared.
import torch

U_MIN = 1.1754944e-38
U_MAX = 0.99999994


def gumbel_sample(logits: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor:
    u = torch.clamp(torch.rand_like(uniforms), U_MIN, U_MAX)
    gumbel = -torch.log(-torch.log(u))
    return torch.argmax(logits + gumbel, dim=-1)
