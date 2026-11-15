# BUG: no upper clamp: u = 1 gives +inf noise, a finite logit with it always wins, and a masked logit with it scores -inf + inf = NaN, which argmax returns.
import torch

U_MIN = 1.1754944e-38
U_MAX = 0.99999994


def gumbel_sample(logits: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor:
    u = torch.clamp(uniforms, min=U_MIN)
    gumbel = -torch.log(-torch.log(u))
    return torch.argmax(logits + gumbel, dim=-1)
