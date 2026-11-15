# BUG: no lower clamp: u = 0 gives -inf noise, so a finite logit scores -inf like a masked one, and a row whose only unmasked token drew 0 returns index 0.
import torch

U_MIN = 1.1754944e-38
U_MAX = 0.99999994


def gumbel_sample(logits: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor:
    u = torch.clamp(uniforms, max=U_MAX)
    gumbel = -torch.log(-torch.log(u))
    return torch.argmax(logits + gumbel, dim=-1)
