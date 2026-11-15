# BUG: adds the Gumbel noise to the probabilities instead of the logits; probabilities lie in [0, 1] and are swamped by the noise.
import torch

U_MIN = 1.1754944e-38
U_MAX = 0.99999994


def gumbel_sample(logits: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor:
    u = torch.clamp(uniforms, U_MIN, U_MAX)
    gumbel = -torch.log(-torch.log(u))
    return torch.argmax(torch.softmax(logits, dim=-1) + gumbel, dim=-1)
