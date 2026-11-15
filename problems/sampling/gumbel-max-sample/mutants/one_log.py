# BUG: applies one logarithm instead of two; the noise -log(u) is exponential, not Gumbel, and the winners do not follow the softmax.
import torch

U_MIN = 1.1754944e-38
U_MAX = 0.99999994


def gumbel_sample(logits: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor:
    u = torch.clamp(uniforms, U_MIN, U_MAX)
    noise = -torch.log(u)
    return torch.argmax(logits + noise, dim=-1)
