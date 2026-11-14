# BUG: ignores the supplied uniforms and draws from the framework's own generator with torch.multinomial, which the statement rules out; the result is not a function of u.
import torch


def sample(probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
