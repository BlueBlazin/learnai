# BUG: ignores `approximate` and always returns the exact erf form, so approximate="tanh" does not reproduce a checkpoint trained with the approximation.
import math

import torch


def gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def silu(x: torch.Tensor) -> torch.Tensor:
    return x / (1.0 + torch.exp(-x))
