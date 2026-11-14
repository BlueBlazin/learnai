# BUG: ignores `approximate` and always returns the tanh approximation, so approximate="none" is off by up to 4.7e-4.
import math

import torch


def gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + torch.tanh(inner))


def silu(x: torch.Tensor) -> torch.Tensor:
    return x / (1.0 + torch.exp(-x))
