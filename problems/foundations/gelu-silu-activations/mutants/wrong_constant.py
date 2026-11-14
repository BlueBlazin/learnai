# BUG: the tanh form uses sqrt(2) / pi (0.450) instead of sqrt(2 / pi) (0.798): the square root does not cover pi.
import math

import torch


def gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    if approximate == "tanh":
        inner = math.sqrt(2.0) / math.pi * (x + 0.044715 * x * x * x)
        return 0.5 * x * (1.0 + torch.tanh(inner))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def silu(x: torch.Tensor) -> torch.Tensor:
    return x / (1.0 + torch.exp(-x))
