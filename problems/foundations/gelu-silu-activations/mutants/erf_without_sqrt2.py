# BUG: the exact form uses erf(x) instead of erf(x / sqrt(2)), which is the CDF of a normal with variance 1/2, not 1.
import math

import torch


def gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    if approximate == "tanh":
        inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)
        return 0.5 * x * (1.0 + torch.tanh(inner))
    return 0.5 * x * (1.0 + torch.erf(x))


def silu(x: torch.Tensor) -> torch.Tensor:
    return x / (1.0 + torch.exp(-x))
