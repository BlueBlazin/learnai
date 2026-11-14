# BUG: approximate="tanh" returns the paper's other approximation, x * sigmoid(1.702 x), which is off by up to 0.02.
import math

import torch


def gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    if approximate == "tanh":
        return x / (1.0 + torch.exp(-1.702 * x))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def silu(x: torch.Tensor) -> torch.Tensor:
    return x / (1.0 + torch.exp(-x))
