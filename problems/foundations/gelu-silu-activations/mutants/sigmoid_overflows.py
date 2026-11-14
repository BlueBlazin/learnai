# BUG: silu computes the sigmoid as exp(x) / (1 + exp(x)); for x above about 88.7, exp(x) overflows float32 and inf / inf is NaN.
import math

import torch


def gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    if approximate == "tanh":
        inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)
        return 0.5 * x * (1.0 + torch.tanh(inner))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.exp(x) / (1.0 + torch.exp(x))
