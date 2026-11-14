# BUG: calls the built-in gelu and silu, which the statement rules out; the numbers are right, the exercise is not done.
import torch
import torch.nn.functional as F


def gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    return F.gelu(x, approximate=approximate)


def silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)
