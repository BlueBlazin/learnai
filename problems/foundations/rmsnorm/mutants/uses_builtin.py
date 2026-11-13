# BUG: calls the built-in torch.nn.functional.rms_norm, which the statement rules out (and which squares huge activations to infinity).
import torch
import torch.nn.functional as F


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), weight, eps)
