# BUG: rescales by max|x| alone and computes eps / s^2: for a tiny vector s * s underflows to 0, the quotient is inf, and the output collapses to 0.
import torch


def rms_norm_f16(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    s = x.abs().amax(dim=-1, keepdim=True)
    s = torch.where(s == 0, 1.0, s)
    u = x / s
    return u / torch.sqrt(u.square().mean(dim=-1, keepdim=True) + eps / (s * s)) * weight
