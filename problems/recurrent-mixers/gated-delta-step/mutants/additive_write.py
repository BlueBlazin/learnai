# BUG: the write adds beta * v k^T without subtracting what the memory already returns for k, so repeated keys pile up instead of being overwritten.
import torch
import torch.nn.functional as F


def gdn_gates(
    a: torch.Tensor, b: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # softplus is positive and exp(A_log) is positive, so the exponent is negative
    # and g lies in (0, 1): the memory can fade, never grow.
    g = torch.exp(-torch.exp(A_log) * F.softplus(a + dt_bias))
    beta = torch.sigmoid(b)
    return g, beta


def gdn_step(
    S: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    S = S * g[:, None, None]
    delta = beta[:, None] * v  # no correction by S k
    S = S + delta[:, :, None] * k[:, None, :]
    y = torch.einsum("hvk,hk->hv", S, q)
    return y, S
