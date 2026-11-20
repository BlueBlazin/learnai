# BUG: the decay is applied after the write, so the value just written is already faded by g when it is read.
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
    predicted = torch.einsum("hvk,hk->hv", S, k)
    delta = beta[:, None] * (v - predicted)
    S = S + delta[:, :, None] * k[:, None, :]
    S = S * g[:, None, None]  # decay last
    y = torch.einsum("hvk,hk->hv", S, q)
    return y, S
