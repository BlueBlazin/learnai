# BUG: accumulates the products in float32 and casts the result to int32; float32 holds integers exactly only up to 2**24, so a wide layer with large codes is off by a few units.
import torch

QMIN, QMAX = -128, 127


def int8_linear(
    x: torch.Tensor, w_q: torch.Tensor, w_scale: torch.Tensor, bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    lo = x.amin(dim=1, keepdim=True).clamp(max=0.0)
    hi = x.amax(dim=1, keepdim=True).clamp(min=0.0)
    x_scale = torch.where(hi > lo, (hi - lo) / (QMAX - QMIN), torch.ones_like(hi))
    zero_point = torch.clamp(QMIN - torch.round(lo / x_scale), QMIN, QMAX).to(torch.int32)
    x_q = torch.clamp(torch.round(x / x_scale) + zero_point, QMIN, QMAX).to(torch.int8)

    w32 = w_q.to(torch.int32)
    acc = (x_q.to(torch.float32) @ w_q.to(torch.float32).T).to(torch.int32)  # (n, out), int32

    corrected = acc - zero_point * w32.sum(dim=1)

    y = corrected.to(torch.float32) * (x_scale * w_scale) + bias
    return y, acc
