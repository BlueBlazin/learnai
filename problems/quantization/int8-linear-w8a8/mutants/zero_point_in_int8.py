# BUG: subtracts the zero-point from the codes in int8 before the product; x_q - zero_point spans [-255, 255] and wraps for every token whose range is not centered on zero.
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
    acc = x_q.to(torch.int32) @ w32.T  # (n, out), int32

    corrected = (x_q - zero_point.to(torch.int8)).to(torch.int32) @ w32.T

    y = corrected.to(torch.float32) * (x_scale * w_scale) + bias
    return y, acc
