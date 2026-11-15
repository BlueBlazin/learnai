import torch

QMIN, QMAX = -128, 127


def int8_linear(
    x: torch.Tensor, w_q: torch.Tensor, w_scale: torch.Tensor, bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # One affine grid per token (row of x), widened to include 0 so that 0.0 is a code.
    lo = x.amin(dim=1, keepdim=True).clamp(max=0.0)
    hi = x.amax(dim=1, keepdim=True).clamp(min=0.0)
    x_scale = torch.where(hi > lo, (hi - lo) / (QMAX - QMIN), torch.ones_like(hi))
    zero_point = torch.clamp(QMIN - torch.round(lo / x_scale), QMIN, QMAX).to(torch.int32)
    x_q = torch.clamp(torch.round(x / x_scale) + zero_point, QMIN, QMAX).to(torch.int8)

    # The integer part of the layer. Widen before multiplying: one product of two int8
    # codes already needs 15 bits, and the sum over `in` of them needs more.
    w32 = w_q.to(torch.int32)
    acc = x_q.to(torch.int32) @ w32.T  # (n, out), int32

    # x_q stands for x_scale * (x_q - zero_point), so the product owes a correction:
    # sum_j (x_q - z) * w_q = acc - z * sum_j w_q. The row sums of w_q do not depend on x.
    corrected = acc - zero_point * w32.sum(dim=1)

    # Back to real numbers once, at the end: one scale per token times one per output.
    y = corrected.to(torch.float32) * (x_scale * w_scale) + bias
    return y, acc
