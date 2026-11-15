# BUG: launches one program per output element instead of one per row; every program reads its whole row again to compute the same root mean square, n_cols times the work.
import torch

from whetstone.kit import tritonshim as tl


@tl.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_cols, eps, BLOCK: tl.constexpr):
    # One program per ELEMENT: each recomputes the sum of squares of its whole row.
    pid = tl.program_id(0)
    row = pid // n_cols
    col = pid % n_cols
    x_row = x_ptr + row * n_cols

    acc = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + cols, mask=cols < n_cols, other=0.0).to(tl.float32)
        acc += x * x
    rms = tl.sqrt(tl.sum(acc, axis=0) / n_cols + eps)

    x = tl.load(x_row + col).to(tl.float32)
    w = tl.load(w_ptr + col).to(tl.float32)
    tl.store(out_ptr + pid, x / rms * w)


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5, block: int = 64
) -> torch.Tensor:
    n_cols = x.shape[-1]
    rows = x.reshape(-1, n_cols).contiguous()
    out = torch.empty_like(rows)
    rmsnorm_kernel[(rows.numel(),)](rows, weight.contiguous(), out, n_cols, eps, BLOCK=block)
    return out.reshape(x.shape)
