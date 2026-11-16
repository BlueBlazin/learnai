# BUG: squares and accumulates in the element type of x; for float16 activations above 255 the square overflows to infinity and the row comes out as zeros.
import torch

from learnai.kit import tritonshim as tl


@tl.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_cols, eps, BLOCK: tl.constexpr):
    # One program per row. A pointer plus a number of elements is a pointer.
    row = tl.program_id(0)
    x_row = x_ptr + row * n_cols
    out_row = out_ptr + row * n_cols

    # Pass 1: the sum of squares, BLOCK columns at a time. `acc` keeps BLOCK partial
    # sums in float32, whatever the element type of x is.
    acc = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        # The last block hangs over the end of the row. Masked-off elements are not
        # read at all, and count as 0 in the sum.
        x = tl.load(x_row + cols, mask=cols < n_cols, other=0.0)
        acc += x * x
    # The mean is over the n_cols real columns; eps goes inside the root.
    rms = tl.sqrt(tl.sum(acc, axis=0) / n_cols + eps)

    # Pass 2: normalize, apply the gain, and write only the real columns.
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        # The store converts the float32 result to the element type of `out`.
        tl.store(out_row + cols, x / rms * w, mask=mask)


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5, block: int = 64
) -> torch.Tensor:
    n_cols = x.shape[-1]
    # A kernel sees raw memory, row after row: flatten the batch axes and make sure
    # the layout really is contiguous (a strided view is not).
    rows = x.reshape(-1, n_cols).contiguous()
    out = torch.empty_like(rows)
    rmsnorm_kernel[(rows.shape[0],)](rows, weight.contiguous(), out, n_cols, eps, BLOCK=block)
    return out.reshape(x.shape)
