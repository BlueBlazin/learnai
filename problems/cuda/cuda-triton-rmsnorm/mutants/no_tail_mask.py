# BUG: loads and stores whole blocks with no mask; the last block of a row reads the next row into the sum and writes over it, and the last row runs off the end of the tensor.
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
        x = tl.load(x_row + cols).to(tl.float32)
        acc += x * x
    # The mean is over the n_cols real columns; eps goes inside the root.
    rms = tl.sqrt(tl.sum(acc, axis=0) / n_cols + eps)

    # Pass 2: normalize, apply the gain, and write only the real columns.
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + cols).to(tl.float32)
        w = tl.load(w_ptr + cols).to(tl.float32)
        # The store converts the float32 result to the element type of `out`.
        tl.store(out_row + cols, x / rms * w)


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
