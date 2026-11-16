import torch

from learnai.kit import tritonshim as tl


@tl.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_cols, eps, BLOCK: tl.constexpr):
    """One program per row: out[row] = x[row] / sqrt(mean(x[row] ** 2) + eps) * w.

    x_ptr, out_ptr: pointers to (n_rows, n_cols) row-major memory, float32 or float16.
    w_ptr: pointer to the n_cols gains, same element type.
    n_cols: int32 scalar. eps: float32 scalar. BLOCK: columns per block, a power of two.
    Launched on a grid of (n_rows,); tl.program_id(0) is the row.
    """
    row = tl.program_id(0)
    # TODO: walk the row in blocks of BLOCK columns, with a mask for the last block.
    tl.store(out_ptr + row * n_cols, 0.0)


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5, block: int = 64
) -> torch.Tensor:
    """RMSNorm over the last axis of `x`, computed by one launch of rmsnorm_kernel.

    x: float32 or float16, shape (..., d); weight: shape (d,), the dtype of x.
    Returns a tensor with the shape, dtype, and device of x.
    """
    raise NotImplementedError
