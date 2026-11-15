import torch

from whetstone.kit import tritonshim as tl


@tl.jit
def qlinear_kernel(
    x_ptr, words_ptr, scales_ptr, biases_ptr, y_ptr, n_in, n_out, group_size, BLOCK: tl.constexpr
):
    """Program (o, b) of a grid of (n_out, batch) writes y[b, o] = sum over columns c of
    x[b, c] * (code[o, c] * scales[o, c // group_size] + biases[o, c // group_size]).

    x_ptr: (batch, n_in) row-major, float32 or float16.
    words_ptr: (n_out, n_in / 8) row-major uint32; code c of a row is bits
        4 * (c % 8) to 4 * (c % 8) + 3 of word c // 8.
    scales_ptr, biases_ptr: (n_out, n_in / group_size) row-major, float32 or float16.
    y_ptr: (batch, n_out) row-major. n_in, n_out, group_size: int32 scalars.
    BLOCK: columns per block, a power of two.
    """
    out_row = tl.program_id(0)
    batch_row = tl.program_id(1)
    # TODO: accumulate the dot product over blocks of BLOCK columns.
    tl.store(y_ptr + batch_row * n_out + out_row, 0.0)


def qlinear(
    x: torch.Tensor,
    words: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int,
    block: int = 64,
) -> torch.Tensor:
    """x @ W_hat.T for the 4-bit quantized matrix W_hat, by one launch of qlinear_kernel.

    x: (..., n_in), float32 or float16. words: (n_out, n_in / 8), uint32.
    scales, biases: (n_out, n_in / group_size), float32 or float16.
    Returns a tensor of shape (..., n_out) with the dtype and device of x.
    """
    raise NotImplementedError
