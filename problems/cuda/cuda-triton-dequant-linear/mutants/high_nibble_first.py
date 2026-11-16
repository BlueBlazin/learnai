# BUG: reads the codes of a word from the highest bits down; the order of the eight codes inside every word is reversed.
import torch

from learnai.kit import tritonshim as tl


@tl.jit
def qlinear_kernel(
    x_ptr, words_ptr, scales_ptr, biases_ptr, y_ptr, n_in, n_out, group_size, BLOCK: tl.constexpr
):
    # One program per output element: axis 0 is the output row, axis 1 the batch row.
    out_row = tl.program_id(0)
    batch_row = tl.program_id(1)
    x_row = x_ptr + batch_row * n_in
    words_row = words_ptr + out_row * (n_in // 8)  # 8 codes per uint32 word
    scales_row = scales_ptr + out_row * (n_in // group_size)
    biases_row = biases_ptr + out_row * (n_in // group_size)

    acc = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, n_in, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_in
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        # Column c is code c % 8 of word c // 8, lowest bits first. The word is
        # unsigned, so the shift brings in zeros.
        word = tl.load(words_row + cols // 8, mask=mask, other=0)
        code = (word >> ((7 - cols % 8) * 4).to(tl.uint32)) & 0xF
        # The group is a property of the column, not of the block.
        group = cols // group_size
        scale = tl.load(scales_row + group, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(biases_row + group, mask=mask, other=0.0).to(tl.float32)
        # The weight exists only here, in registers, for the length of one product.
        acc += x * (code.to(tl.float32) * scale + bias)
    tl.store(y_ptr + batch_row * n_out + out_row, tl.sum(acc, axis=0))


def qlinear(
    x: torch.Tensor,
    words: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int,
    block: int = 64,
) -> torch.Tensor:
    n_out, n_groups = scales.shape
    n_in = n_groups * group_size
    rows = x.reshape(-1, n_in).contiguous()
    y = torch.empty((rows.shape[0], n_out), dtype=x.dtype, device=x.device)
    grid = (n_out, rows.shape[0])
    qlinear_kernel[grid](
        rows,
        words.contiguous(),
        scales.contiguous(),
        biases.contiguous(),
        y,
        n_in,
        n_out,
        group_size,
        BLOCK=block,
    )
    return y.reshape(*x.shape[:-1], n_out)
