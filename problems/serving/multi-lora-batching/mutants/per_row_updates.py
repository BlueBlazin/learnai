# BUG: the base layer is called once, but the low-rank updates are added in a Python loop over the rows: correct, and far slower than one gathered batched matmul.
import torch


def batched_lora_forward(
    x: torch.Tensor,
    base,
    a_stack: torch.Tensor,
    b_stack: torch.Tensor,
    ranks: torch.Tensor,
    adapter_ids: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    y = base(x)
    for row in range(x.shape[0]):
        n = int(adapter_ids[row])
        if n >= 0:
            scale = alpha / int(ranks[n])
            y[row] += scale * ((x[row] @ a_stack[n].transpose(0, 1)) @ b_stack[n].transpose(0, 1))
    return y
