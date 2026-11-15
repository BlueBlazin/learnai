# BUG: the batch is processed one row at a time in a Python loop: the base layer is called once per row and nothing is batched.
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
    rows = []
    for row in range(x.shape[0]):
        y = base(x[row : row + 1])[0]
        n = int(adapter_ids[row])
        if n >= 0:
            scale = alpha / int(ranks[n])
            y = y + scale * ((x[row] @ a_stack[n].transpose(0, 1)) @ b_stack[n].transpose(0, 1))
        rows.append(y)
    return torch.stack(rows)
