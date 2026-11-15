# BUG: the batch is split into one sub-batch per adapter and each sub-batch goes through the base layer separately, so the shared matmul is fragmented into as many calls as there are adapters in the batch.
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
    out = None
    for n in torch.unique(adapter_ids).tolist():
        rows = torch.nonzero(adapter_ids == n)[:, 0]
        part = x[rows]
        y = base(part)
        if n >= 0:
            scale = alpha / int(ranks[n])
            y = y + scale * ((part @ a_stack[n].transpose(0, 1)) @ b_stack[n].transpose(0, 1))
        if out is None:
            out = y.new_zeros((x.shape[0], *y.shape[1:]))
        out[rows] = y
    return out
