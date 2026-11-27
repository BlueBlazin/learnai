# BUG: reorder_steps gathers along axis 0, so perm picks a batch row instead of a step: out[b, t] = x[perm[b, t], t]; it raises when T > B.
import torch


def token_scores(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return torch.gather(logits, 2, ids[:, :, None]).squeeze(2)


def take_columns(values: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return torch.gather(values, 1, idx)


def reorder_steps(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    index = perm[:, :, None].expand(-1, -1, x.shape[2])
    return torch.gather(x, 0, index)
