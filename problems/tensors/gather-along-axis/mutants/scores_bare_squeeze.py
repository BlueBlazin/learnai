# BUG: token_scores removes the size-1 axis left by the gather with a bare squeeze(), which also removes the batch axis when B == 1 and the position axis when T == 1.
import torch


def token_scores(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return torch.gather(logits, 2, ids[:, :, None]).squeeze()


def take_columns(values: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return torch.gather(values, 1, idx)


def reorder_steps(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    index = perm[:, :, None].expand(-1, -1, x.shape[2])
    return torch.gather(x, 1, index)
