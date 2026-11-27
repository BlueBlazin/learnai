# BUG: take_columns gathers along axis 0 (the batch), returning values[idx[b, k], k]; it raises when an index is not a valid row number.
import torch


def token_scores(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return torch.gather(logits, 2, ids[:, :, None]).squeeze(2)


def take_columns(values: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return torch.gather(values, 0, idx)


def reorder_steps(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    index = perm[:, :, None].expand(-1, -1, x.shape[2])
    return torch.gather(x, 1, index)
