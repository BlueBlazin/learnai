# BUG: token_scores gathers along axis 1 (the positions) instead of axis 2 (the vocabulary), returning logits[b, ids[b, t], 0]; it raises when an id is not a valid position.
import torch


def token_scores(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return torch.gather(logits, 1, ids[:, :, None]).squeeze(2)


def take_columns(values: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return torch.gather(values, 1, idx)


def reorder_steps(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    index = perm[:, :, None].expand(-1, -1, x.shape[2])
    return torch.gather(x, 1, index)
