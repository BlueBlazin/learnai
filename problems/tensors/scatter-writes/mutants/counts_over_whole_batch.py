# BUG: token_counts counts over the flattened batch and gives every row the totals of all rows; it is right only for a batch of one.
import torch


def one_hot_rows(ids: torch.Tensor, V: int) -> torch.Tensor:
    out = torch.zeros(ids.shape[0], V, dtype=torch.float32)
    return out.scatter(1, ids[:, None], 1.0)


def token_counts(ids: torch.Tensor, V: int) -> torch.Tensor:
    totals = torch.bincount(ids.flatten(), minlength=V)
    return totals[None, :].expand(ids.shape[0], V)


def write_at(x: torch.Tensor, idx: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return x.scatter(1, idx, values)
