# BUG: write_at uses the columns of row 0 for every row, x[:, idx[0]] = values; it is right only when all rows of idx are equal.
import torch


def one_hot_rows(ids: torch.Tensor, V: int) -> torch.Tensor:
    out = torch.zeros(ids.shape[0], V, dtype=torch.float32)
    return out.scatter(1, ids[:, None], 1.0)


def token_counts(ids: torch.Tensor, V: int) -> torch.Tensor:
    counts = torch.zeros(ids.shape[0], V, dtype=torch.int64)
    return counts.scatter_add(1, ids, torch.ones_like(ids))


def write_at(x: torch.Tensor, idx: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    out[:, idx[0]] = values
    return out
