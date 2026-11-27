# BUG: write_at uses scatter_, which writes into x itself: the returned values are right, but the caller's x has been changed.
import torch


def one_hot_rows(ids: torch.Tensor, V: int) -> torch.Tensor:
    out = torch.zeros(ids.shape[0], V, dtype=torch.float32)
    return out.scatter(1, ids[:, None], 1.0)


def token_counts(ids: torch.Tensor, V: int) -> torch.Tensor:
    counts = torch.zeros(ids.shape[0], V, dtype=torch.int64)
    return counts.scatter_add(1, ids, torch.ones_like(ids))


def write_at(x: torch.Tensor, idx: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return x.scatter_(1, idx, values)
