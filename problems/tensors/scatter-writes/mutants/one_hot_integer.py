# BUG: one_hot_rows returns the int64 table of a comparison instead of float32.
import torch


def one_hot_rows(ids: torch.Tensor, V: int) -> torch.Tensor:
    return (ids[:, None] == torch.arange(V)[None, :]).to(torch.int64)


def token_counts(ids: torch.Tensor, V: int) -> torch.Tensor:
    counts = torch.zeros(ids.shape[0], V, dtype=torch.int64)
    return counts.scatter_add(1, ids, torch.ones_like(ids))


def write_at(x: torch.Tensor, idx: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return x.scatter(1, idx, values)
