# BUG: one_hot_rows calls torch.nn.functional.one_hot, which the statement rules out.
import torch


def one_hot_rows(ids: torch.Tensor, V: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(ids, V).to(torch.float32)


def token_counts(ids: torch.Tensor, V: int) -> torch.Tensor:
    counts = torch.zeros(ids.shape[0], V, dtype=torch.int64)
    return counts.scatter_add(1, ids, torch.ones_like(ids))


def write_at(x: torch.Tensor, idx: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return x.scatter(1, idx, values)
