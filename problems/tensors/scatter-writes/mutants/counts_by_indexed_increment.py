# BUG: token_counts increments with counts[rows, ids] += 1, which reads every entry once, adds 1, and writes it back: a token that occurs several times in a row is counted once.
import torch


def one_hot_rows(ids: torch.Tensor, V: int) -> torch.Tensor:
    out = torch.zeros(ids.shape[0], V, dtype=torch.float32)
    return out.scatter(1, ids[:, None], 1.0)


def token_counts(ids: torch.Tensor, V: int) -> torch.Tensor:
    counts = torch.zeros(ids.shape[0], V, dtype=torch.int64)
    rows = torch.arange(ids.shape[0])[:, None]
    counts[rows, ids] += 1
    return counts


def write_at(x: torch.Tensor, idx: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return x.scatter(1, idx, values)
