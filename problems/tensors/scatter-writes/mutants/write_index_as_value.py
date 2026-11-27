# BUG: write_at swaps the meaning of its arguments' contents: it writes the column numbers themselves, out[b, idx[b, k]] = idx[b, k], and never uses values.
import torch


def one_hot_rows(ids: torch.Tensor, V: int) -> torch.Tensor:
    out = torch.zeros(ids.shape[0], V, dtype=torch.float32)
    return out.scatter(1, ids[:, None], 1.0)


def token_counts(ids: torch.Tensor, V: int) -> torch.Tensor:
    counts = torch.zeros(ids.shape[0], V, dtype=torch.int64)
    return counts.scatter_add(1, ids, torch.ones_like(ids))


def write_at(x: torch.Tensor, idx: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return x.scatter(1, idx, idx.to(x.dtype))
