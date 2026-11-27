# BUG: common_prefix_len keeps the position axis through the sum and removes it with a bare squeeze(), which also removes the batch axis of a single row pair.
import torch


def segment_means(x: torch.Tensor, seg: torch.Tensor, S: int) -> torch.Tensor:
    sums = torch.zeros(S, x.shape[1], dtype=x.dtype).index_add_(0, seg, x)
    counts = torch.zeros(S, dtype=x.dtype).index_add_(0, seg, torch.ones_like(x[:, 0]))
    return sums / counts.clamp(min=1.0)[:, None]


def common_prefix_len(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    still_equal = torch.cumprod((a == b).to(torch.int64), dim=1)
    return still_equal.sum(dim=1, keepdim=True).squeeze()


def repeated_bigram_ban(ids: torch.Tensor, V: int) -> torch.Tensor:
    first = ids[:, :-1]
    second = ids[:, 1:]
    starts_with_last = (first == ids[:, -1:]).to(torch.int64)
    counts = torch.zeros(ids.shape[0], V, dtype=torch.int64)
    counts.scatter_add_(1, second, starts_with_last)
    return counts > 0
