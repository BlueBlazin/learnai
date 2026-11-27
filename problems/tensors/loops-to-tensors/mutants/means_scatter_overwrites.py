# BUG: segment_means writes the rows with an indexed assignment, which keeps one row per segment instead of adding them, so a segment with several rows gets one row divided by its count.
import torch


def segment_means(x: torch.Tensor, seg: torch.Tensor, S: int) -> torch.Tensor:
    sums = torch.zeros(S, x.shape[1], dtype=x.dtype)
    sums[seg] = x
    counts = torch.zeros(S, dtype=x.dtype).index_add_(0, seg, torch.ones_like(x[:, 0]))
    return sums / counts.clamp(min=1.0)[:, None]


def common_prefix_len(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    still_equal = torch.cumprod((a == b).to(torch.int64), dim=1)
    return still_equal.sum(dim=1)


def repeated_bigram_ban(ids: torch.Tensor, V: int) -> torch.Tensor:
    first = ids[:, :-1]
    second = ids[:, 1:]
    starts_with_last = (first == ids[:, -1:]).to(torch.int64)
    counts = torch.zeros(ids.shape[0], V, dtype=torch.int64)
    counts.scatter_add_(1, second, starts_with_last)
    return counts > 0
