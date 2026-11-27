# BUG: repeated_bigram_ban writes the match flags with a setting scatter, so for a token that follows the last token in one place and another token elsewhere, one write overwrites the other and the ban can be lost.
import torch


def segment_means(x: torch.Tensor, seg: torch.Tensor, S: int) -> torch.Tensor:
    sums = torch.zeros(S, x.shape[1], dtype=x.dtype).index_add_(0, seg, x)
    counts = torch.zeros(S, dtype=x.dtype).index_add_(0, seg, torch.ones_like(x[:, 0]))
    return sums / counts.clamp(min=1.0)[:, None]


def common_prefix_len(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    still_equal = torch.cumprod((a == b).to(torch.int64), dim=1)
    return still_equal.sum(dim=1)


def repeated_bigram_ban(ids: torch.Tensor, V: int) -> torch.Tensor:
    first = ids[:, :-1]
    second = ids[:, 1:]
    starts_with_last = first == ids[:, -1:]
    banned = torch.zeros(ids.shape[0], V, dtype=torch.bool)
    banned.scatter_(1, second, starts_with_last)
    return banned
