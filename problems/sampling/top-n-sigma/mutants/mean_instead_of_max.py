# BUG: the band hangs below the mean of the finite logits instead of below their maximum.
import torch


def top_n_sigma_filter(logits: torch.Tensor, n: float) -> torch.Tensor:
    if n <= 0:
        return logits
    finite = torch.isfinite(logits)
    count = finite.sum(dim=-1, keepdim=True)
    mean = torch.where(finite, logits, 0.0).sum(dim=-1, keepdim=True) / count
    squares = torch.where(finite, (logits - mean) ** 2, 0.0)
    std = torch.sqrt(squares.sum(dim=-1, keepdim=True) / count)
    cutoff = mean - n * std
    return torch.where(logits >= cutoff, logits, float("-inf"))
