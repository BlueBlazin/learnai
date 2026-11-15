# BUG: one mean and one standard deviation for the whole batch instead of one per row.
import torch


def top_n_sigma_filter(logits: torch.Tensor, n: float) -> torch.Tensor:
    if n <= 0:
        return logits
    finite = torch.isfinite(logits)
    count = finite.sum()
    mean = torch.where(finite, logits, 0.0).sum() / count
    squares = torch.where(finite, (logits - mean) ** 2, 0.0)
    std = torch.sqrt(squares.sum() / count)
    cutoff = torch.amax(logits, dim=-1, keepdim=True) - n * std
    return torch.where(logits >= cutoff, logits, float("-inf"))
