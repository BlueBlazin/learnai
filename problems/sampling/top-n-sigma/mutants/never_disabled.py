# BUG: n <= 0 is not treated as "disabled"; n = 0 keeps only the maximum and a negative n removes every token.
import torch


def top_n_sigma_filter(logits: torch.Tensor, n: float) -> torch.Tensor:
    finite = torch.isfinite(logits)
    count = finite.sum(dim=-1, keepdim=True)
    mean = torch.where(finite, logits, 0.0).sum(dim=-1, keepdim=True) / count
    squares = torch.where(finite, (logits - mean) ** 2, 0.0)
    std = torch.sqrt(squares.sum(dim=-1, keepdim=True) / count)
    cutoff = torch.amax(logits, dim=-1, keepdim=True) - n * std
    return torch.where(logits >= cutoff, logits, float("-inf"))
