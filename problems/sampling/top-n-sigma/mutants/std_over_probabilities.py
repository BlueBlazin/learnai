# BUG: the band is measured on probabilities (top probability minus n standard deviations of the softmax) instead of on logits, so the kept set changes with temperature.
import torch


def top_n_sigma_filter(logits: torch.Tensor, n: float) -> torch.Tensor:
    if n <= 0:
        return logits
    probs = torch.softmax(logits, dim=-1)
    finite = torch.isfinite(logits)
    count = finite.sum(dim=-1, keepdim=True)
    mean = torch.where(finite, probs, 0.0).sum(dim=-1, keepdim=True) / count
    squares = torch.where(finite, (probs - mean) ** 2, 0.0)
    std = torch.sqrt(squares.sum(dim=-1, keepdim=True) / count)
    cutoff = torch.amax(probs, dim=-1, keepdim=True) - n * std
    return torch.where(probs >= cutoff, logits, float("-inf"))
