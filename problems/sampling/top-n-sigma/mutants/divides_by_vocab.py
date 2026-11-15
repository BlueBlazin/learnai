# BUG: masked entries are zeroed out of the sums, but the mean and variance still divide by V instead of by the number of finite logits.
import torch


def top_n_sigma_filter(logits: torch.Tensor, n: float) -> torch.Tensor:
    if n <= 0:
        return logits
    finite = torch.isfinite(logits)
    mean = torch.where(finite, logits, 0.0).mean(dim=-1, keepdim=True)
    squares = torch.where(finite, (logits - mean) ** 2, 0.0)
    std = torch.sqrt(squares.mean(dim=-1, keepdim=True))
    cutoff = torch.amax(logits, dim=-1, keepdim=True) - n * std
    return torch.where(logits >= cutoff, logits, float("-inf"))
