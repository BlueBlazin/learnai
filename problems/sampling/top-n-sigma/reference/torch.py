import torch


def top_n_sigma_filter(logits: torch.Tensor, n: float) -> torch.Tensor:
    if n <= 0:
        return logits
    # The statistics are over the finite logits only: a masked entry is not a small
    # number, it is absent. Zero it out of the sums and divide by the finite count.
    finite = torch.isfinite(logits)
    count = finite.sum(dim=-1, keepdim=True)
    mean = torch.where(finite, logits, 0.0).sum(dim=-1, keepdim=True) / count
    squares = torch.where(finite, (logits - mean) ** 2, 0.0)
    std = torch.sqrt(squares.sum(dim=-1, keepdim=True) / count)  # population: divide by |F|
    # A band of width n * std below the row maximum. `>=` keeps the maximum itself when
    # std is 0, and -inf entries fail the comparison and stay -inf.
    cutoff = torch.amax(logits, dim=-1, keepdim=True) - n * std
    return torch.where(logits >= cutoff, logits, float("-inf"))
