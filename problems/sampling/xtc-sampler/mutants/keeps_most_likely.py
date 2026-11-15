# BUG: keeps the most likely top choice and removes the others; XTC keeps the least likely one.
import torch


def xtc_filter(
    logits: torch.Tensor, threshold: float, probability: float, u: torch.Tensor
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    above = probs >= threshold  # the "top choices" of each row
    fires = (above.sum(dim=-1, keepdim=True) >= 2) & (u[:, None] < probability)
    floor = torch.where(above, logits, float("-inf")).amax(dim=-1, keepdim=True)
    index = torch.arange(logits.shape[-1])
    candidate = above & (logits == floor)
    survivor = torch.where(candidate, index, -1).amax(dim=-1, keepdim=True)
    remove = fires & above & (index != survivor)
    return torch.where(remove, float("-inf"), logits)
