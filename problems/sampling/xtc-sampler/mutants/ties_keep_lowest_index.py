# BUG: when several top choices tie for least likely, keeps the one with the lowest index instead of the highest.
import torch


def xtc_filter(
    logits: torch.Tensor, threshold: float, probability: float, u: torch.Tensor
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    above = probs >= threshold  # the "top choices" of each row
    fires = (above.sum(dim=-1, keepdim=True) >= 2) & (u[:, None] < probability)
    floor = torch.where(above, logits, float("inf")).amin(dim=-1, keepdim=True)
    index = torch.arange(logits.shape[-1])
    candidate = above & (logits == floor)
    survivor = torch.where(candidate, index, logits.shape[-1]).amin(dim=-1, keepdim=True)
    remove = fires & above & (index != survivor)
    return torch.where(remove, float("-inf"), logits)
