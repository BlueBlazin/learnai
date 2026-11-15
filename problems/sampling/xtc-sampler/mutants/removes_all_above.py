# BUG: removes every token at or above the threshold, including the least likely of them, which XTC keeps.
import torch


def xtc_filter(
    logits: torch.Tensor, threshold: float, probability: float, u: torch.Tensor
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    above = probs >= threshold  # the "top choices" of each row
    fires = (above.sum(dim=-1, keepdim=True) >= 2) & (u[:, None] < probability)
    remove = fires & above
    return torch.where(remove, float("-inf"), logits)
