# BUG: when several top choices tie for least likely, keeps all of them (the behavior of mlx-lm) instead of only the one with the highest index.
import torch


def xtc_filter(
    logits: torch.Tensor, threshold: float, probability: float, u: torch.Tensor
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    above = probs >= threshold  # the "top choices" of each row
    fires = (above.sum(dim=-1, keepdim=True) >= 2) & (u[:, None] < probability)
    floor = torch.where(above, logits, float("inf")).amin(dim=-1, keepdim=True)
    candidate = above & (logits == floor)
    remove = fires & above & ~candidate
    return torch.where(remove, float("-inf"), logits)
