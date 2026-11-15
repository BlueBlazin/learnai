# BUG: removes only the single most likely token instead of every top choice but the last.
import torch


def xtc_filter(
    logits: torch.Tensor, threshold: float, probability: float, u: torch.Tensor
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    above = probs >= threshold  # the "top choices" of each row
    fires = (above.sum(dim=-1, keepdim=True) >= 2) & (u[:, None] < probability)
    index = torch.arange(logits.shape[-1])
    top = torch.argmax(logits, dim=-1, keepdim=True)
    remove = fires & above & (index == top)
    return torch.where(remove, float("-inf"), logits)
