# BUG: also acts on a row with a single token above the threshold, and removes it; XTC needs at least two top choices.
import torch


def xtc_filter(
    logits: torch.Tensor, threshold: float, probability: float, u: torch.Tensor
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    above = probs >= threshold  # the "top choices" of each row
    count = above.sum(dim=-1, keepdim=True)
    fires = (count >= 1) & (u[:, None] < probability)
    floor = torch.where(above, logits, float("inf")).amin(dim=-1, keepdim=True)
    index = torch.arange(logits.shape[-1])
    candidate = above & (logits == floor)
    survivor = torch.where(candidate, index, -1).amax(dim=-1, keepdim=True)
    remove = fires & above & ((index != survivor) | (count == 1))
    return torch.where(remove, float("-inf"), logits)
