import torch


def xtc_filter(
    logits: torch.Tensor, threshold: float, probability: float, u: torch.Tensor
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    above = probs >= threshold  # the "top choices" of each row
    # XTC acts only when a row has at least two top choices and its coin says so.
    fires = (above.sum(dim=-1, keepdim=True) >= 2) & (u[:, None] < probability)
    # The survivor is the least likely top choice: the smallest logit among them, and
    # of equal logits the one with the highest index.
    floor = torch.where(above, logits, float("inf")).amin(dim=-1, keepdim=True)
    index = torch.arange(logits.shape[-1])
    candidate = above & (logits == floor)
    survivor = torch.where(candidate, index, -1).amax(dim=-1, keepdim=True)
    # Remove every other top choice. Tokens below the threshold are not touched.
    remove = fires & above & (index != survivor)
    return torch.where(remove, float("-inf"), logits)
