# BUG: a greedy row keeps every entry equal to the row maximum instead of only the one with the lowest index, so tied rows are not one-hot.
import torch


def apply_temperature(logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    t = temperature[:, None]
    greedy = t == 0
    scaled = logits / torch.where(greedy, 1.0, t)
    is_top = logits == torch.amax(logits, dim=-1, keepdim=True)
    one_hot = torch.where(is_top, 0.0, float("-inf"))
    return torch.where(greedy, one_hot, scaled)
