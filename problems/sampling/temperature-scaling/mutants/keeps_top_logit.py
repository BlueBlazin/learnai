# BUG: a greedy row keeps the original value of its largest logit instead of setting it to 0.
import torch


def apply_temperature(logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    t = temperature[:, None]
    greedy = t == 0
    scaled = logits / torch.where(greedy, 1.0, t)
    top = torch.argmax(logits, dim=-1, keepdim=True)
    is_top = torch.arange(logits.shape[-1]) == top
    one_hot = torch.where(is_top, logits, float("-inf"))
    return torch.where(greedy, one_hot, scaled)
