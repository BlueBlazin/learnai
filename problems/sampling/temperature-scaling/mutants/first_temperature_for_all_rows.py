# BUG: reads one temperature, the first row's, and applies it to the whole batch instead of using each row's own.
import torch


def apply_temperature(logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    t = temperature[0]
    greedy = t == 0
    scaled = logits / torch.where(greedy, 1.0, t)
    top = torch.argmax(logits, dim=-1, keepdim=True)
    is_top = torch.arange(logits.shape[-1]) == top
    one_hot = torch.where(is_top, 0.0, float("-inf"))
    return torch.where(greedy, one_hot, scaled)
