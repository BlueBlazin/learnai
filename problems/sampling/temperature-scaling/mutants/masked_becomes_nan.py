# BUG: builds the greedy row as `logits * 0` plus a mask; a masked logit gives -inf * 0 = NaN, so -inf entries of a greedy row come out as NaN.
import torch


def apply_temperature(logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    t = temperature[:, None]
    greedy = t == 0
    scaled = logits / torch.where(greedy, 1.0, t)
    top = torch.argmax(logits, dim=-1, keepdim=True)
    is_top = torch.arange(logits.shape[-1]) == top
    one_hot = logits * 0.0 + torch.where(is_top, 0.0, float("-inf"))
    return torch.where(greedy, one_hot, scaled)
