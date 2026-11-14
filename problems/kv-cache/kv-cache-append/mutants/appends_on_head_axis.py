# BUG: concatenates along axis 1 (heads) instead of axis 2 (sequence), as if the layout were (B, T, H, D).
import math

import torch


def attend_step(
    q_new: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    cache: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    if cache is None:
        k, v = k_new, v_new
    else:
        k = torch.cat((cache[0], k_new), dim=1)
        v = torch.cat((cache[1], v_new), dim=1)
    new, total = q_new.shape[2], k.shape[2]
    past = total - new
    visible = torch.arange(total)[None, :] <= past + torch.arange(new)[:, None]
    scores = q_new @ k.transpose(-1, -2) / math.sqrt(q_new.shape[-1])
    weights = torch.softmax(scores.masked_fill(~visible, float("-inf")), dim=-1)
    return weights @ v, (k, v)
