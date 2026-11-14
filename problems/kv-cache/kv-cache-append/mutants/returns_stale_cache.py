# BUG: attends over the right keys and values but returns the cache it was given, without the new entries; the next step no longer sees these tokens.
import torch

from whetstone.kit import ops


def attend_step(
    q_new: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    cache: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    if cache is None:
        k, v = k_new, v_new
        cache = (k, v)
    else:
        k = torch.cat((cache[0], k_new), dim=2)
        v = torch.cat((cache[1], v_new), dim=2)
    new, total = q_new.shape[2], k.shape[2]
    past = total - new
    visible = torch.arange(total)[None, :] <= past + torch.arange(new)[:, None]
    return ops.attention(q_new, k, v, visible), cache
