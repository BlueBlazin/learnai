# BUG: the new keys and values are put before the cached ones, so the cache is no longer in sequence order and the mask hides the wrong columns.
import torch

from learnai.kit import ops


def attend_step(
    q_new: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    cache: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    if cache is None:
        k, v = k_new, v_new
    else:
        k = torch.cat((k_new, cache[0]), dim=2)
        v = torch.cat((v_new, cache[1]), dim=2)
    new, total = q_new.shape[2], k.shape[2]
    past = total - new
    visible = torch.arange(total)[None, :] <= past + torch.arange(new)[:, None]
    return ops.attention(q_new, k, v, visible), (k, v)
