# BUG: the causal mask is built as if the chunk started at position 0: new token i sees only keys 0..i of the concatenated sequence instead of 0..S+i.
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
    else:
        k = torch.cat((cache[0], k_new), dim=2)
        v = torch.cat((cache[1], v_new), dim=2)
    new, total = q_new.shape[2], k.shape[2]
    visible = torch.arange(total)[None, :] <= torch.arange(new)[:, None]
    return ops.attention(q_new, k, v, visible), (k, v)
