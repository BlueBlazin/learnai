# BUG: when there is a cache, the new queries attend over the cached entries only; the new keys and values are appended afterwards, so no new token sees itself or the new tokens before it.
import torch

from whetstone.kit import ops


def attend_step(
    q_new: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    cache: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    if cache is None:
        new = q_new.shape[2]
        visible = torch.arange(new)[None, :] <= torch.arange(new)[:, None]
        return ops.attention(q_new, k_new, v_new, visible), (k_new, v_new)
    out = ops.attention(q_new, cache[0], cache[1])
    k = torch.cat((cache[0], k_new), dim=2)
    v = torch.cat((cache[1], v_new), dim=2)
    return out, (k, v)
