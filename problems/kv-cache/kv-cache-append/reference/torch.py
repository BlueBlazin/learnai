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
        # Append along the sequence axis (2), old entries first. `cat` builds new
        # tensors, so the cache that was passed in is left as it was.
        k = torch.cat((cache[0], k_new), dim=2)
        v = torch.cat((cache[1], v_new), dim=2)
    new, total = q_new.shape[2], k.shape[2]
    past = total - new
    # New token i sits at absolute position past + i and may see keys 0 .. past + i:
    # every cached key, and the new keys up to and including its own.
    visible = torch.arange(total)[None, :] <= past + torch.arange(new)[:, None]  # (T, S + T)
    return ops.attention(q_new, k, v, visible), (k, v)
