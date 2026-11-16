# BUG: no mask at all: correct for a single new token, but in a multi-token chunk every new token also attends to the new tokens after it.
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
        k = torch.cat((cache[0], k_new), dim=2)
        v = torch.cat((cache[1], v_new), dim=2)
    return ops.attention(q_new, k, v), (k, v)
