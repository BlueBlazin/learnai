# BUG: computes attention with matmul and softmax instead of calling the fused kernel; the numbers are right and none of the fused implementations can be used.
import torch


def attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_cached: int,
    key_padding: torch.Tensor | None,
) -> torch.Tensor:
    steps, slots = q.shape[-2], k.shape[-2]
    groups = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
    rows = torch.arange(steps, device=q.device)
    cols = torch.arange(slots, device=q.device)
    mask = cols[None, :] <= n_cached + rows[:, None]
    if key_padding is not None:
        mask = mask & ~key_padding[:, None, None, :]
    scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v
