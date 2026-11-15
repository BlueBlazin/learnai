# BUG: expands the key/value heads with repeat, which tiles them (0, 1, 0, 1); grouped-query attention pairs consecutive query heads with one key/value head (0, 0, 1, 1).
import torch
import torch.nn.functional as F


def attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_cached: int,
    key_padding: torch.Tensor | None,
) -> torch.Tensor:
    steps, slots = q.shape[-2], k.shape[-2]
    groups = q.shape[1] // k.shape[1]
    k, v = k.repeat(1, groups, 1, 1), v.repeat(1, groups, 1, 1)
    if n_cached == 0 and key_padding is None:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    rows = torch.arange(steps, device=q.device)
    cols = torch.arange(slots, device=q.device)
    mask = cols[None, :] <= n_cached + rows[:, None]
    if key_padding is not None:
        mask = mask & ~key_padding[:, None, None, :]
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
