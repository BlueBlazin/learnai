# BUG: builds the mask from torch.arange without device=, so the mask is a CPU tensor whatever device q is on.
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
    if n_cached == 0 and key_padding is None:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    rows = torch.arange(steps)
    cols = torch.arange(slots)
    mask = cols[None, :] <= n_cached + rows[:, None]
    if key_padding is not None:
        mask = mask & ~key_padding[:, None, None, :]
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
