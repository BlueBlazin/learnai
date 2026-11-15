# BUG: converts the boolean mask to an additive float mask (0 to attend, -inf to hide) before the call; the statement asks for the boolean form, which is a quarter of the size in float32 and independent of the dtype of q.
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
    rows = torch.arange(steps, device=q.device)
    cols = torch.arange(slots, device=q.device)
    mask = cols[None, :] <= n_cached + rows[:, None]
    if key_padding is not None:
        mask = mask & ~key_padding[:, None, None, :]
    bias = torch.zeros(mask.shape, dtype=q.dtype, device=q.device)
    bias = bias.masked_fill(~mask, float("-inf"))
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias, enable_gqa=True)
