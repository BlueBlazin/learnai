# BUG: copies the keys and values once per query head with repeat_interleave before the kernel call instead of passing them with enable_gqa=True; the numbers are right and the cache is materialized N_q / N_kv times.
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
    k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
    if n_cached == 0 and key_padding is None:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    rows = torch.arange(steps, device=q.device)
    cols = torch.arange(slots, device=q.device)
    mask = cols[None, :] <= n_cached + rows[:, None]
    if key_padding is not None:
        mask = mask & ~key_padding[:, None, None, :]
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
