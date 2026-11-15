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

    # A square prefill with nothing to hide: the kernel builds the triangle itself.
    # is_causal aligns query i with key i, which is only right without a cache.
    if n_cached == 0 and key_padding is None:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

    # Otherwise one boolean mask, True = may attend, built on the device of q.
    # New token i sits in slot n_cached + i and sees every slot up to its own.
    rows = torch.arange(steps, device=q.device)
    cols = torch.arange(slots, device=q.device)
    mask = cols[None, :] <= n_cached + rows[:, None]  # (T, S)
    if key_padding is not None:
        # key_padding is True where a slot must be hidden: the opposite convention.
        mask = mask & ~key_padding[:, None, None, :]  # (B, 1, T, S)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
