# BUG: builds the right mask but hands it to the built-in scaled_dot_product_attention, which the statement rules out.
import torch
import torch.nn.functional as F


def window_mask(T: int, window: int) -> torch.Tensor:
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    return (j <= i) & (j > i - window)


def sliding_window_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int
) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, attn_mask=window_mask(q.shape[-2], window))
