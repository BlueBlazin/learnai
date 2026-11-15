import torch
import torch.nn.functional as F


def attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_cached: int,
    key_padding: torch.Tensor | None,
) -> torch.Tensor:
    """Causal attention of T new tokens over a KV cache, with the fused kernel.

    q: (B, N_q, T, D) queries of the new tokens.
    k, v: (B, N_kv, S, D) keys and values, S = n_cached + T: the cached tokens first,
        then the new tokens. N_kv divides N_q (grouped-query attention).
    n_cached: number of cached tokens, >= 0.
    key_padding: None, or a bool tensor (B, S) that is True where a slot is padding.
    Returns (B, N_q, T, D) in the dtype of q, from exactly one call of
    F.scaled_dot_product_attention: with is_causal=True and no mask when n_cached == 0
    and key_padding is None, with a boolean attn_mask otherwise.
    """
    raise NotImplementedError
