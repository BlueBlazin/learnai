import torch

Quant = tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # (codes, scale, bias)
Cache = tuple[Quant, Quant, torch.Tensor, torch.Tensor]  # (k_quant, v_quant, k_recent, v_recent)


def quantized_append(
    cache: Cache | None,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    bits: int,
    group_size: int,
    keep_recent: int,
) -> Cache:
    """Append T new tokens; quantize every token that leaves the float window.

    cache: None before the first call, else (k_quant, v_quant, k_recent, v_recent).
        k_quant, v_quant: (codes, scale, bias) for the S_q oldest tokens. codes: uint8,
        shape (B, H, S_q, D). scale, bias: float32, shape (B, H, S_q, D // group_size).
        k_recent, v_recent: float32, shape (B, H, S_r, D), the newest tokens, unquantized.
    k_new, v_new: float32, shape (B, H, T, D), T >= 1.
    bits: 2 to 8, codes are 0 .. 2**bits - 1. group_size: divides D.
    keep_recent: the number of newest tokens that stay float (0 or more).
    Returns the new cache, with S_r = min(S, keep_recent) and S_q = S - S_r for the
    S tokens seen so far. S_q or S_r may be 0 (an axis of length 0).
    """
    raise NotImplementedError


def read_cache(cache: Cache) -> tuple[torch.Tensor, torch.Tensor]:
    """The keys and values to attend over: dequantized old tokens, then the float ones.

    Returns (keys, values), float32, shape (B, H, S_q + S_r, D), oldest token first.
    """
    raise NotImplementedError
