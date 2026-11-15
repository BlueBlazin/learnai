# BUG: the cap is a hard clip to [-softcap, softcap] instead of softcap * tanh(logits / softcap).
import torch


def embed(tokens: torch.Tensor, table: torch.Tensor, scale: float) -> torch.Tensor:
    # Scale the rows that were looked up, not the table: a tied head reads the same
    # matrix and must see it unscaled.
    return table[tokens] * scale


def finalize_logits(
    h: torch.Tensor,
    w_out: torch.Tensor,
    logit_scale: float,
    softcap: float | None = None,
) -> torch.Tensor:
    # w_out is (V, d), so the projection is h @ W^T: (..., d) -> (..., V).
    logits = h @ w_out.T / logit_scale  # logit_scale is a divisor
    if softcap is not None:
        logits = torch.clamp(logits, -softcap, softcap)
    return logits
