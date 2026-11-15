# BUG: the cap is computed as softcap * tanh(logits), without dividing the logits by softcap inside the tanh.
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
        # Smoothly squash into (-softcap, softcap); near zero this is the identity.
        logits = softcap * torch.tanh(logits)
    return logits
