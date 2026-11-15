# BUG: the embeddings are multiplied by sqrt(d), Gemma's constant, whatever `scale` says.
import torch


def embed(tokens: torch.Tensor, table: torch.Tensor, scale: float) -> torch.Tensor:
    return table[tokens] * table.shape[-1] ** 0.5


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
        logits = softcap * torch.tanh(logits / softcap)
    return logits
