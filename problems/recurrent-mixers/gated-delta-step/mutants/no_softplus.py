# BUG: the decay is exp(-exp(A_log) * (a + dt_bias)) with no softplus, so g exceeds 1 whenever a + dt_bias is negative and the memory grows.
import torch


def gdn_gates(
    a: torch.Tensor, b: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # softplus is positive and exp(A_log) is positive, so the exponent is negative
    # and g lies in (0, 1): the memory can fade, never grow.
    g = torch.exp(-torch.exp(A_log) * (a + dt_bias))
    beta = torch.sigmoid(b)
    return g, beta


def gdn_step(
    S: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 1. Decay: every stored association fades by g, one factor per head.
    S = S * g[:, None, None]
    # 2. Write. S k is what the decayed memory returns for this key; only the
    #    difference to v is stored, along the direction k.
    predicted = torch.einsum("hvk,hk->hv", S, k)
    delta = beta[:, None] * (v - predicted)
    S = S + delta[:, :, None] * k[:, None, :]  # outer product, (H, Dv, Dk)
    # 3. Read from the updated memory: the token sees its own write.
    y = torch.einsum("hvk,hk->hv", S, q)
    return y, S
