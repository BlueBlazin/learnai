import torch


def gdn_gates(
    a: torch.Tensor, b: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The decay and the write strength of each head, from their raw projections.

    a, b: float32, shape (T, H). One row per token, one column per head.
    A_log, dt_bias: float32, shape (H,).
    Returns (g, beta), both float32 of shape (T, H):
        g = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b).
    """
    raise NotImplementedError


def gdn_step(
    S: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One token of the gated delta rule: decay, write, then read.

    S: float32, shape (H, Dv, Dk). The memory of each head; S @ key is a value.
    q, k: float32, shape (H, Dk). Used as given (not normalized here).
    v: float32, shape (H, Dv).
    g, beta: float32, shape (H,). One row of the gates.
    Returns (y, S): y float32 of shape (H, Dv), read from the updated memory, and
    the updated memory, float32 of shape (H, Dv, Dk).
    """
    raise NotImplementedError
