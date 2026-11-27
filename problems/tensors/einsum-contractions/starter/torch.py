import torch


def attention_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """The dot product of every query with every key of the same batch row and head.

    q: float32, shape (B, H, T, D).
    k: float32, shape (B, H, S, D).
    Returns float32, shape (B, H, T, S), with
    out[b, h, t, s] = sum over d of q[b, h, t, d] * k[b, h, s, d].
    """
    raise NotImplementedError


def bilinear(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The bilinear form x[b] @ w @ y[b] for every row b.

    x: float32, shape (B, D).
    w: float32, shape (D, E).
    y: float32, shape (B, E).
    Returns float32, shape (B,), with
    out[b] = sum over d and e of x[b, d] * w[d, e] * y[b, e].
    """
    raise NotImplementedError


def per_head_project(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Multiply the vectors of head h by that head's own matrix w[h].

    x: float32, shape (B, H, T, D).
    w: float32, shape (H, D, E).
    Returns float32, shape (B, H, T, E), with
    out[b, h, t, e] = sum over d of x[b, h, t, d] * w[h, d, e].
    """
    raise NotImplementedError
