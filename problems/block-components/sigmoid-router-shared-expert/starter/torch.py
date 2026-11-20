import torch
import torch.nn.functional as F

from learnai.kit import ops

Weights = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def route(
    x: torch.Tensor,
    w_router: torch.Tensor,
    bias: torch.Tensor,
    k: int,
    scale: float,
    score_fn: str = "sigmoid",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose k experts per token by biased score; weight them by unbiased score.

    x: float32, shape (N, D). w_router: float32, shape (E, D). bias: float32, shape (E,).
    k: int, 1 <= k <= E. scale: float, applied after the normalization.
    score_fn: "sigmoid" or "sqrtsoftplus"; anything else raises ValueError.
    Returns (indices, weights): integer and float32, both of shape (N, k), best first.
    """
    raise NotImplementedError


def moe_shared(
    x: torch.Tensor,
    w_router: torch.Tensor,
    bias: torch.Tensor,
    experts_w: Weights,
    shared_w: Weights,
    k: int,
    scale: float,
    score_fn: str = "sigmoid",
) -> torch.Tensor:
    """The routed experts, weighted as `route` says, plus the shared expert unweighted.

    experts_w: (w_gate, w_up, w_down) of shapes (E, H, D), (E, H, D), (E, D, H).
    shared_w: (w_gate, w_up, w_down) of shapes (Hs, D), (Hs, D), (D, Hs).
    Returns float32, shape (N, D).
    """
    raise NotImplementedError
