import numpy as np
import torch

from learnai.kit import CountingExperts, ops

Weights = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def latent_moe(
    x: torch.Tensor,
    w_router: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    norm_gain: torch.Tensor,
    experts: CountingExperts,
    shared_w: Weights,
    k: int,
) -> torch.Tensor:
    """A mixture of experts whose routed experts work at a narrower width.

    x: float32, shape (N, D). w_router: float32, shape (E, D).
    w_down: float32, shape (L, D). w_up: float32, shape (D, L). norm_gain: float32, shape (L,).
    experts: CountingExperts of width L; experts(e, rows) maps (n, L) to (n, L) and is counted.
    shared_w: (w_gate, w_up, w_down) of the full-width shared SwiGLU expert:
        shapes (Hs, D), (Hs, D), (D, Hs).
    k: int, 1 <= k <= E.
    Returns float32, shape (N, D).
    """
    raise NotImplementedError


def expert_params(hidden: int, latent: int, inter: int, n_experts: int) -> tuple[int, int]:
    """Parameters of the routed experts: (with the latent projection, without it).

    hidden: model width. latent: width the experts work at. inter: an expert's inner width.
    Returns two Python ints.
    """
    raise NotImplementedError
