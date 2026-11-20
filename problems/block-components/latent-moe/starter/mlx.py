import mlx.core as mx
import numpy as np

from learnai.kit import CountingExperts, ops

Weights = tuple[mx.array, mx.array, mx.array]


def latent_moe(
    x: mx.array,
    w_router: mx.array,
    w_down: mx.array,
    w_up: mx.array,
    norm_gain: mx.array,
    experts: CountingExperts,
    shared_w: Weights,
    k: int,
) -> mx.array:
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
