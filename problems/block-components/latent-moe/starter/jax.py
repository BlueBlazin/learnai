import jax
import jax.numpy as jnp
import numpy as np

from learnai.kit import CountingExperts, ops

Weights = tuple[jax.Array, jax.Array, jax.Array]


def latent_moe(
    x: jax.Array,
    w_router: jax.Array,
    w_down: jax.Array,
    w_up: jax.Array,
    norm_gain: jax.Array,
    experts: CountingExperts,
    shared_w: Weights,
    k: int,
) -> jax.Array:
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
