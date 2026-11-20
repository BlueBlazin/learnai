import jax
import jax.numpy as jnp

from learnai.kit import ops

Weights = tuple[jax.Array, jax.Array, jax.Array]


def route(
    x: jax.Array,
    w_router: jax.Array,
    bias: jax.Array,
    k: int,
    scale: float,
    score_fn: str = "sigmoid",
) -> tuple[jax.Array, jax.Array]:
    """Choose k experts per token by biased score; weight them by unbiased score.

    x: float32, shape (N, D). w_router: float32, shape (E, D). bias: float32, shape (E,).
    k: int, 1 <= k <= E. scale: float, applied after the normalization.
    score_fn: "sigmoid" or "sqrtsoftplus"; anything else raises ValueError.
    Returns (indices, weights): integer and float32, both of shape (N, k), best first.
    """
    raise NotImplementedError


def moe_shared(
    x: jax.Array,
    w_router: jax.Array,
    bias: jax.Array,
    experts_w: Weights,
    shared_w: Weights,
    k: int,
    scale: float,
    score_fn: str = "sigmoid",
) -> jax.Array:
    """The routed experts, weighted as `route` says, plus the shared expert unweighted.

    experts_w: (w_gate, w_up, w_down) of shapes (E, H, D), (E, H, D), (E, D, H).
    shared_w: (w_gate, w_up, w_down) of shapes (Hs, D), (Hs, D), (D, Hs).
    Returns float32, shape (N, D).
    """
    raise NotImplementedError
