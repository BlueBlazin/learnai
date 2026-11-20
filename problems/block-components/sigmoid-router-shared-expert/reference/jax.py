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
    logits = x @ w_router.T  # (N, E)
    # Each expert is scored on its own: no normalization across experts.
    if score_fn == "sigmoid":
        scores = jax.nn.sigmoid(logits)
    elif score_fn == "sqrtsoftplus":
        scores = jnp.sqrt(jax.nn.softplus(logits))
    else:
        raise ValueError(f"unknown score_fn: {score_fn!r}")
    # The bias takes part in the choice only. The sort is stable, so equal biased scores
    # are ranked by expert index, lowest first.
    indices = jnp.argsort(-(scores + bias), axis=-1)[:, :k]  # (N, k); stable
    chosen = jnp.take_along_axis(scores, indices, axis=-1)  # unbiased scores of the chosen
    # Normalize first, scale second: the scale survives, and the weights sum to it.
    weights = chosen / (chosen.sum(axis=-1, keepdims=True) + 1e-20) * scale
    return indices, weights


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
    w_gate, w_up, w_down = experts_w
    n_experts = w_router.shape[0]
    indices, weights = route(x, w_router, bias, k, scale, score_fn)
    # (N, E): each token's weight for every expert, zero for the ones it did not choose.
    one_hot = indices[..., None] == jnp.arange(n_experts)
    dense = (one_hot * weights[..., None]).sum(axis=1)
    y = ops.swiglu(x, *shared_w)  # the shared expert: every token, weight 1, no scale
    for e in range(n_experts):
        y = y + dense[:, e : e + 1] * ops.swiglu(x, w_gate[e], w_up[e], w_down[e])
    return y
