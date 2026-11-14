import jax
import jax.numpy as jnp

from whetstone.kit import ops


def moe(
    x: jax.Array, w_router: jax.Array, experts_w: tuple[jax.Array, jax.Array, jax.Array], k: int
) -> jax.Array:
    w_gate, w_up, w_down = experts_w
    logits = x @ w_router.T  # (B, T, E): one score per token and expert
    probs = ops.softmax(logits)  # over the experts
    # The rank of every expert for every token: 0 for the largest logit. The sort is
    # stable, so equal logits are ranked by expert index, lowest first.
    order = jnp.argsort(-logits, axis=-1)  # stable
    rank = jnp.argsort(order, axis=-1)
    # Keep the k best experts of each token and renormalize what is left to sum to 1.
    weights = jnp.where(rank < k, probs, 0.0)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    # Dense evaluation: every expert runs on every token, and the weights (zero for
    # the experts a token did not select) pick out the k that count.
    y = jnp.zeros_like(x)
    for e in range(w_router.shape[0]):
        y = y + weights[..., e : e + 1] * ops.swiglu(x, w_gate[e], w_up[e], w_down[e])
    return y
