import jax
import jax.numpy as jnp


def top_p_filter(logits: jax.Array, p: float) -> jax.Array:
    probs = jax.nn.softmax(logits, axis=-1)
    # argsort is stable, so equal probabilities are ranked by token index, lowest first.
    order = jnp.argsort(-probs, axis=-1)
    sorted_probs = jnp.take_along_axis(probs, order, axis=-1)
    # Mass of the tokens ranked strictly before each one. A token is kept while that
    # mass is still short of p, so the token that crosses p is kept, and rank 0 (mass
    # before it: exactly 0) always survives.
    mass_before = jnp.cumsum(sorted_probs, axis=-1) - sorted_probs
    kept = jnp.where(mass_before < p, sorted_probs, 0.0)
    kept = kept / jnp.sum(kept, axis=-1, keepdims=True)
    # `order[b, k]` is the token at rank k of row b: scatter sends rank k back there.
    rows = jnp.arange(probs.shape[0])[:, None]
    return jnp.zeros_like(probs).at[rows, order].set(kept)
