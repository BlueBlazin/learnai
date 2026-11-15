import jax
import jax.numpy as jnp


def top_n_sigma_filter(logits: jax.Array, n: float) -> jax.Array:
    if n <= 0:
        return logits
    # The statistics are over the finite logits only: a masked entry is not a small
    # number, it is absent. Zero it out of the sums and divide by the finite count.
    finite = jnp.isfinite(logits)
    count = jnp.sum(finite, axis=-1, keepdims=True)
    mean = jnp.sum(jnp.where(finite, logits, 0.0), axis=-1, keepdims=True) / count
    squares = jnp.where(finite, (logits - mean) ** 2, 0.0)
    std = jnp.sqrt(jnp.sum(squares, axis=-1, keepdims=True) / count)  # population: divide by |F|
    # A band of width n * std below the row maximum. `>=` keeps the maximum itself when
    # std is 0, and -inf entries fail the comparison and stay -inf.
    cutoff = jnp.max(logits, axis=-1, keepdims=True) - n * std
    return jnp.where(logits >= cutoff, logits, -jnp.inf)
