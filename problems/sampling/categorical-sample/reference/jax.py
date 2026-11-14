import jax
import jax.numpy as jnp


def sample(probs: jax.Array, u: jax.Array) -> jax.Array:
    cdf = jnp.cumsum(probs, axis=-1)
    # Scale each row's uniform by that row's total, the last cumulative sum. This is
    # the same as normalizing the row, and it puts the threshold strictly below
    # cdf[:, -1] for every u < 1, whether the row sums to 1, 0.9999, or 1.0001.
    threshold = u[:, None] * cdf[:, -1:]
    # Inverse CDF: the first index whose cumulative sum exceeds the threshold, which is
    # the number of entries that do not exceed it. A zero-probability token repeats the
    # previous cumulative sum, so it is counted or skipped together with its
    # predecessor and is never the first to exceed.
    return jnp.sum(cdf <= threshold, axis=-1)
