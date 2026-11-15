import jax
import jax.numpy as jnp


def xtc_filter(logits: jax.Array, threshold: float, probability: float, u: jax.Array) -> jax.Array:
    probs = jax.nn.softmax(logits, axis=-1)
    above = probs >= threshold  # the "top choices" of each row
    # XTC acts only when a row has at least two top choices and its coin says so.
    fires = (jnp.sum(above, axis=-1, keepdims=True) >= 2) & (u[:, None] < probability)
    # The survivor is the least likely top choice: the smallest logit among them, and
    # of equal logits the one with the highest index.
    floor = jnp.min(jnp.where(above, logits, jnp.inf), axis=-1, keepdims=True)
    index = jnp.arange(logits.shape[-1])
    candidate = above & (logits == floor)
    survivor = jnp.max(jnp.where(candidate, index, -1), axis=-1, keepdims=True)
    # Remove every other top choice. Tokens below the threshold are not touched.
    remove = fires & above & (index != survivor)
    return jnp.where(remove, -jnp.inf, logits)
