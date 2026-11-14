import jax
import jax.numpy as jnp


def apply_temperature(logits: jax.Array, temperature: jax.Array) -> jax.Array:
    t = temperature[:, None]  # (B, 1): one temperature per row, broadcast over the vocabulary
    greedy = t == 0
    # Greedy rows are divided by 1 instead of 0, so no inf or NaN is ever computed, not
    # even in a branch that is thrown away below.
    scaled = logits / jnp.where(greedy, 1.0, t)
    # Greedy rows: the first maximum (argmax returns the lowest index among ties) keeps
    # logit 0 and every other entry is masked. Softmax of that row is exactly one-hot.
    top = jnp.argmax(logits, axis=-1, keepdims=True)
    is_top = jnp.arange(logits.shape[-1]) == top
    one_hot = jnp.where(is_top, 0.0, -jnp.inf)
    return jnp.where(greedy, one_hot, scaled)
