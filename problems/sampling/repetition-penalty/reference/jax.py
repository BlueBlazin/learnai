import jax
import jax.numpy as jnp


def apply_repetition_penalty(
    logits: jax.Array, prev_tokens: jax.Array, penalty: float, pad_id: int
) -> jax.Array:
    # seen[b, v]: does token v occur in row b's history? One scatter along the vocab
    # axis marks every history entry of every row. A token that occurs several times is
    # written several times with the same value, so it is still penalized once, and the
    # value written for a padding entry is False, so pad_id is never marked.
    rows = jnp.arange(logits.shape[0])[:, None]
    seen = jnp.zeros(logits.shape, dtype=bool).at[rows, prev_tokens].set(prev_tokens != pad_id)
    # Push the logit toward "less likely" whatever its sign.
    penalized = jnp.where(logits > 0, logits / penalty, logits * penalty)
    return jnp.where(seen, penalized, logits)
