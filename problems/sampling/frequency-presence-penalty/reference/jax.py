import jax
import jax.numpy as jnp


def count_generated(tokens: jax.Array, prompt_lengths: jax.Array, vocab_size: int) -> jax.Array:
    # generated[b, t]: is position t of row b past that row's prompt?
    generated = jnp.arange(tokens.shape[1]) >= prompt_lengths[:, None]
    # A histogram per row: add 1 at column tokens[b, t] for every generated position and
    # 0 for every prompt position. `.at[...].add` accumulates repeated indices, which is
    # the point: a token generated three times must count three.
    rows = jnp.arange(tokens.shape[0])[:, None]
    counts = jnp.zeros((tokens.shape[0], vocab_size), dtype=jnp.int32)
    return counts.at[rows, tokens].add(generated.astype(jnp.int32))


def apply_penalties(
    logits: jax.Array,
    token_counts: jax.Array,
    frequency_penalty: float,
    presence_penalty: float,
) -> jax.Array:
    # Frequency grows with every occurrence; presence is charged once, as soon as the
    # token has occurred at all. Both are subtracted, whatever the sign of the logit.
    penalty = token_counts * frequency_penalty + (token_counts > 0) * presence_penalty
    return logits - penalty
