import jax
import jax.numpy as jnp


def schedule(n_masked: int, steps: int) -> list[int]:
    # Spread the masked positions evenly; the first n_masked % steps steps take one
    # more, so the counts add up to n_masked and nothing is left for after the end.
    base, extra = divmod(n_masked, steps)
    return [base + 1 if i < extra else base for i in range(steps)]


def unmask_step(
    tokens: jax.Array,
    logits: jax.Array,
    mask_id: int,
    n_unmask: int,
    temperature: float,
    uniforms: jax.Array,
) -> jax.Array:
    masked = tokens == mask_id
    # The model never predicts the mask token itself.
    logits = jnp.where(jnp.arange(logits.shape[-1]) == mask_id, -jnp.inf, logits)

    # A candidate token for every position, from that position's own uniform number.
    if temperature == 0:
        candidates = jnp.argmax(logits, axis=-1)
    else:
        cdf = jnp.cumsum(jax.nn.softmax(logits / temperature, axis=-1), axis=-1)
        # Inverse CDF: the first index whose cumulative sum exceeds u times the total.
        candidates = (cdf <= uniforms[:, None] * cdf[:, -1:]).sum(axis=-1)

    # Confidence: the probability the model itself (temperature 1) gives the candidate
    # that was drawn, not the probability of its favorite token.
    probs = jax.nn.softmax(logits, axis=-1)
    confidence = jnp.take_along_axis(probs, candidates[:, None], axis=-1)[:, 0]
    # A position that is already unmasked can never be chosen.
    confidence = jnp.where(masked, confidence, -jnp.inf)

    # Commit the n most confident masked positions. jnp.argsort is stable, so sorting
    # the negated values puts equal confidences in index order; rank[i] is the place
    # of position i in that order.
    n = min(n_unmask, int(masked.sum()))
    order = jnp.argsort(-confidence)
    rank = jnp.argsort(order)
    return jnp.where(rank < n, candidates, tokens)
