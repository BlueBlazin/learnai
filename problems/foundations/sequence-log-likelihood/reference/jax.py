import jax
import jax.numpy as jnp


def _pair_logprobs(
    logits: jax.Array, tokens: jax.Array, mask: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Log-probability of each counted (context t, target t + 1) pair, 0 elsewhere."""
    # logits[:, t] predicts tokens[:, t + 1]: drop the last position's logits (they
    # predict a token that is not in the batch) and the first token (nothing predicts it).
    logp = jax.nn.log_softmax(logits[:, :-1], axis=-1)  # (B, T - 1, V)
    targets = tokens[:, 1:]  # (B, T - 1)
    picked = jnp.take_along_axis(logp, targets[:, :, None], axis=2)[:, :, 0]  # (B, T - 1)
    # A pair counts when both of its positions are real tokens.
    counted = mask[:, :-1] & mask[:, 1:]
    # where, not multiplication: an uncounted entry may be -inf, and 0 * -inf is NaN.
    return jnp.where(counted, picked, 0.0), counted


def sequence_logprob(logits: jax.Array, tokens: jax.Array, mask: jax.Array) -> jax.Array:
    picked, _ = _pair_logprobs(logits, tokens, mask)
    return picked.sum(axis=1)


def perplexity(logits: jax.Array, tokens: jax.Array, mask: jax.Array) -> jax.Array:
    picked, counted = _pair_logprobs(logits, tokens, mask)
    # exp of the mean negative log-probability per counted pair, not of the sum.
    return jnp.exp(-picked.sum(axis=1) / counted.sum(axis=1))
