import jax
import jax.numpy as jnp


def apply_bias(
    logits: jax.Array,
    bias: dict[int, float],
    allowed_token_ids: list[int] | None,
    bad_words: list[list[int]],
    history: list[int],
) -> jax.Array:
    vocab = logits.shape[0]
    # The request data is a handful of token ids: collect what it says in two Python
    # lists, one entry per token, and apply them to the logits in a single step.
    delta = [0.0] * vocab
    for token, value in bias.items():
        delta[token] = value

    removed = [allowed_token_ids is not None] * vocab
    if allowed_token_ids is not None:
        for token in allowed_token_ids:
            removed[token] = False

    for word in bad_words:
        prefix = word[:-1]
        # Compare with the end of the history. The slice starts at an explicit index:
        # history[-len(prefix):] would be the whole history when the prefix is empty.
        start = len(history) - len(prefix)
        if start >= 0 and history[start:] == prefix:
            removed[word[-1]] = True

    # Removal is decided after everything else and overrides the bias.
    biased = logits + jnp.array(delta, dtype=logits.dtype)
    return jnp.where(jnp.array(removed), -jnp.inf, biased)
