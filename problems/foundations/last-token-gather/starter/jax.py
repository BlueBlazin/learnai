import jax
import jax.numpy as jnp


def last_token_logits(logits: jax.Array, lengths: jax.Array, padding_side: str) -> jax.Array:
    """For each row of a padded batch, the logits at its last real token.

    logits: float32 array, shape (B, T, V); entries may be -inf.
    lengths: integer array, shape (B,). Row b holds lengths[b] real tokens, 1 <= lengths[b] <= T.
    padding_side: "right" (real tokens first) or "left" (real tokens last).
    Returns float32, shape (B, V): row b is logits[b, p] with p the position of the
    last real token of row b.
    """
    raise NotImplementedError
