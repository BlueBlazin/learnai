import jax
import jax.numpy as jnp


def next_token_logits(hidden: jax.Array, w_out: jax.Array) -> jax.Array:
    """Logits for the token that follows each sequence: the last position only.

    hidden: float32, shape (B, T, D). Final hidden states, one vector per position.
    w_out: float32, shape (V, D). Output (unembedding) matrix, (out, in) layout.
    Returns float32, shape (B, V): row b is w_out applied to hidden[b, T - 1].
    """
    raise NotImplementedError
