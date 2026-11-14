import jax
import jax.numpy as jnp


def apply_temperature(logits: jax.Array, temperature: jax.Array) -> jax.Array:
    """Divide each row of logits by its own temperature; temperature 0 means greedy.

    logits: float32 array of shape (B, V); entries may be -inf (masked).
    temperature: float32 array of shape (B,), one temperature per row, each >= 0.
    Returns a float32 array of shape (B, V). A row with temperature t > 0 is
    logits / t. A row with temperature exactly 0 is 0 at its largest logit (the
    lowest index among ties) and -inf everywhere else.
    """
    raise NotImplementedError
