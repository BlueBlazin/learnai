import jax
import jax.numpy as jnp


def last_token_logits(logits: jax.Array, lengths: jax.Array, padding_side: str) -> jax.Array:
    batch, seq, _ = logits.shape
    # The position of the last real token in each row.
    if padding_side == "right":
        last = lengths - 1  # real tokens are 0 .. length - 1
    elif padding_side == "left":
        last = jnp.full_like(lengths, seq - 1)  # real tokens end at the last column
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}")
    # Pair row b with position last[b]: two index arrays of shape (B,) pick one
    # (V,)-vector per row. A slice (logits[:, last]) would pick every position for
    # every row instead.
    rows = jnp.arange(batch)
    return logits[rows, last]
