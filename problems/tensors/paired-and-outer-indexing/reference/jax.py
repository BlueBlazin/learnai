import jax
import jax.numpy as jnp


def pick_pairs(x: jax.Array, rows: jax.Array, cols: jax.Array) -> jax.Array:
    # Two index tensors of the same shape are read together, entry by entry.
    return x[rows, cols]


def pick_block(x: jax.Array, rows: jax.Array, cols: jax.Array) -> jax.Array:
    # Index tensors are broadcast against each other before they are paired:
    # (R, 1) and (1, C) broadcast to (R, C), every row index against every column index.
    return x[rows[:, None], cols[None, :]]


def pick_per_row(x: jax.Array, cols: jax.Array) -> jax.Array:
    # The row numbers as a column, (B, 1), broadcast against cols, (B, K): entry
    # [b, k] of the pair is (b, cols[b, k]). The D axis is not indexed and is kept.
    batch = jnp.arange(x.shape[0])
    return x[batch[:, None], cols]
