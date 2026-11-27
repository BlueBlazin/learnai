import jax
import jax.numpy as jnp


def one_hot_rows(ids: jax.Array, V: int) -> jax.Array:
    # Paired indices: entry (b, ids[b]) of each row. .at[...].set returns a new array.
    rows = jnp.arange(ids.shape[0])
    return jnp.zeros((ids.shape[0], V), dtype=jnp.float32).at[rows, ids].set(1.0)


def token_counts(ids: jax.Array, V: int) -> jax.Array:
    # rows is (B, 1) and broadcasts against ids (B, T): one (row, column) pair per
    # token. .add accumulates the values that land on one entry; .set would keep a
    # single 1 for a repeated token.
    rows = jnp.arange(ids.shape[0])[:, None]
    return jnp.zeros((ids.shape[0], V), dtype=jnp.int32).at[rows, ids].add(1)


def write_at(x: jax.Array, idx: jax.Array, values: jax.Array) -> jax.Array:
    # out[b, idx[b, k]] = values[b, k]; every other entry is copied from x.
    rows = jnp.arange(x.shape[0])[:, None]
    return x.at[rows, idx].set(values)
