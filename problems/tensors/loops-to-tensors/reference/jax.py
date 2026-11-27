import jax
import jax.numpy as jnp


def segment_means(x: jax.Array, seg: jax.Array, S: int) -> jax.Array:
    # An adding scatter along axis 0: row n of x is added to row seg[n] of the sums,
    # and a 1 to entry seg[n] of the counts. Rows that share a segment accumulate.
    sums = jnp.zeros((S, x.shape[1]), dtype=x.dtype).at[seg].add(x)
    counts = jnp.zeros((S,), dtype=x.dtype).at[seg].add(1.0)
    # An empty segment has sum 0 and count 0; dividing by 1 there leaves the 0.
    return sums / jnp.maximum(counts, 1.0)[:, None]


def common_prefix_len(a: jax.Array, b: jax.Array) -> jax.Array:
    # The running product of the 0/1 matches is 1 while every position so far has
    # matched and 0 from the first mismatch on, whatever comes after it.
    still_equal = jnp.cumprod((a == b).astype(jnp.int32), axis=1)  # (B, T)
    return still_equal.sum(axis=1)


def repeated_bigram_ban(ids: jax.Array, V: int) -> jax.Array:
    # A pair starts at every position except the final one. Position t starts a pair
    # that bans its second token when its first token is the row's last token.
    first = ids[:, :-1]  # (B, T - 1)
    second = ids[:, 1:]  # (B, T - 1)
    starts_with_last = (first == ids[:, -1:]).astype(jnp.int32)
    # Count, for every token v, the pairs (last, v): an adding scatter of 0s and 1s
    # into column second[b, t]. A setting scatter of False could erase a True.
    rows = jnp.arange(ids.shape[0])[:, None]
    counts = jnp.zeros((ids.shape[0], V), dtype=jnp.int32).at[rows, second].add(starts_with_last)
    return counts > 0
