import jax
import jax.numpy as jnp

Cache = tuple[jax.Array, jax.Array]  # (k, v)


def h2o_step(
    cache: Cache,
    scores: jax.Array | None,
    attn_weights: jax.Array,
    budget: int,
    recent: int,
) -> tuple[Cache, jax.Array, jax.Array]:
    """Accumulate attention scores, then evict down to `budget` tokens per head.

    cache: (k, v), float32, shape (B, H, S, D): the cache after this step's T new
        tokens were appended (they are its last T entries).
    scores: float32, shape (B, H, S - T), the accumulated score of every older entry,
        or None when there is no older entry (S == T).
    attn_weights: float32, shape (B, H, T, S): row i is the attention of new query i
        over the S entries.
    budget: the largest number of entries a head may keep. recent: 0 <= recent <= budget,
        the number of newest entries that are never evicted.
    Returns ((k, v), scores, kept) with N = min(S, budget): k, v of shape (B, H, N, D),
    scores float32 of shape (B, H, N), and kept, an integer tensor of shape (B, H, N):
    the positions (in the cache passed in) that each head keeps, in increasing order.
    """
    raise NotImplementedError
