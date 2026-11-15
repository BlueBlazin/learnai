import jax
import jax.numpy as jnp


def verify_batch(
    p: jax.Array,
    q: jax.Array,
    draft_tokens: jax.Array,
    u_accept: jax.Array,
    u_resample: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    b, k = draft_tokens.shape
    index = draft_tokens[:, :, None]
    p_x = jnp.take_along_axis(p[:, :k], index, axis=2)[:, :, 0]  # (B, K): p[b, i, x_bi]
    q_x = jnp.take_along_axis(q, index, axis=2)[:, :, 0]
    passes = u_accept * q_x < p_x
    # Per row, the accepted prefix ends at the first failed test: the running product
    # along the draft axis is 1 up to it and 0 afterwards.
    n_accepted = jnp.sum(jnp.cumprod(passes.astype(jnp.int32), axis=1), axis=1)  # (B,)
    # Each row needs the residual of its own stopping position. A row of zeros for
    # q[:, K] makes the bonus case (weights p[b, K]) the same expression.
    q_ext = jnp.concatenate([q, jnp.zeros_like(q[:, :1])], axis=1)
    rows = jnp.arange(b)
    weights = jnp.maximum(p[rows, n_accepted] - q_ext[rows, n_accepted], 0.0)  # (B, V)
    # Inverse CDF per row, each with its own uniform and its own total.
    cdf = jnp.cumsum(weights, axis=1)
    threshold = u_resample[:, None] * cdf[:, -1:]
    next_tokens = jnp.sum(cdf <= threshold, axis=1)
    return n_accepted, next_tokens


def advance_batch(
    tokens: jax.Array,
    lengths: jax.Array,
    draft_tokens: jax.Array,
    n_accepted: jax.Array,
    next_tokens: jax.Array,
    pad_id: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    cache_lengths = lengths + n_accepted  # tokens whose cache entries stay valid
    new_lengths = cache_lengths + 1  # the next token is appended, not yet in any cache
    width = int(jnp.max(new_lengths))
    cols = jnp.arange(width)[None, :]  # (1, W), compared with per-row boundaries (B, 1)
    start, stop = lengths[:, None], cache_lengths[:, None]
    # What each source would put in column c; indices are clamped so that every
    # gather is in range, and `where` picks the source that actually owns the column.
    old_index = jnp.broadcast_to(jnp.minimum(cols, tokens.shape[1] - 1), (len(lengths), width))
    old = jnp.take_along_axis(tokens, old_index, axis=1)
    draft_index = jnp.clip(cols - start, 0, draft_tokens.shape[1] - 1)
    drafts = jnp.take_along_axis(draft_tokens, draft_index, axis=1)
    new_tokens = jnp.full_like(old, pad_id)
    new_tokens = jnp.where(cols == stop, next_tokens[:, None], new_tokens)
    new_tokens = jnp.where((cols >= start) & (cols < stop), drafts, new_tokens)
    new_tokens = jnp.where(cols < start, old, new_tokens)
    return new_tokens, new_lengths, cache_lengths
