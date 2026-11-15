import mlx.core as mx


def verify_batch(
    p: mx.array,
    q: mx.array,
    draft_tokens: mx.array,
    u_accept: mx.array,
    u_resample: mx.array,
) -> tuple[mx.array, mx.array]:
    b, k = draft_tokens.shape
    index = draft_tokens[:, :, None]
    p_x = mx.take_along_axis(p[:, :k], index, axis=2)[:, :, 0]  # (B, K): p[b, i, x_bi]
    q_x = mx.take_along_axis(q, index, axis=2)[:, :, 0]
    passes = u_accept * q_x < p_x
    # Per row, the accepted prefix ends at the first failed test: the running product
    # along the draft axis is 1 up to it and 0 afterwards.
    n_accepted = mx.sum(mx.cumprod(passes.astype(mx.int32), axis=1), axis=1)  # (B,)
    # Each row needs the residual of its own stopping position. A row of zeros for
    # q[:, K] makes the bonus case (weights p[b, K]) the same expression.
    q_ext = mx.concatenate([q, mx.zeros_like(q[:, :1])], axis=1)
    rows = mx.arange(b)
    weights = mx.maximum(p[rows, n_accepted] - q_ext[rows, n_accepted], 0.0)  # (B, V)
    # Inverse CDF per row, each with its own uniform and its own total.
    cdf = mx.cumsum(weights, axis=1)
    threshold = u_resample[:, None] * cdf[:, -1:]
    next_tokens = mx.sum(cdf <= threshold, axis=1)
    return n_accepted, next_tokens


def advance_batch(
    tokens: mx.array,
    lengths: mx.array,
    draft_tokens: mx.array,
    n_accepted: mx.array,
    next_tokens: mx.array,
    pad_id: int,
) -> tuple[mx.array, mx.array, mx.array]:
    cache_lengths = lengths + n_accepted  # tokens whose cache entries stay valid
    new_lengths = cache_lengths + 1  # the next token is appended, not yet in any cache
    width = int(mx.max(new_lengths).item())
    cols = mx.arange(width)[None, :]  # (1, W), compared with per-row boundaries (B, 1)
    start, stop = lengths[:, None], cache_lengths[:, None]
    # What each source would put in column c; indices are clamped so that every
    # gather is in range, and `where` picks the source that actually owns the column.
    old_index = mx.broadcast_to(mx.minimum(cols, tokens.shape[1] - 1), (len(lengths), width))
    old = mx.take_along_axis(tokens, old_index, axis=1)
    draft_index = mx.clip(cols - start, 0, draft_tokens.shape[1] - 1)
    drafts = mx.take_along_axis(draft_tokens, draft_index, axis=1)
    new_tokens = mx.full(old.shape, pad_id, dtype=old.dtype)
    new_tokens = mx.where(cols == stop, next_tokens[:, None], new_tokens)
    new_tokens = mx.where((cols >= start) & (cols < stop), drafts, new_tokens)
    new_tokens = mx.where(cols < start, old, new_tokens)
    return new_tokens, new_lengths, cache_lengths
