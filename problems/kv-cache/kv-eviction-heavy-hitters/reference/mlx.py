import mlx.core as mx

Cache = tuple[mx.array, mx.array]  # (k, v)


def h2o_step(
    cache: Cache,
    scores: mx.array | None,
    attn_weights: mx.array,
    budget: int,
    recent: int,
) -> tuple[Cache, mx.array, mx.array]:
    k, v = cache
    b, h, t, s = attn_weights.shape
    # What each cached token received in this step: the column sums over the new queries.
    received = attn_weights.sum(axis=2)
    if scores is None:
        scores = received
    else:
        # The T new tokens start at 0; the older ones keep what they have accumulated.
        scores = mx.concatenate((scores, mx.zeros((b, h, t))), axis=2) + received
    if s <= budget:
        kept = mx.broadcast_to(mx.arange(s), (b, h, s))
    else:
        old = s - recent  # positions 0 .. old - 1 compete; the last `recent` are safe
        # argsort is stable: sorting the negated scores ranks ties by position, oldest first.
        ranked = mx.argsort(-scores[:, :, :old], axis=-1)
        heavy = mx.sort(ranked[:, :, : budget - recent], axis=-1)  # back in time order
        window = mx.broadcast_to(mx.arange(old, s), (b, h, recent))
        kept = mx.concatenate((heavy, window), axis=-1)
    # Every head keeps its own tokens: gather along the sequence axis with per-head indices.
    index = mx.broadcast_to(kept[..., None], (b, h, kept.shape[-1], k.shape[-1]))
    cache = (mx.take_along_axis(k, index, axis=2), mx.take_along_axis(v, index, axis=2))
    return cache, mx.take_along_axis(scores, kept, axis=2), kept
